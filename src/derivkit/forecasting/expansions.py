"""Provides tools for facilitating experimental forecasts.

The user must specify the observables, fiducial values and covariance matrix
at which the derivative should be evaluated. Derivatives of the first order
are Fisher derivatives. Derivatives of second order are evaluated using the
derivative approximation for likelihoods (DALI) technique as described in
https://doi.org/10.1103/PhysRevD.107.103506.

More details about available options can be found in the documentation of
the methods.
"""

from functools import partial
from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus.jacobian import build_jacobian
from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.linalg import invert_covariance, solve_or_pinv
from derivkit.utils.sandbox import get_partial_function


class LikelihoodExpansion:
    """Provides tools for facilitating experimental forecasts.

    Attributes:
         function: The scalar or vector-valued function to
             differentiate. It should accept a list or array of parameter
             values as input and return either a scalar or a
             :class:`np.ndarray` of observable values.
         theta0: The point(s) at which the
             derivative is evaluated. A 1D array or list of parameter values
             matching the expected input of the function.
         cov: The covariance matrix of
             the observables. Should be a square matrix with shape
             (n_observables, n_observables), where n_observables is the
             number of observables returned by the function.
         n_parameters: The number of elements of `theta0`.
         n_observables: The number of cosmic observables. Determined
             from the dimension of `cov`.
    """

    def __init__(
            self,
            function: Callable[[ArrayLike], float | NDArray[np.floating]],
            theta0: ArrayLike,
            cov: ArrayLike,
    ) -> None:
        """Initialises the class.

        Args:
            function: The scalar or vector-valued function to
                differentiate. It should accept a list or array of parameter
                values as input and return either a scalar or a
                :class:`np.ndarray` of observable values.
            theta0: The points at which the
                derivative is evaluated. A 1D array or list of parameter values
                matching the expected input of the function.
            cov: The covariance matrix of the observables. Should be a square
                matrix with shape (n_observables, n_observables), where n_observables
                is the number of observables returned by the function.

        Raises:
            ValueError: raised if cov is not a square numpy array.
        """
        self.function = function
        self.theta0 = np.atleast_1d(theta0)

        cov = np.asarray(cov)
        if cov.ndim > 2:
            raise ValueError(
                f"cov must be at most two-dimensional; got ndim={cov.ndim}."
            )
        if cov.ndim == 2 and cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square; got shape={cov.shape}.")

        self.cov = cov
        self.n_parameters = self.theta0.shape[0]
        self.n_observables = self.cov.shape[0]

    def get_forecast_tensors(
            self,
            forecast_order: int = 1,
            method: str | None = None,
            n_workers: int = 1,
            **dk_kwargs: Any,
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Returns a set of tensors according to the requested order of the forecast.

        Args:
            forecast_order: The requested order D of the forecast:

                    - D = 1 returns a Fisher matrix.
                    - D = 2 returns the 3-d and 4-d tensors required for the
                      doublet-DALI approximation.
                    - D = 3 would be the triplet-DALI approximation.

                Currently only D = 1, 2 are supported.
            method: Method name or alias (e.g., "adaptive", "finite"). If None,
                the DerivativeKit default ("adaptive") is used.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default 1 (serial). Inner batch evaluation is kept serial to avoid
                nested pools.
            **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

        Returns:
            If ``D = 1``: Fisher matrix of shape ``(P, P)``.
            If ``D = 2``: tuple ``(G, H)`` with shapes ``(P, P, P)`` and ``(P, P, P, P)``.

        Raises:
            ValueError: If `forecast_order` is not 1 or 2.

        Warns:
            RuntimeWarning: If `cov` is not symmetric (proceeds as-is, no symmetrization),
                is ill-conditioned (large condition number), or inversion
                falls back to the pseudoinverse.
        """
        if forecast_order not in [1, 2]:
            raise ValueError(
                "Only Fisher (order 1) and doublet-DALI (order 2) forecasts are currently supported."
            )

        # Check model output dimension
        y0 = np.atleast_1d(self.function(self.theta0))
        if y0.shape[0] != self.n_observables:
            raise ValueError(
                f"Expected {self.n_observables} observables from model (from cov {self.cov.shape}), "
                f"but got {y0.shape[0]} (output shape {y0.shape})."
            )

        # Compute inverse covariance matrix
        invcov = invert_covariance(self.cov, warn_prefix=self.__class__.__name__)
        # Compute first-order derivatives
        d1 = self._get_derivatives(order=1, n_workers=n_workers, method=method, **dk_kwargs)

        if forecast_order == 1:
            return self._build_fisher(d1, invcov)  # Fisher

        # Compute second-order derivatives
        d2 = self._get_derivatives(order=2, n_workers=n_workers, method=method, **dk_kwargs)
        return self._build_dali(d1, d2, invcov)  # doublet-DALI (G, H)

    def _get_derivatives(self,
                         order: int,
                         method: str | None = None,
                         n_workers: int = 1,
                         **dk_kwargs: Any,
                         ) -> NDArray[np.float64]:
        """Returns derivatives of the observables of the requested order.

        Args:
            order (int): The requested order d of the derivatives:

                - d = 1 returns first-order derivatives.
                - d = 2 returns second-order derivatives.

                Currently only d = 1, 2 are supported.

            method: Method name or alias (e.g., "adaptive", "finite"). If None,
                the DerivativeKit default ("adaptive") is used.
            n_workers (int, optional): Number of workers for per-parameter parallelization
             (threads). Default 1 (serial).
            **dk_kwargs (dict, optional): Additional keyword arguments passed to DerivativeKit.differentiate.

        Returns:
            :class:`np.ndarray`: An array of derivative values:

                - d = 1 returns an array with shape
                  (`n_parameters`, `n_observables`) containing
                  first-order derivatives.
                - d = 2 returns an array with shape
                  `n_parameters`, `n_parameters`, `n_observables`)
                  containing second-order derivatives.

        Raises:
            ValueError: An error occurred if a derivative was requested of
                higher order than 2.
            RuntimeError: An error occurred if a ValueError was not raised
                after calling the function.
        """
        if order not in (1, 2):
            raise ValueError("Only first- and second-order derivatives are currently supported.")

        n_workers = self._normalize_workers(n_workers)
        #dk_kwargs = dk_kwargs or {}

        # First-order path: compute Jacobian and return immediately
        if order == 1:
            j_raw = np.asarray(
                build_jacobian(
                    self.function,
                    self.theta0,
                    method=method,
                    n_workers=n_workers,  # allow outer parallelism across params
                    **dk_kwargs,
                ),
                dtype=float,
            )
            # Accept (N, P) or (P, N); return (P, N)
            if j_raw.shape == (self.n_observables, self.n_parameters):
                return j_raw.T
            if j_raw.shape == (self.n_parameters, self.n_observables):
                return j_raw
            raise ValueError(
                f"build_jacobian returned unexpected shape {j_raw.shape}; "
                f"expected ({self.n_observables},{self.n_parameters}) or "
                f"({self.n_parameters},{self.n_observables})."
            )

        # Second-order path (order is guaranteed to be 2 here)
        second_order = np.zeros((self.n_parameters, self.n_parameters, self.n_observables), dtype=float)

        inner_workers = resolve_inner_from_outer(n_workers)

        worker = partial(
            self._row_worker,
            method=method,
            inner_workers=inner_workers,
            **dk_kwargs,
        )

        rows = parallel_execute(
            worker=worker,
            arg_tuples=[(m1,) for m1 in range(self.n_parameters)],
            outer_workers=n_workers,
            inner_workers=inner_workers,
        )

        for m1, row in rows:
            second_order[m1, :, :] = row

        return second_order

    def _row_worker(
        self,
        m1: int,
        *,
        method: str | None,
        inner_workers: int,
        **dk_kwargs: Any,
    ) -> tuple[int, np.ndarray]:
        """Build one row of the second-order derivative tensor.

        For a fixed primary parameter index, this computes the second derivative
        columns against all parameters. Pure second derivatives are computed on
        the diagonal, and mixed second derivatives are computed off the diagonal.
        Work is organized so that it can be executed in parallel across rows.

        Args:
            m1: Index of the primary parameter for this row.
            method: Derivative method to use (for example, "adaptive" or "finite").
                If None, the DerivativeKit default is used.
            inner_workers: Number of workers used by the internal derivative calls.
            **dk_kwargs: Additional keyword arguments forwarded to
                `DerivativeKit.differentiate`.

        Returns:
            A tuple of the row index and a two-dimensional array with shape
            (number of parameters, number of observables) containing the second
            derivatives for parameter `m1` against every parameter.
        """
        row = np.zeros((self.n_parameters, self.n_observables), dtype=float)

        for m2 in range(self.n_parameters):
            if m1 == m2:
                row[m2] = self._pure_second_column(
                    m=m1,
                    method=method,
                    inner_workers=inner_workers,
                    **dk_kwargs,
                )
            else:
                row[m2] = self._mixed_second_column(
                    m1=m1,
                    m2=m2,
                    method=method,
                    inner_workers=inner_workers,
                    **dk_kwargs,
                )

        return m1, row

    def _pure_second_column(
        self,
        m: int,
        *,
        method: str | None,
        inner_workers: int,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        """Compute the second derivative with respect to one parameter.

        Builds a single-variable view of the model where only parameter `m` varies
        and all other parameters are fixed at the expansion point. It then uses
        DerivativeKit to evaluate the second derivative of each observable with
        respect to that parameter.

        Args:
            m: Index of the parameter to differentiate with respect to.
            method: Derivative method to use (for example, "adaptive" or "finite").
                If None, the DerivativeKit default is used.
            inner_workers: Number of workers for the internal derivative step.
            **dk_kwargs: Additional keyword arguments forwarded to
                `DerivativeKit.differentiate`.

        Returns:
            A one-dimensional array containing the second derivative of each
            observable with respect to parameter `m` at the expansion point.
        """
        theta_fix = self.theta0.copy()
        f1 = get_partial_function(self.function, m, theta_fix)
        kit1 = DerivativeKit(f1, float(self.theta0[m]))
        return kit1.differentiate(order=2, method=method, n_workers=inner_workers, **dk_kwargs)

    def _mixed_second_column(
            self,
            m1: int,
            m2: int,
            *,
            method: str | None,
            inner_workers: int,
            **dk_kwargs: Any,
    ) -> np.ndarray:
        """Compute the mixed second derivative for two parameters.

        Creates a single-argument path function that, for each trial value of the
        second parameter, evaluates the first derivative of the model with respect
        to the first parameter while holding all other parameters fixed at the
        expansion point. It then differentiates that path with respect to the
        second parameter to obtain the mixed second derivative.

        Args:
            m1: Index of the first parameter in the mixed derivative.
            m2: Index of the second parameter in the mixed derivative.
            method: Derivative method to use (for example, "adaptive" or "finite").
                If None, the DerivativeKit default is used.
            inner_workers: Number of workers for the internal derivative step.
            **dk_kwargs: Additional keyword arguments forwarded to
                `DerivativeKit.differentiate`.

        Returns:
            A one-dimensional array containing the mixed second derivative of each
            observable with respect to parameters `m1` and `m2` at the expansion point.
        """
        path = partial(
            mixed_partial_path,  # <-- remove leading underscore
            function=self.function,
            theta0=self.theta0,
            i=m1,
            j=m2,
            method=method,
            **dk_kwargs,
        )
        kit2 = DerivativeKit(path, float(self.theta0[m2]))
        return kit2.differentiate(order=1, method=method, n_workers=inner_workers, **dk_kwargs)

    def _build_fisher(self, d1, invcov):
        """Assemble the Fisher information matrix F from first derivatives.

        Args:
            d1 (np.ndarray): First-order derivatives of observables w.r.t. parameters,
                shape (n_parameters, n_observables).
            invcov (np.ndarray): Inverse covariance of observables,
                shape (n_observables, n_observables).

        Returns:
            np.ndarray: Fisher matrix, shape (n_parameters, n_parameters).

        Notes:
            Uses `np.einsum("ai,ij,bj->ab", d1, invcov, d1)`.
        """
        # F_ab = Σ_ij d1[a,i] invcov[i,j] d1[b,j]
        return np.einsum("ai,ij,bj->ab", d1, invcov, d1)


    def _build_dali(
            self,
            d1: NDArray[np.float64],
            d2: NDArray[np.float64],
            invcov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Assemble the doublet-DALI tensors (G, H) from first- and second-order derivatives.

        Computes:
            G_abc = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d1[c,j]
            H_abcd = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d2[c,d,j]

        Args:
            d1: First-order derivatives of the observables with respect to parameters,
                shape (P, N).
            d2: Second-order derivatives of the observables with respect to parameters,
                shape (P, P, N).
            invcov: Inverse covariance matrix of the observables, shape (N, N).

        Returns:
            A tuple ``(G, H)`` where:
                - G has shape (P, P, P)
                - H has shape (P, P, P, P)
        """
        # G_abc = Σ_ij d2[a,b,i] invcov[i,j] d1[c,j]
        g_tensor = np.einsum("abi,ij,cj->abc", d2, invcov, d1)
        # H_abcd = Σ_ij d2[a,b,i] invcov[i,j] d2[c,d,j]
        h_tensor = np.einsum("abi,ij,cdj->abcd", d2, invcov, d2)
        return g_tensor, h_tensor


    def build_fisher_bias(
            self,
            fisher_matrix: NDArray[np.floating],
            delta_nu: NDArray[np.floating],
            n_workers: int = 1,
            method: str | None = None,
            rcond: float = 1e-12,
            **dk_kwargs: Any,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Estimate parameter bias using the stored model, expansion point, and covariance.

        This method quantifies how differences between two data sets (for example,
        a fiducial prediction and one affected by a systematic) propagate into
        parameter biases when interpreted through a Fisher forecast. It evaluates
        the model response internally and uses it, together with the stored
        covariance and provided Fisher matrix, to estimate both the bias vector
        and the resulting shift in parameter values.
        For more information, see https://arxiv.org/abs/0710.5171.

        Args:
          fisher_matrix: Square matrix describing information about the parameters.
            Its shape must be (p, p), where p is the number of parameters.
          delta_nu: Difference between two data vectors (for example, with and without
            a systematic). Accepts a 1D array of length n or a 2D array that will be
            flattened in row-major order (“C”) to length n, where n is the number of observables.
            If supplied as a 1D array, it must already follow the same row-major (“C”)
            flattening convention used throughout the package.
          n_workers: Number of workers used by the internal derivative routine when
            forming the Jacobian.
          method: Method name or alias (e.g., "adaptive", "finite").
            If None, the DerivativeKit default ("adaptive") is used.
          **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.
          rcond: Regularization cutoff for pseudoinverse. Default is 1e-12.

        Returns:
          A tuple ``(bias_vec, delta_theta)`` where both entries are 1D arrays of length ``p``:
            - bias_vec: parameter-space bias vector.
            - delta_theta: estimated parameter shifts.

        Raises:
          ValueError: If input shapes are inconsistent with the stored model, covariance,
            or the Fisher matrix dimensions.
          FloatingPointError: If the difference vector contains NaNs.
        """
        n_workers = self._normalize_workers(n_workers)

        fisher_matrix = np.asarray(fisher_matrix, dtype=float)
        if fisher_matrix.ndim != 2 or fisher_matrix.shape[0] != fisher_matrix.shape[1]:
            raise ValueError(f"fisher_matrix must be square; got shape {fisher_matrix.shape}.")

        # Jacobian — we are enforcing (n_obs, n_params) throughout the package
        j_matrix = np.asarray(
            build_jacobian(
                self.function,
                self.theta0,
                method=method,
                n_workers=n_workers,
                **dk_kwargs,
            ),
            dtype=float,
        )
        n_obs, n_params = self.n_observables, self.n_parameters
        if j_matrix.shape != (n_obs, n_params):
            raise ValueError(
                f"build_jacobian must return shape (n_obs, n_params)=({n_obs},{n_params}); "
                f"got {j_matrix.shape}."
            )

        # Shape checks consistent with J
        if self.cov.shape != (j_matrix.shape[0], j_matrix.shape[0]):
            raise ValueError(
                f"covariance shape {self.cov.shape} must be (n, n) = "
                f"{(j_matrix.shape[0], j_matrix.shape[0])} from the Jacobian."
            )
        if fisher_matrix.shape != (j_matrix.shape[1], j_matrix.shape[1]):
            raise ValueError(
                f"fisher_matrix shape {fisher_matrix.shape} must be (p, p) = "
                f"{(j_matrix.shape[1], j_matrix.shape[1])} from the Jacobian."
            )

        # Make delta_nu a 1D array of length n; 2D inputs are flattened in row-major ("C") order.
        delta_nu = np.asarray(delta_nu, dtype=float)
        if delta_nu.ndim == 2:
            delta_nu = delta_nu.ravel(order="C")
        if delta_nu.ndim != 1 or delta_nu.size != n_obs:
            raise ValueError(f"delta_nu must have length n={n_obs}; got shape {delta_nu.shape}.")
        if not np.isfinite(delta_nu).all():
            raise FloatingPointError("Non-finite values found in delta_nu.")

        # GLS weighting by the inverse covariance:
        # If C is diagonal, compute invcov * delta_nu by elementwise division (fast).
        # Otherwise solve with a symmetric solver; on ill-conditioning/failure,
        # fall back to a pseudoinverse and emit a warning.
        off = self.cov.copy()
        np.fill_diagonal(off, 0.0)
        is_diag = not np.any(off)  # True iff all off-diagonals are exactly zero

        if is_diag:
            diag = np.diag(self.cov)
            if np.all(diag > 0):
                cinv_delta = delta_nu / diag
            else:
                cinv_delta = solve_or_pinv(
                    self.cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
                )
        else:
            cinv_delta = solve_or_pinv(
                self.cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
            )

        bias_vec = j_matrix.T @ cinv_delta
        delta_theta = solve_or_pinv(
            fisher_matrix, bias_vec, rcond=rcond, assume_symmetric=True, warn_context="Fisher solve"
        )

        return bias_vec, delta_theta

    def build_delta_nu(
            self,
            data_with: NDArray[np.floating],
            data_without: NDArray[np.floating],
            *,
            dtype: type | np.dtype = float,
    ) -> NDArray[np.floating]:
        """Compute the difference between two data vectors.

        This function is typically used for Fisher-bias estimates, taking two data vectors—
        one with a systematic included and one without—and returning their difference as a
        1D array that matches the expected number of observables in this instance. It works
        with both 1D inputs and 2D arrays (for example, correlation × ell) and flattens 2D
        arrays using NumPy's row-major ("C") order, our standard convention throughout the package.

        We standardize on row-major (“C”) flattening of 2D arrays, where the last
        axis varies fastest. The user must ensure that any data vectors and associated covariances
        are constructed with the same convention for consistent results.

        Args:
          data_with: Data vector that includes the systematic effect. Can be 1D or 2D.
            If 1D, it must follow the NumPy's row-major (“C”) flattening convention used
            throughout the package.
          data_without: Reference data vector without the systematic. Can be 1D or 2D. If 1D,
            it must follow the NumPy's row-major (“C”) flattening convention used throughout
            the package.
          dtype: Data type of the output array (defaults to float).

        Returns:
          A 1D NumPy array of length ``self.n_observables`` representing the data
          mismatch (delta_nu = data_with − data_without).

        Raises:
          ValueError: If input shapes differ, inputs are not 1D/2D, or the flattened
            length does not match ``self.n_observables``.
          FloatingPointError: If non-finite values are detected in the result.
        """
        # define flattening orders for numpy
        a = np.asarray(data_with, dtype=dtype)
        b = np.asarray(data_without, dtype=dtype)

        if a.shape != b.shape:
            raise ValueError(f"Shapes must match: got {a.shape} vs {b.shape}.")

        if a.ndim == 1:
            delta_nu = a - b
        elif a.ndim == 2:
            delta_nu = (a - b).ravel(order="C")
        else:
            raise ValueError(f"Only 1D or 2D inputs are supported; got ndim={a.ndim}.")

        if delta_nu.size != self.n_observables:
            raise ValueError(
                f"Flattened length {delta_nu.size} != expected self.n_observables {self.n_observables}."
            )

        if not np.isfinite(delta_nu).all():
            raise FloatingPointError("Non-finite values found in delta vector.")

        return delta_nu


    def _normalize_workers(self, n_workers):
        """Ensure n_workers is a positive integer, defaulting to 1.

        Args:
            n_workers: Input number of workers (can be None, float, negative, etc.)

        Returns:
            int: A positive integer number of workers (at least 1).

        Raises:
            None: Invalid inputs are coerced to 1.
        """
        try:
            n = int(n_workers)
        except (TypeError, ValueError):
            n = 1
        return 1 if n < 1 else n


def mixed_partial_path(
    y: float,
    *,
    function: Callable,
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Compute a first derivative while temporarily fixing another parameter.

    This helper builds a single-argument path where parameter ``j`` is set to
    the provided value and all other parameters are held at the expansion
    point. Along that path it evaluates the first derivative of the model with
    respect to parameter ``i``. It is used to construct mixed second
    derivatives without nested function definitions.

    Args:
      y: Value to assign to parameter ``j`` along the evaluation path.
      function: Model function that returns a vector of observables given a
        parameter vector.
      theta0: Parameter vector at the expansion point.
      i: Index of the parameter to differentiate with respect to.
      j: Index of the parameter that is temporarily set to ``y``.
      method: Derivative method to use, such as "adaptive" or "finite". If
        None, the DerivativeKit default is used.
      **dk_kwargs: Extra keyword arguments forwarded to
        DerivativeKit.differentiate.

    Returns:
      A one-dimensional array with the first derivative of each observable
      with respect to parameter ``i`` evaluated at the path where parameter
      ``j`` equals ``y``.
    """
    theta_fix = theta0.copy()
    theta_fix[j] = float(y)
    f1 = get_partial_function(function, i, theta_fix)
    kit = DerivativeKit(f1, float(theta_fix[i]))
    return kit.differentiate(order=1, method=method, **dk_kwargs)
