"""Provides tools for facilitating experimental forecasts.

The user must specify the observables, fiducial values and covariance matrix
at which the derivative should be evaluated. Derivatives of the first order
are Fisher derivatives. Derivatives of second order are evaluated using the
derivative approximation for likelihoods (DALI) technique as described in
https://doi.org/10.1103/PhysRevD.107.103506.

More details about available options can be found in the documentation of
the methods.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.forecasting.calculus import jacobian
from derivkit.utils import (
    get_partial_function,
    invert_covariance,
    solve_or_pinv,
)


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
            n_workers: int = 1,
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Returns a set of tensors according to the requested order of the forecast.

        Args:
            forecast_order: The requested order D of the forecast:

                    - D = 1 returns a Fisher matrix.
                    - D = 2 returns the 3-d and 4-d tensors required for the
                      doublet-DALI approximation.
                    - D = 3 would be the triplet-DALI approximation.

                Currently only D = 1, 2 are supported.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default 1 (serial). Inner batch evaluation is kept serial to avoid
                nested pools.

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
        d1 = self._get_derivatives(order=1, n_workers=n_workers)

        if forecast_order == 1:
            return self._build_fisher(d1, invcov)  # Fisher

        # Compute second-order derivatives
        d2 = self._get_derivatives(order=2, n_workers=n_workers)
        return self._build_dali(d1, d2, invcov)  # doublet-DALI (G, H)

    def _get_derivatives(self, order, n_workers=1):
        """Returns derivatives of the observables of the requested order.

        Args:
            order (int): The requested order d of the derivatives:

                - d = 1 returns first-order derivatives.
                - d = 2 returns second-order derivatives.

                Currently only d = 1, 2 are supported.

            n_workers (int, optional): Number of workers for per-parameter parallelization
             (threads). Default 1 (serial).

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
        if order not in [1, 2]:
            raise ValueError(
                "Only first- and second-order derivatives are currently supported."
            )

        n_workers = self._normalize_workers(n_workers)
        inner_workers = 1 if n_workers > 1 else 1  # keep inner serial; safest

        if order == 1:
            first_order_derivatives = np.zeros(
                (self.n_parameters, self.n_observables), dtype=float
            )

            def compute_m(m: int) -> np.ndarray:
                theta0_x = deepcopy(self.theta0)
                f_to_diff = get_partial_function(self.function, m, theta0_x)
                kit = DerivativeKit(f_to_diff, self.theta0[m])
                # still pass n_workers through to adaptive (it can batch-eval)
                return kit.adaptive.differentiate(order=1, n_workers=inner_workers)

            results = self._map_threads(compute_m, range(self.n_parameters), n_workers)
            for m, val in enumerate(results):
                first_order_derivatives[m] = val
            return first_order_derivatives

        elif order == 2:
            second_order_derivatives = np.zeros(
                (self.n_parameters, self.n_parameters, self.n_observables), dtype=float
            )

            def compute_row(m1: int) -> tuple[int, np.ndarray]:
                row = np.zeros((self.n_parameters, self.n_observables), dtype=float)
                for m2 in range(self.n_parameters):
                    if m1 == m2:
                        theta0_x = deepcopy(self.theta0)
                        f1 = get_partial_function(self.function, m1, theta0_x)
                        kit1 = DerivativeKit(f1, self.theta0[m1])
                        row[m2] = kit1.adaptive.differentiate(order=2, n_workers=inner_workers)
                    else:
                        def f2(y):
                            theta0_y = deepcopy(self.theta0)
                            theta0_y[m2] = y
                            f1_inner = get_partial_function(self.function, m1, theta0_y)
                            kit1_inner = DerivativeKit(f1_inner, self.theta0[m1])
                            return kit1_inner.adaptive.differentiate(order=1)

                        kit2 = DerivativeKit(f2, self.theta0[m2])
                        row[m2] = kit2.adaptive.differentiate(order=1, n_workers=inner_workers)
                return m1, row

            rows = self._map_threads(compute_row, range(self.n_parameters), n_workers)
            for m1, row in rows:
                second_order_derivatives[m1, :, :] = row
            return second_order_derivatives

        raise RuntimeError("Unreachable code reached in get_forecast_tensors.")

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
            rcond: float = 1e-12,
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
          rcond: Regularization cutoff for pseudoinverse.

        Returns:
          A tuple ``(bias_vec, delta_theta)`` where both entries are 1D arrays of length ``p``:
            - bias_vec: parameter-space bias vector.
            - delta_theta: estimated parameter shifts.

        Raises:
          ValueError: If input shapes are inconsistent with the stored model, covariance,
            or the Fisher matrix dimensions.
          FloatingPointError: If the difference vector contains NaNs.
        """
        fisher_matrix = np.asarray(fisher_matrix, dtype=float)
        if fisher_matrix.ndim != 2 or fisher_matrix.shape[0] != fisher_matrix.shape[1]:
            raise ValueError(f"fisher_matrix must be square; got shape {fisher_matrix.shape}.")

        # Jacobian matrix has shape (n, p)
        j_matrix = jacobian(self.function, self.theta0, n_workers=n_workers)
        n_obs, n_params = j_matrix.shape

        # Check shapes of the covariance and Fisher matrices against the Jacobian
        if self.cov.shape != (n_obs, n_obs):
            raise ValueError(
                f"covariance shape {self.cov.shape} must be (n, n) = {(n_obs, n_obs)} from the Jacobian."
            )
        if fisher_matrix.shape != (n_params, n_params):
            raise ValueError(
                f"fisher_matrix shape {fisher_matrix.shape} must be (p, p) = {(n_params, n_params)} from the Jacobian."
            )

        # Make delta_nu a 1D array of length n; 2D inputs are flattened in row-major ("C") order.
        delta_nu = np.asarray(delta_nu, dtype=float)
        if delta_nu.ndim == 2:
            delta_nu = delta_nu.ravel(order="C")
        if delta_nu.ndim != 1 or delta_nu.size != n_obs:
            raise ValueError(f"delta_nu must have length n={n_obs}; got shape {delta_nu.shape}.")

        if not np.isfinite(delta_nu).all():
            raise FloatingPointError("Non-finite values found in delta_nu.")

        cinv_delta = solve_or_pinv(
            self.cov,
            delta_nu,
            rcond=rcond,
            assume_symmetric=True,
            warn_context="covariance solve",
        )

        bias_vec = j_matrix.T @ cinv_delta

        delta_theta = solve_or_pinv(
            fisher_matrix,
            bias_vec,
            rcond=rcond,
            assume_symmetric=True,
            warn_context="Fisher solve",
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

    def _map_threads(self, fn, tasks, n_workers):
        """Map a function over tasks using threads.

        Args:
            fn: Function to apply to each task.
            tasks: Iterable of tasks to process.
            n_workers: Number of worker threads to use.

        Returns:
            List of results from applying fn to each task.

        Raises:
            None: Invalid n_workers values are coerced to 1.
        """
        n_workers = self._normalize_workers(n_workers)
        if n_workers <= 1:
            return [fn(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(fn, t) for t in tasks]
            return [f.result() for f in futs]

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
