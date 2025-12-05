"""Provides tools for facilitating experimental forecasts.

The user must specify the observables, fiducial values and covariance matrix
at which the derivative should be evaluated. Derivatives of the first order
are Fisher derivatives. Derivatives of second order are evaluated using the
derivative approximation for likelihoods (DALI) technique as described in
https://doi.org/10.1103/PhysRevD.107.103506.

More details about available options can be found in the documentation of
the methods.
"""

from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.linalg import invert_covariance, solve_or_pinv
from derivkit.forecasting import fisher, dali


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
            **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

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

        ckit = CalculusKit(self.function, self.theta0)

        # First-order path: compute Jacobian and return immediately
        if order == 1:
            j_raw = np.asarray(
                ckit.jacobian(
                    method=method,
                    n_workers=n_workers,  # allow outer parallelism across params
                    **dk_kwargs,
                ),
                dtype=float,
            )
            # Accept (N, P) or (P, N); return (P, N)
            if j_raw.shape == (self.n_observables, self.n_parameters):
                return j_raw.T
            elif j_raw.shape == (self.n_parameters, self.n_observables):
                return j_raw
            else:
                raise ValueError(
                    f"build_jacobian returned unexpected shape {j_raw.shape}; "
                    f"expected ({self.n_observables},{self.n_parameters}) or "
                    f"({self.n_parameters},{self.n_observables})."
                )

        # Second-order path (order is guaranteed to be 2 here)
        # Build Hessian tensor once (shape expected (n_observables, n_parameters, n_parameters)),
        # then return as (n_parameters, n_parameters, n_observables) for downstream einsum.
        h_raw = np.asarray(
            ckit.hessian(
                method=method,
                n_workers=n_workers,  # allow outer parallelism across params
                **dk_kwargs,
            ),
            dtype=float,
        )
        if h_raw.shape == (self.n_observables, self.n_parameters, self.n_parameters):
            return np.moveaxis(h_raw,[1,2],[0,1])
        elif h_raw.shape == (self.n_parameters, self.n_parameters, self.n_observables):
            return h_raw
        else:
            raise ValueError(
                f"build_hessian_tensor returned unexpected shape {h_raw.shape}; "
                f"expected ({self.n_observables},{self.n_parameters},{self.n_parameters}) or "
                f"({self.n_parameters},{self.n_parameters},{self.n_observables})."
            )


    def _build_fisher(self, d1, invcov):
        """Return the Fisher information matrix with shape (P, P) with P being the number of model parameters."""
        return fisher._build_fisher(self, d1, invcov)


    def _build_dali(
            self,
            d1: NDArray[np.float64],
            d2: NDArray[np.float64],
            invcov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the doublet-DALI tensors (G, H).

        Shapes are (P,P,P) and (P,P,P,P), where P is the number of model parameters.
        """
        return dali._build_dali(self, d1, d2, invcov)


    def build_fisher_bias(
            self,
            fisher_matrix: NDArray[np.floating],
            delta_nu: NDArray[np.floating],
            n_workers: int = 1,
            method: str | None = None,
            rcond: float = 1e-12,
            **dk_kwargs: Any,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the Fisher bias vector with shape (P,) with P being the number of model parameters."""
        return fisher.build_fisher_bias(
            self,
            fisher_matrix=fisher_matrix,
            delta_nu=delta_nu,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            **dk_kwargs,
        )

    def build_delta_nu(
            self,
            data_with: NDArray[np.floating],
            data_without: NDArray[np.floating],
            *,
            dtype: type | np.dtype = float,
    ) -> NDArray[np.floating]:
        """Return the delta_nu vector with shape (N,) with N being the number of observables."""
        return fisher.build_delta_nu(
            self,
            data_with=data_with,
            data_without=data_without
        )


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

