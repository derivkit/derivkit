"""Provides the ForecastKit class.

A light wrapper around :class:`LikelihoodExpansion` that exposes a simple
API for Fisher and DALI tensors.

Typical usage example:

>>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
>>> fisher_matrix = fk.fisher(method="adaptive", n_workers=2)
>>> dali_g, dali_h = fk.dali(method="adaptive", n_workers=4)
>>> dn = fk.delta_nu(data_with=data_with_systematics, data_without=data_without_systematics)
>>> bias, dtheta = fk.fisher_bias(fisher_matrix=fisher_matrix, delta_nu=dn, method="finite")
"""

from collections.abc import Callable
from typing import Any, Sequence

import numpy as np

from derivkit.forecasting.dali import build_dali
from derivkit.forecasting.fisher import (
    build_delta_nu,
    build_fisher_bias,
    build_fisher_matrix,
)


class ForecastKit:
    """Provides access to Fisher and DALI likelihood-expansion tensors."""

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], np.ndarray],
        theta0: Sequence[float] | np.ndarray,
        cov: np.ndarray,
    ):
        """Initialises the forecaster with model, fiducials, and covariance.

        Args:
            function: Model mapping parameters to observables (1D array-like in, 1D array out).
            theta0: Fiducial parameter values (shape (P,)). Here, P is the number of
                    model parameters (P = len(theta0))
            cov: Observables covariance (shape (N, N)). N is the number of observables (N = cov.shape[0])
        """
        self.function = function
        self.theta0 = theta0
        self.cov = cov

    def fisher(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        """Computes the Fisher information matrix for a given model and covariance.

        Args:
            method: Derivative method name or alias (e.g., ``"adaptive"``, ``"finite"``).
                If ``None``, the DerivativeKit default is used.
            n_workers: Number of workers for per-parameter parallelisation. Default is 1 (serial).
            **dk_kwargs: Additional keyword arguments forwarded to
                :meth:`DerivativeKit.differentiate`.

        Returns:
            Fisher matrix with shape ``(n_parameters, n_parameters)``.
        """
        fisher_matrix = build_fisher_matrix(
            function=self.function,
            theta0=self.theta0,
            cov=self.cov,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )
        return fisher_matrix

    def fisher_bias(
        self,
        *,
        fisher_matrix: np.ndarray,
        delta_nu: np.ndarray,
        method: str | None = None,
        n_workers: int = 1,
        rcond: float = 1e-12,
        **dk_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Estimates parameter bias using the stored model, expansion point, and covariance.

        This method quantifies how differences between two data sets—typically an
        "unbiased" or reference data vector and a "biased" one including a given
        systematic—propagate into parameter biases when interpreted through a Fisher
        forecast. It evaluates the model response internally and uses it, together
        with the stored covariance and provided Fisher matrix, to estimate both the
        bias vector and the resulting shift in parameter values.
        For more information, see https://arxiv.org/abs/0710.5171.

        Args:
            fisher_matrix: Square matrix describing information about the parameters.
                Its shape must be (p, p), where p is the number of parameters.
            delta_nu: Difference between a "biased" and an "unbiased" data vector,
                for example :math:`\Delta\nu = \nu_{\mathrm{with\,sys}} - \nu_{\mathrm{without\,sys}}`.
                Accepts a 1D array of length n or a 2D array that will be flattened in
                row-major order (“C”) to length n, where n is the number of observables.
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
        bias = build_fisher_bias(
            function=self.function,
            theta0=self.theta0,
            cov=self.cov,
            fisher_matrix=fisher_matrix,
            delta_nu=delta_nu,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            **dk_kwargs,
        )
        return bias

    def delta_nu(self,
                 data_with: np.ndarray,
                 data_without: np.ndarray,
                 ):
        """Computes the difference between two data vectors.

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
        nu = build_delta_nu(
            cov=self.cov,
            data_with=data_with,
            data_without=data_without,
        )
        return nu

    def dali(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the doublet-DALI tensors (G, H) for the given model.

        Args:
            method: Method name or alias (e.g., "adaptive", "finite"). If None,
                the DerivativeKit default ("adaptive") is used.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default 1 (serial). Inner batch evaluation is kept serial to avoid
                oversubscription.
            dk_kwargs: Additional keyword arguments passed to the CalculusKit.

        Returns:
            A tuple (G, H) where G has shape (P, P, P) and H has shape (P, P, P, P),
            with P being the number of model parameters.
        """
        dali_tensors = build_dali(
            function=self.function,
            theta0=self.theta0,
            cov=self.cov,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )
        return dali_tensors
