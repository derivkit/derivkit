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

from derivkit.forecasting.expansions import LikelihoodExpansion


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
        self._lx = LikelihoodExpansion(function, theta0, cov)

    def fisher(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        """Return the Fisher information matrix with shape (P, P) with P being the number of model parameters."""
        return self._lx.get_forecast_tensors(
            forecast_order=1,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )

    def dali(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the doublet-DALI tensors (G, H).

        Shapes are (P,P,P) and (P,P,P,P), where P is the number of model parameters.
        """
        return self._lx.get_forecast_tensors(
            forecast_order=2,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )

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
        """Return the Fisher bias vector with shape (P,) with P being the number of model parameters."""
        return self._lx.build_fisher_bias(
            fisher_matrix=fisher_matrix,
            delta_nu=delta_nu,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            **dk_kwargs,
        )

    def delta_nu(self,
                 data_with: np.ndarray,
                 data_without: np.ndarray,
                 ):
        """Return the delta_nu vector with shape (N,) with N being the number of observables."""
        return self._lx.build_delta_nu(
            data_with=data_with,
            data_without=data_without
        )
