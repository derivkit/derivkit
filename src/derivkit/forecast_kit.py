"""Provides the ForecastKit class.

A light wrapper around :class:`LikelihoodExpansion` that exposes a simple
API for Fisher and DALI tensors.

Typical usage example:

>>> forecaster = ForecastKit(function=model, theta0=theta0, cov=cov)
>>> F = forecaster.fisher()
>>> G, H = forecaster.dali()
"""

from collections.abc import Callable
from typing import Sequence

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

    def fisher(self, *, n_workers: int = 1):
        """Return the Fisher information matrix with shape (P, P) with P being the number of model parameters."""
        return self._lx.get_forecast_tensors(forecast_order=1, n_workers=n_workers)

    def dali(self, *, n_workers: int = 1):
        """Return the doublet-DALI tensors (G, H).

        Shapes are (P,P,P) and (P,P,P,P), where P is the number of model parameters.
        """
        return self._lx.get_forecast_tensors(forecast_order=2, n_workers=n_workers)
