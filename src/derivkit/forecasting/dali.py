"""DALI forecasting utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.forecasting.forecast_core import get_forecast_tensors

__all__ = [
    "build_dali",
]


def build_dali(
    function: Callable[[ArrayLike], np.floating | NDArray[np.floating]],
    theta0: ArrayLike,
    cov: ArrayLike,
    *,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Builds the doublet-DALI tensors (G, H) for the given model.

    Args:
        function: The scalar or vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return either a scalar or a
            :class:`np.ndarray` of observable values.
        theta0: The points at which the
            derivative is evaluated. A 1D array or list of parameter values
            matching the expected input of the function.
        cov: The covariance matrix of the observables. Should be a square
            matrix with shape ``(n_observables, n_observables)``, where
            ``n_observables`` is the number of observables returned by the
            function.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default is used.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial). Inner batch evaluation is kept serial to
            avoid oversubscription.
        dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.calculus_kit.CalculusKit`.

    Returns:
        A tuple ``(G, H)`` where ``G`` has shape ``(P, P, P)`` and ``H`` has
        shape ``(P, P, P, P)``, with ``P`` being the number of model parameters.
    """
    return get_forecast_tensors(
        function,
        theta0,
        cov,
        forecast_order=2,
        method=method,
        n_workers=n_workers,
        **dk_kwargs,
)
