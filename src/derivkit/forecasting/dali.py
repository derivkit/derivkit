"""DALI forecasting utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from derivkit.forecasting.forecast_core import get_forecast_tensors
from derivkit.utils.types import Array, ArrayLike1D, ArrayLike2D, FloatArray

__all__ = [
    "build_dali",
]


def build_dali(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    *,
    method: str | None = None,
    forecast_order: int = 2,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> dict[int, tuple[FloatArray, ...]]:
    """Builds the DALI expansion for the given model of the supplied order.

    Args:
        function: The scalar or vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return either a scalar or a
            :class:`np.ndarray` of observable values.
        theta0: The expansion point (a 1D parameter vector) at which
            derivatives are evaluated. Accepts a list/array of length ``p``,
            with ``p`` the number of parameters.
        cov: The covariance matrix of the observables. Should be a square
            matrix with shape ``(n_observables, n_observables)``, where
            ``n_observables`` is the number of observables returned by the
            function.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default is used.
        forecast_order: The requested order of the forecast.
            Currently supported values and their meaning are given in
            :data:`derivkit.forecasting.forecast_core.SUPPORTED_FORECAST_ORDERS`.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial). Inner batch evaluation is kept serial to
            avoid oversubscription.
        **dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.calculus_kit.CalculusKit`.

    Returns:
        A dict mapping ``order -> multiplet`` for all ``order = 1..forecast_order``.

        For each forecast order k, the returned multiplet contains the tensors
        introduced at that order. Concretely:

        - order 1: ``(F_{(1,1)},)`` (Fisher matrix)
        - order 2: ``(D_{(2,1)}, D_{(2,2)})``
        - order 3: ``(T_{(3,1)}, T_{(3,2)}, T_{(3,3)})``

        Here ``D_{(k,l)}`` and ``T_{(k,l)}`` denote contractions of the
        ``k``-th and ``l``-th order derivatives via the inverse covariance.

        Each tensor axis has length ``p = len(theta0)``. The additional tensors at
        order ``k`` have parameter-axis ranks from ``k+1`` through ``2*k``.
    """
    return get_forecast_tensors(
        function,
        theta0,
        cov,
        method=method,
        forecast_order=forecast_order,
        n_workers=n_workers,
        **dk_kwargs,
    )
