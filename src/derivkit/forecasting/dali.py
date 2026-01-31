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
    forecast_order: int | None = 2,
    single_forecast_order: bool = False,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> Union[
        NDArray[np.floating],
        tuple[NDArray[np.floating],...],
        dict[int, tuple[NDArray[np.floating],...]]
    ]:
    """Builds the DALI expansion for the given model of the supplied order.

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
        forecast_order: The requested order of the forecast.
            Currently supported values and their meaning are given in
            :data:`derivkit.forecasting.forecast_core.SUPPORTED_FORECAST_ORDERS`.
        single_forecast_order: If set to ``True``, the function will return only
            the requested order. If set to ``False``, the function will return
            the tensors up to the requested order.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial). Inner batch evaluation is kept serial to
            avoid oversubscription.
        dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.calculus_kit.CalculusKit`.

    Returns:
        If ``single_forecast_order`` is ``False`` the result is a dictionary
        with the keys equal to the order of the DALI expansion and values
        equal to the DALI multiplet at that order. For example, for
        ``forecast_order == 3`` the result is::

            {
                1: DALI_singlet,
                2: DALI_doublet,
                3: DALI_triplet,
            }

        where ``DALI_singlet`` contains the Fisher matrix, ``DALI_doublet``
        contains the DALI doublet tensors, etc.

        If ``single_forecast_order`` is ``True`` the result instead is the
        multiplet of the requested order. Using the example above the output
        would be ``DALI_triplet``. In the case that ``forecast_order == 1``
        the Fisher matrix itself is returned instead of ``DALI_singlet``.

        The number of axes of the tensors in the multiplet at order ``order``
        ranges from ``order+1`` to ``2*order`` in order of increasing number.
        All axes have length ``len(theta0)``.
    """
    return get_forecast_tensors(
        function,
        theta0,
        cov,
        method=method,
        forecast_order=forecast_order,
        single_forecast_order=single_forecast_order,
        n_workers=n_workers,
        **dk_kwargs,
)
