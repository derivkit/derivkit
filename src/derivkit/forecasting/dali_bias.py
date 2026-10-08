"""DALI systematic-bias tensor utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from derivkit.forecasting.forecast_core import (
    SUPPORTED_DERIVATIVE_ORDERS,
    _get_derivatives,
)
from derivkit.utils.linalg import invert_covariance
from derivkit.utils.types import Array, ArrayLike1D, ArrayLike2D, FloatArray
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "build_dali_bias",
]


def build_dali_bias(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D,
    *,
    bias_order: int = 3,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> dict[int, FloatArray]:
    """Builds the systematic mismatch tensors entering the DALI bias expansion.

    The tensors describe how the data-model mismatch ``delta_nu``
    couples to successive derivatives of the model.

    Args:
        function: The scalar or vector-valued model function.
        theta0: Fiducial parameter values at which derivatives are evaluated.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between two data vectors, which may represent
            a systematic mismatch or a difference between model predictions.
        bias_order: Highest mismatch tensor order to compute. Supported values
            are given in
            :data:`derivkit.forecasting.forecast_core.SUPPORTED_DERIVATIVE_ORDERS`.
        method: Numerical differentiation method. If ``None``, the
            :class:`derivkit.derivative_kit.DerivativeKit` default is used.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial).
        **dk_kwargs: Additional keyword arguments passed to DerivKit's
            differentiation machinery.

    Returns:
        A dictionary mapping each derivative order to its systematic mismatch
        tensor.

    Raises:
        TypeError: If ``bias_order`` cannot be converted to an integer.
        ValueError: If ``bias_order`` is unsupported, ``theta0`` is empty, or
            the data mismatch does not match the number of observables.
    """
    try:
        bias_order = int(bias_order)
    except Exception as e:
        raise TypeError(
            f"bias_order must be an int; got {type(bias_order)}."
        ) from e

    if bias_order not in SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            f"bias_order={bias_order} is not supported. "
            f"Supported values: {SUPPORTED_DERIVATIVE_ORDERS}."
        )

    theta0_arr = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError("theta0 must be non-empty 1D.")

    cov_arr = validate_covariance_matrix_shape(cov)
    n_observables = cov_arr.shape[0]

    delta_nu_arr = np.asarray(delta_nu, dtype=float).reshape(-1)
    if delta_nu_arr.size != n_observables:
        raise ValueError(
            f"Expected {n_observables} elements in delta_nu "
            f"(from cov {cov_arr.shape}), but got {delta_nu_arr.size}."
        )

    y0 = np.asarray(function(theta0_arr), dtype=float)
    y0_flat = y0.reshape(-1)

    if y0_flat.size != n_observables:
        raise ValueError(
            f"Expected {n_observables} observables from model "
            f"(from cov {cov_arr.shape}), "
            f"but got {y0_flat.size} (output shape {y0.shape})."
        )

    inv_cov = invert_covariance(cov_arr, warn_prefix="build_dali_bias")

    bias: dict[int, FloatArray] = {}

    for order in range(1, bias_order + 1):
        derivative = _get_derivatives(
            function,
            theta0_arr,
            cov_arr,
            order=order,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )

        bias[order] = np.einsum(
            "i...,ij,j->...",
            derivative,
            inv_cov,
            delta_nu_arr,
        ).astype(np.float64, copy=False)

    return bias
