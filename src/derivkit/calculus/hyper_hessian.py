"""Construct third-derivative tensors ("hyper-Hessians") for scalar- or tensor-valued functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus.calculus_core import (
    component_scalar_eval,
    dispatch_tensor_output,
)
from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import resolve_inner_from_outer
from derivkit.utils.validate import ensure_finite

__all__ = [
    "build_hyper_hessian",
]


def build_hyper_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64] | Sequence[float],
    *,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns the third-derivative tensor ("hyper-Hessian") of a function.

    For scalar outputs, returns an array with shape ``(p, p, p)`` with ``p``
    the number of parameters.
    For tensor outputs with shape ``out_shape``, returns ``(*out_shape, p, p, p)``.

    Args:
        function: Function to differentiate.
        theta0: Parameter vector (1D).
        method: Derivative method name or alias. If ``None``, DerivativeKit default is used.
        n_workers: Outer parallelism across output components (tensor outputs only).
        **dk_kwargs: Extra keyword args forwarded to :meth:``DerivativeKit.differentiate``.
            You may pass ``inner_workers=<int>`` here to override inner parallelism.

    Returns:
        Third-derivative tensor as described above.

    Raises:
        ValueError: If ``theta0`` is empty.
        FloatingPointError: If non-finite values are encountered.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    y0 = np.asarray(function(theta))
    ensure_finite(y0, msg="Non-finite values in model output at theta0.")

    inner_override = dk_kwargs.pop("inner_workers", None)
    outer_workers = int(n_workers) if n_workers is not None else 1
    inner_workers = (
        int(inner_override)
        if inner_override is not None
        else resolve_inner_from_outer(outer_workers)
    )

    if y0.ndim == 0:
        out = _build_hyper_hessian_scalar(
            function=function,
            theta=theta,
            method=method,
            inner_workers=inner_workers,
            **dk_kwargs,
        )
        ensure_finite(out, msg="Non-finite values encountered in hyper-Hessian.")
        return out

    # Tensor output: compute per-component scalar hyper-Hessian and reshape it back.
    return dispatch_tensor_output(
        function=function,
        theta=theta,
        method=method,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
        dk_kwargs=dk_kwargs,
        build_component=_compute_component_hyper_hessian,
    )


def _build_hyper_hessian_scalar(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar-valued function.

    Args:
        function: Scalar-valued function to differentiate.
        theta: 1D parameter vector.
        method: Derivative method name or alias.
        inner_workers: Parallelism forwarded to DerivativeKit.
        **dk_kwargs: Keyword args forwarded to DerivativeKit.differentiate.

    Returns:
        Hyper-Hessian array with shape ``(p, p, p)`` with ``p`` the number of parameters.

    Raises:
        TypeError: If ``function`` is not scalar-valued.
        ValueError: If DerivativeKit returns an unexpected shape.
    """
    probe = np.asarray(function(theta), dtype=np.float64)
    if probe.ndim != 0:
        raise TypeError(
            "Scalar hyper-Hessian path expects a scalar-valued function; "
            f"got output with shape {probe.shape}."
        )

    kit = DerivativeKit(function, theta)
    out = kit.differentiate(
        order=3,
        method=method,
        n_workers=inner_workers or 1,
        **dk_kwargs,
    )

    arr = np.asarray(out, dtype=np.float64)

    p = int(theta.size)
    if arr.shape != (p, p, p):
        raise ValueError(
            "DerivativeKit returned unexpected hyper-Hessian shape "
            f"{arr.shape}; expected {(p, p, p)}."
        )

    return arr


def _compute_component_hyper_hessian(
    idx: int,
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict[str, Any],
    function: Callable[[ArrayLike], float | np.ndarray],
) -> NDArray[np.float64]:
    """Computes the hyper-Hessian for one output component of a tensor-valued function.

    Args:
        idx: Output component index.
        theta: 1D parameter vector.
        method: Derivative method name or alias.
        inner_workers: Parallelism forwarded to DerivativeKit.
        dk_kwargs: Keyword args forwarded to DerivativeKit.differentiate.
        function: Original tensor-valued function.

    Returns:
        Hyper-Hessian array with shape ``(p, p, p)`` with ``p`` the number of parameters.

    Raises:
        ValueError: If DerivativeKit returns an unexpected shape.
    """
    g = partial(component_scalar_eval, function=function, idx=int(idx))

    kit = DerivativeKit(g, theta)
    out = kit.differentiate(
        order=3,
        method=method,
        n_workers=inner_workers or 1,
        **dk_kwargs,
    )
    arr = np.asarray(out, dtype=np.float64)

    p = int(theta.size)
    if arr.shape != (p, p, p):
        raise ValueError(
            f"Component hyper-Hessian shape {arr.shape} does not match expected {(p, p, p)}."
        )
    return arr
