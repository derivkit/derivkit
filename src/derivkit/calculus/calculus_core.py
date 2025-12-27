"""Core utilities for calculus-based derivative computations.

This module provides shared helper functions for building derivative
objects (gradients, Jacobians, Hessians, higher-order tensors) of scalar-
and tensor-valued functions using DerivativeKit.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.utils.concurrency import (
    parallel_execute,
)
from derivkit.utils.validate import ensure_finite

__all__ = [
    "component_scalar_eval",
    "dispatch_tensor_output",
]


def component_scalar_eval(
    theta_vec: NDArray[np.floating],
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    idx: int,
) -> float:
    """Evaluates a single element of the model output ``function(theta_vec)`` as a scalar.

    This helper is used internally when building derivatives of models that
    return multiple outputs (e.g. vectors or arrays). Derivative routines
    operate on scalar-valued functions, so one output component is selected
    and treated as a scalar function of the parameters.

    Args:
        theta_vec: 1D parameter vector.
        function: Original function.
        idx: Flat (row-major) index into the model output ``y = function(theta_vec)`` after flattening it to 1D.

    Returns:
        Scalar value of the specified output component.
    """
    val = np.asarray(function(theta_vec))
    return float(val.ravel()[int(idx)])


def dispatch_tensor_output(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.floating],
    *,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    dk_kwargs: dict[str, Any],
    build_component: Callable[
        [
          int, NDArray[np.floating],
          str | None,
          int | None,
          dict[str, Any],
          Callable[[ArrayLike],
          float | np.ndarray]
        ],
        NDArray[np.floating],
    ],
) -> NDArray[np.float64]:
    """Computes per-output-component derivative objects for tensor-valued outputs and reshapes back.

    Pattern:
      - Evaluate ``y0 = function(theta)``
      - Flatten tensor output into m scalar components
      - For each component idx, compute a scalar-output derivative object
      - Stack and reshape to ``(*out_shape, *derivative_object_shape)``

    Args:
        function: Original function.
        theta: 1D parameter vector.
        method: Derivative method name or alias.
        outer_workers: Parallelism across output components.
        inner_workers: Parallelism forwarded to DerivativeKit inside each component calculation.
        dk_kwargs: Keyword args forwarded to DerivativeKit.differentiate (already stripped of ``inner_workers``).
        build_component: Per-component builder with signature:
            (idx, theta, method, inner_workers, dk_kwargs, function) -> ndarray

    Returns:
        Array with shape ``(*out_shape, *derivative_object_shape)``.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``function(theta)`` is scalar (use scalar path in caller).
    """
    y0 = np.asarray(function(theta))
    ensure_finite(y0, msg="Non-finite values in model output at theta0.")

    if y0.ndim == 0:
        raise ValueError(
            "dispatch_tensor_output is only for tensor outputs; handle scalar output in the caller."
        )

    out_shape = y0.shape
    m = int(y0.size)

    worker = partial(
        build_component,
        theta=theta,
        method=method,
        inner_workers=inner_workers,
        dk_kwargs=dk_kwargs,
        function=function,
    )

    vals = parallel_execute(
        worker,
        arg_tuples=[(i,) for i in range(m)],
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    arr = np.stack([np.asarray(v, dtype=np.float64) for v in vals], axis=0)
    arr = arr.reshape(out_shape + arr.shape[1:])

    ensure_finite(arr, msg="Non-finite values encountered in tensor-output derivative object.")
    return arr
