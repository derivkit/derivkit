"""Core utilities for calculus-based derivative computations.

This module provides shared helper functions for building derivative
objects (gradients, Jacobians, Hessians, higher-order tensors) of scalar-
and tensor-valued functions using DerivativeKit.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
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
    "cache_theta_function",
]


def component_scalar_eval(
    theta_vec: NDArray[np.float64],
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
        idx: Index of the output component to differentiate, where the output is first
            flattened with NumPy C-order (i.e. ``np.ravel(y, order="C")``).

    Returns:
        Scalar value of the specified output component.

    Raises:
        IndexError: If ``idx`` is out of bounds for the model output.
    """
    theta_vec = np.asarray(theta_vec, dtype=np.float64).reshape(-1)
    val = np.asarray(function(theta_vec))

    flat = np.ravel(val, order="C")
    i = int(idx)

    if i < 0 or i >= flat.size:
        raise IndexError(
            f"Output index {i} out of bounds for model output of size {flat.size}."
        )

    return float(flat[i])


def dispatch_tensor_output(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    *,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    dk_kwargs: dict[str, Any],
    build_component: Callable[
        [
          int, NDArray[np.float64],
          str | None,
          int | None,
          dict[str, Any],
          Callable[[ArrayLike],
          float | np.ndarray]
        ],
        NDArray[np.float64],
    ],
) -> NDArray[np.float64]:
    """Computes per-output-component derivative objects for tensor-valued outputs and reshapes back.

    This helper is intended for functions whose output has one or more dimensions
    (i.e. ``function(theta)`` returns an array). Scalar-valued functions should be
    handled by the scalar derivative routines (e.g. gradient, hessian, or derivative)
    and must not be routed through this dispatcher.

    The function uses the following strategy:

      1. Evaluate ``y0 = function(theta)``
      2. Flatten tensor output into m scalar components
      3. For each component ``idx``, compute a scalar-output derivative object
      4. Stack and reshape to ``(*out_shape, *derivative_object_shape)``

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
    theta = np.asarray(theta, dtype=np.float64)

    # NOTE: callers should prefer passing a cached wrapper for `function`.
    # We still need one evaluation here to discover out_shape/m.
    y0 = np.asarray(function(theta))
    ensure_finite(y0, msg="Non-finite values in model output at theta0.")

    if y0.ndim == 0:
        raise ValueError(
            "dispatch_tensor_output requires an array-valued model output (ndim >= 1), "
            f"but function(theta) returned a scalar (dtype={y0.dtype}). "
            "Use the scalar-output derivative path instead."
        )

    out_shape = y0.shape
    m = int(y0.size)

    # IMPORTANT: copy dk_kwargs per task (thread-safety + prevents accidental mutation leakage)
    tasks: list[tuple[int, NDArray[np.float64], str | None, int | None, dict[
        str, Any], Any]] = [
        (i, theta, method, inner_workers, dict(dk_kwargs), function) for i in
        range(m)
    ]

    vals = parallel_execute(
        build_component,
        arg_tuples=tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    arr = np.stack([np.asarray(v, dtype=np.float64) for v in vals], axis=0)
    arr = arr.reshape(out_shape + arr.shape[1:])

    ensure_finite(arr,
                  msg="Non-finite values encountered in tensor-output derivative object.")
    return arr


def cache_theta_function(
    function: Callable[[ArrayLike], float | np.ndarray],
) -> Callable[[NDArray[np.floating]], np.ndarray]:
    """Wrap `function(theta)` with a simple thread-safe cache.

    Keyed by the bytes of the 1D float64 theta vector. This avoids repeated
    expensive forward-model evaluations when the same theta is requested many
    times (common in tensor-output Hessians / hyper-Hessians and mixed partials).
    """
    cache: dict[bytes, np.ndarray] = {}
    lock = Lock()

    def wrapped(theta: NDArray[np.floating]) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1)
        key = t.tobytes()

        with lock:
            hit = cache.get(key)
        if hit is not None:
            return hit.copy()

        y = np.asarray(function(t))
        y_store = np.asarray(y).copy()  # protect cache from mutation

        with lock:
            cache[key] = y_store
        return y_store.copy()

    return wrapped
