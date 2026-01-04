"""Batch evaluation utilities for derivative estimation.

Evaluate a user function over a 1D grid with optional parallelism and return
a 2D array with consistent shape suitable for downstream polynomial fitting
and diagnostics (e.g., in :class:`adaptive.adaptive_fit.AdaptiveFitDerivative`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from multiprocess import Pool

__all__ = ["eval_function_batch"]


def eval_function_batch(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int = 1,
) -> np.ndarray:
    """Evaluate a function over 1D inputs and return a (n_points, n_comp) float array.

    Evaluates ``function(x)`` for each ``x`` in ``xs``. If ``n_workers > 1``,
    uses a ``multiprocess.Pool``; otherwise runs serially. Scalars become a single
    column. For array outputs, this routine coerces to a consistent 2D shape that
    downstream polynomial-fitting code expects.

    Args:
      function: Callable mapping a float to a scalar or array-like. Must be
        picklable if used with multiple processes.
      xs: 1D array of abscissae.
      n_workers: If > 1, evaluate in parallel using ``multiprocess.Pool``.

    Returns:
      np.ndarray: Array of shape ``(n_points, n_comp)`` with dtype ``float``.

    Raises:
      ValueError: If ``xs`` is not 1D or outputs cannot be coerced consistently.

    Examples:
      >>> import numpy as np
      >>> from derivkit.adaptive.batch_eval import eval_function_batch
      >>> def f(x):
      ...     return np.array([x, x**2])
      >>> xs = np.linspace(-1.0, 1.0, 5)
      >>> y = eval_function_batch(f, xs)
      >>> y.shape
      (5, 2)
    """
    xs = np.asarray(xs, dtype=float)
    if xs.ndim != 1:
        raise ValueError(
            f"eval_function_batch: xs.ndim must be 1 but is {xs.ndim}."
        )

    ys = (
        _eval_parallel(function, xs, n_workers)
        if n_workers > 1
        else _eval_serial(function, xs)
    )

    # Convert outputs to a consistent 2D float array (n_points × n_outputs).
    y = _coerce_stack(ys, n_points=xs.size)

    if not np.all(np.isfinite(y)):
        pass

    return y


def _eval_serial(
    function: Callable[[float], Any], xs: np.ndarray
) -> list[np.ndarray]:
    """Evaluate a function over points serially.

    Args:
      function: Callable mapping a float to a scalar or array-like.
      xs: 1D array of x-axis points to evaluate.

    Returns:
      list[np.ndarray]: One array per input x (each at least 1D).
    """
    return [np.atleast_1d(function(float(x))) for x in xs]


def _eval_parallel(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int,
) -> list[np.ndarray]:
    """Evaluate a function over points in parallel using multiprocess.Pool.

    Falls back to the serial path for tiny workloads or if pool creation/execution fails.

    Args:
      function: Maps a float to a scalar or array-like.
      xs: 1D points on x-axis to evaluate.
      n_workers: Desired number of processes.

    Returns:
      list[np.ndarray]: One 1D array per input, order-preserving.
    """
    if n_workers <= 1:
        return _eval_serial(function, xs)

    # Avoid pool overhead for very small batches.
    n = max(1, min(int(n_workers), int(xs.size)))
    if xs.size < max(8, 2 * n):
        return _eval_serial(function, xs)

    try:
        with Pool(n) as pool:
            ys = pool.map(function, xs.tolist())
    except Exception:
        # Spawn/pickle/start-method issues → graceful serial fallback.
        return _eval_serial(function, xs)

    return [np.atleast_1d(y) for y in ys]


def _coerce_stack(ys: list[np.ndarray], n_points: int) -> np.ndarray:
    """Coerce a list of per-point outputs into an (n_points, n_comp) float array.

    The user function may return scalars or arrays of varying shapes. This routine
    coerces them into a consistent 2D array shape that downstream code expects. The rules are:
        - scalar → column vector
        - 1D → row
        - 2D with a transposed batch (n_comp, n_points) → auto-transpose
        - higher-D → flattened per row

    Args:
        ys: List of arrays, one per input point. Each array is at least 1
            dimensional (scalars become shape (1,)).
        n_points: Number of input points (length of ys).

    Returns:
        np.ndarray: Array of shape (n_points, n_comp) with dtype float.

    Raises:
        ValueError: If outputs cannot be coerced to a consistent shape.
    """
    arr = np.asarray(ys, dtype=float)

    # Common cases fast-path
    if arr.ndim == 1:
        # all scalars
        return arr.reshape(n_points, 1)
    if arr.ndim == 2 and arr.shape[0] == n_points:
        # already (n_points, n_comp)
        return arr
    if arr.ndim == 2 and arr.shape[1] == n_points:
        # likely (n_comp, n_points) → transpose
        return arr.T

    # Fallback: stack row-wise and flatten per sample if needed.
    rows = []
    for y in ys:
        y = np.asarray(y, dtype=float)
        if y.ndim == 0:
            y = y.reshape(1)
        elif y.ndim >= 2:
            y = y.reshape(-1)  # flatten higher-D deterministically
        rows.append(y)
    y = np.vstack(rows)

    # Ensure (n_points, n_comp)
    if y.shape[0] != n_points:
        # If the function accidentally returned (n_comp, n_points), fix it once more.
        if y.shape[1] == n_points and y.shape[0] != n_points:
            y = y.T
        else:
            raise ValueError(
                f"eval_function_batch: cannot coerce outputs to (n_points, n_comp); "
                f"got shape {y.shape} for n_points={n_points}"
            )
    return y
