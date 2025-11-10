"""Batch evaluation utilities for finite-difference methods."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)

__all__ = ["eval_points"]


def _all_scalar_like(xs: Sequence[Any]) -> bool:
    """Returns True if all entries in xs are scalar-like.

    Scalar-like means:
    - Python scalar, or
    - 0-dim numpy array.
    """
    for x in xs:
        if np.isscalar(x):
            continue
        if isinstance(x, np.ndarray) and x.shape == ():
            continue
        return False
    return True


def eval_points(
    func: Callable[[Any], Any],
    xs: Sequence[Any],
    n_workers: int | None = None,
) -> np.ndarray:
    """Evaluates ``func`` at a sequence of points.

    Args:
        func: Callable taking a single argument (scalar or array-like).
        xs: 1D sequence of points at which to evaluate ``func``.
            Entries may be scalars or array-like objects (e.g. vectors/tensors).
        n_workers: Number of parallel outer workers. If None or <=1, runs serially.
            If greater than the number of points, capped to that number.

    Returns:
        An array of function values at the specified points.
    """
    xs_list = list(xs)
    if not xs_list:
        return np.asarray([], dtype=float)

    scalar_like = _all_scalar_like(xs_list)
    args = _to_eval_args(xs_list, scalar_like)

    outer_workers = _cap_outer_workers(n_workers, len(args))
    inner_workers = resolve_inner_from_outer(outer_workers)

    # parallel_execute handles both outer >1 and outer == 1 paths.
    arg_tuples = [(x,) for x in args]
    vals = parallel_execute(
        worker=func,
        arg_tuples=arg_tuples,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    return np.asarray(vals)


def _to_eval_args(xs: Sequence[Any], scalar_like: bool) -> list[Any]:
    """Prepare arguments for evaluation.

    If all entries are scalar-like, cast to float (legacy behaviour).
    Otherwise, pass through as-is to support tensor/array inputs.
    """
    if scalar_like:
        return [float(x) for x in xs]
    return list(xs)


def _cap_outer_workers(n_workers: int | None, n_tasks: int) -> int:
    """Cap outer workers by number of tasks; ensure at least 1."""
    if n_workers is None or n_workers <= 1:
        return 1
    n = int(n_workers)
    if n_tasks <= 0:
        return 1
    return max(1, min(n, int(n_tasks)))
