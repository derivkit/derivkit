"""Offset grid builders for adaptive sampling around ``x0``.

This module works with *relative* offsets (around 0). You typically:

1) Start from a small set of strictly positive seed offsets near zero;
2) Mirror them to obtain a symmetric grid about 0 (optionally including 0);
3) If that symmetric grid is too small for a stable fit, extend the positive
   side geometrically until the mirrored grid meets a size target
   (the "required point count").
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from derivkit.adaptive.offsets import (
    get_adaptive_offsets as _default_get_adaptive_offsets,
)

__all__ = [
    "build_x_offsets",
    "symmetric_offsets",
    "extend_offsets_to_required",
]


def symmetric_offsets(offsets: np.ndarray, include_zero: bool) -> np.ndarray:
    """Make a symmetric set around 0 from strictly positive seeds.

    You pass *positive* offsets (e.g., [a, b, c]) and choose whether to include
    0 itself. The function returns the sorted union of {±a, ±b, ±c} and
    optionally {0}. It does *not* invent new step sizes; it only mirrors.

    Args:
      offsets: Strictly positive offsets (1D).
      include_zero: If True, also include 0 in the final grid.

    Returns:
      np.ndarray: A sorted 1D array symmetric about 0.

    Raises:
      ValueError: If any input offset is non-positive.
    """
    pos = np.asarray(offsets, float)
    if np.any(pos <= 0):
        raise ValueError("symmetric_offsets expects strictly positive offsets.")
    steps = np.insert(pos, 0, 0.0) if include_zero else pos
    out = np.unique(np.concatenate([steps, -steps]))
    return np.sort(out)


def extend_offsets_to_required(
    offsets: np.ndarray,
    include_zero: bool,
    factor: float,
    growth_limit: float,
    required_points: int,
) -> np.ndarray:
    """Extend seeds until the *symmetric* grid reaches a target size.

    You provide initial strictly positive seeds. If, after mirroring (and
    optionally adding 0), the symmetric grid has fewer than ``required_points``
    samples, we *extend the positive seeds* by multiplying the last seed by
    ``factor`` (geometric growth), then mirror again. We stop when the size
    target is met or when the next candidate would exceed ``growth_limit``.

    Args:
      offsets: Initial strictly positive offsets (1D).
      include_zero: Whether to include 0 when forming the symmetric grid.
      factor: Geometric growth factor applied to the last positive seed.
      growth_limit: Maximum allowed positive offset value.
      required_points: Target number of samples in the final symmetric grid.
        This is the “required point count” used by the fit loop (see
        ``build_x_offsets``).

    Returns:
      np.ndarray: The final symmetric 1D grid meeting size/limit criteria.
    """
    cur = np.array(offsets, float)
    while True:
        grid = symmetric_offsets(cur, include_zero)
        if grid.size >= int(required_points):
            return grid
        nxt = cur[-1] * float(factor)
        if nxt > float(growth_limit):
            return grid
        cur = np.append(cur, nxt)


def build_x_offsets(
    *,
    x0: float,
    order: int,
    is_relative: bool = False,
    delta: float,
    include_zero: bool,
    min_samples: int,
    min_used_points: int,
    get_adaptive_offsets: Callable[..., np.ndarray] = _default_get_adaptive_offsets,
) -> tuple[np.ndarray, int]:
    """Build the symmetric relative grid and compute ``required_points``.

    ``required_points`` is the minimum total samples the fit loop may use:
    ``max(min_samples, min_used_points, order + 2)``. Offsets are relative
    to 0; callers typically evaluate at ``x0 + offsets``.

    Args:
      x0: Expansion point (forwarded to ``get_adaptive_offsets``).
      order: Derivative order (sets a stability floor).
      is_relative: When set to try, take ``delta`` to be relative to ``x0``.
      delta: The change in ``x0`` used to evaluate the derivative.
      include_zero: Include 0 in the symmetric grid.
      min_samples: Requested minimum total samples.
      min_used_points: Hard floor for usable samples.
      get_adaptive_offsets: Factory for strictly positive seed offsets.

    Returns:
      (x_offsets, required_points): symmetric 1D offsets and the effective
      minimum sample count.
    """
    order_floor = order + 2
    required = max(min_samples, max(min_used_points, order_floor))
    pos = get_adaptive_offsets(x0=x0, base=delta, is_relative=is_relative)
    growth_limit = pos[-1] * (1.5**3)
    x_offsets = extend_offsets_to_required(
        offsets=pos,
        include_zero=include_zero,
        factor=1.5,
        growth_limit=growth_limit,
        required_points=required,
    )
    return x_offsets, required
