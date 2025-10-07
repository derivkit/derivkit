"""Adaptive positive offset generator around a given expansion point ``x0``."""

from __future__ import annotations

import numpy as np

__all__ = ["get_adaptive_offsets"]


def get_adaptive_offsets(
    x0: float,
    *,
    base: float = 1e-6,
    factor: float = 1.5,
    num_offsets: int = 10,
    is_relative: bool = False,
) -> np.ndarray:
    """Return strictly positive step sizes tailored to the scale of ``x0``.

    This produces a 1D array of monotonically increasing offsets without
    symmetry, zero insertion, or extension. Offsets are grown geometrically
    by ``factor`` starting from either an absolute or relative base, capped
    by ``max_abs`` or ``max_rel`` (times the chosen scale).

    Args:
        x0: Expansion point used to determine relative scaling.
        base: Base step for the first offset.
        factor: Geometric growth factor between consecutive offsets.
        num_offsets: Number of candidate offsets to generate before deduplication.
        is_relative: When set to true, base is taken relative to x0.

    Returns:
        A strictly positive, increasing ``np.ndarray`` of offsets.

    Raises:
        ValueError: If no valid offsets are generated (e.g., after clipping).
    """
    x0 = float(x0)
    if not is_relative:
        bases = [base * (factor**i) for i in range(num_offsets)]
    else:
        bases = [
            base * (factor**i) * abs(x0) for i in range(num_offsets)
        ]
    offs = np.unique([b for b in bases if b > 0.0])
    if offs.size == 0:
        raise ValueError("No valid offsets generated.")
    return offs
