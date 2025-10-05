"""Adaptive positive offset generator around a given expansion point ``x0``."""

from __future__ import annotations

import numpy as np

__all__ = ["get_adaptive_offsets"]


def get_adaptive_offsets(
    x0: float,
    *,
    base_rel: float = 0.01,
    base_abs: float = 1e-6,
    factor: float = 1.5,
    num_offsets: int = 10,
    max_rel: float = 0.05,
    max_abs: float = 1e-2,
    step_mode: str = "auto",  # now supports: "auto", "absolute", "relative"
    x_small_threshold: float = 1e-3,
) -> np.ndarray:
    """Return strictly positive step sizes tailored to the scale of ``x0``.

    This produces a 1D array of monotonically increasing offsets without
    symmetry, zero insertion, or extension. Offsets are grown geometrically
    by ``factor`` starting from either an absolute or relative base, capped
    by ``max_abs`` or ``max_rel`` (times the chosen scale).

    Args:
        x0: Expansion point used to determine relative scaling.
        base_rel: Base relative step (fraction of the scale) for the first offset.
        base_abs: Base absolute step (in data units) for the first offset.
        factor: Geometric growth factor between consecutive offsets.
        num_offsets: Number of candidate offsets to generate before deduplication.
        max_rel: Maximum relative step (fraction of the scale).
        max_abs: Maximum absolute step (in data units).
        step_mode: One of ``"auto"``, ``"absolute"``. In ``"auto"``, absolute
            mode is used when ``|x0| < x_small_threshold``.
        x_small_threshold: Threshold determining when ``x0`` is considered small.

    Returns:
        A strictly positive, increasing ``np.ndarray`` of offsets.

    Raises:
        ValueError: If no valid offsets are generated (e.g., after clipping).
    """
    x0 = float(x0)
    use_abs = (step_mode == "absolute") or (
            step_mode == "auto" and abs(x0) < x_small_threshold
    )
    if use_abs:
        bases = [min(base_abs * (factor ** i), max_abs) for i in range(num_offsets)]
    else:
        # treat both "auto" (large |x0|) and explicit "relative" the same here
        scale = max(abs(x0), x_small_threshold)
        bases = [
            min(base_rel * (factor ** i), max_rel) * scale for i in range(num_offsets)
        ]
    offs = np.unique([b for b in bases if b > 0.0])
    if offs.size == 0:
        raise ValueError("No valid offsets generated.")
    return offs
