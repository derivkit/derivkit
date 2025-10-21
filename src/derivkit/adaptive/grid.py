"""Utility functions for building grids of points."""

from __future__ import annotations

import numpy as np

from .spacing import resolve_spacing

__all__ = ["make_offsets", "make_grid", "chebyshev_offsets"]


def make_offsets(n_points: int, base: float, direction: str) -> np.ndarray:
    """Construct a grid of offsets around zero, never including 0.

    Args:
        n_points: number of points to generate (>=1)
        base: spacing between points (>0)
        direction: 'both', 'pos', or 'neg'. 'both' gives a symmetric grid around 0,
                   'pos' gives points > 0, 'neg' gives points < 0.

    Returns:
        Array of offsets (length n_points), never including 0.
    """
    if not np.isfinite(base) or base <= 0:
        raise ValueError("Resolved spacing is not a positive finite number.")
    if n_points < 1:
        raise ValueError("n_points must be >= 1.")
    if direction not in {"both", "pos", "neg"}:
        raise ValueError("direction must be 'both', 'pos', or 'neg'.")

    h = float(base)

    if direction == "both":
        left = n_points // 2
        right = n_points - left
        k = np.concatenate(
            (
                -np.arange(left, 0, -1, dtype=float),
                np.arange(1, right + 1, dtype=float),
            )
        )
        return h * k

    if direction == "pos":
        return h * np.arange(1, n_points + 1, dtype=float)

    # direction == "neg"
    return -h * np.arange(1, n_points + 1, dtype=float)


def make_grid(
    x0: float,
    *,
    n_points: int,
    spacing: str | float | np.ndarray,
    base_abs: float | None,
    need_min: int,
    use_physical_grid: bool,
) -> tuple[np.ndarray, np.ndarray, int, float, str]:
    """Unified grid builder.

    Args:
        x0: expansion point
        n_points: number of points to generate (if not use_physical_grid)
        spacing: 'auto', '<pct>%', numeric > 0, or array of physical
                    sample points (if use_physical_grid)
        base_abs: absolute fallback (also used by 'auto'); if None, uses 1
        need_min: minimum number of points required (for validation)
        use_physical_grid: if True, spacing is an array of physical sample points

    Returns:
      x: array of physical sample points
      t: offsets (x - x0)
      n_pts: number of samples
      spacing_resolved: numeric spacing used (np.nan if physical grid given)
      direction_used: 'custom' if physical grid, else the input direction
    """
    x0 = float(x0)

    if use_physical_grid:
        x = np.asarray(spacing, dtype=float)
        if x.ndim != 1:
            raise ValueError("When use_physical_grid=True, spacing must be a 1D array of x-samples.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Physical grid contains non-finite values.")
        if x.size < need_min:
            raise ValueError(f"Physical grid must have at least {need_min} points for requested order.")
        t = x - x0
        return x, t, x.size, float("nan"), "physical"

    # Chebyshev mode
    if n_points < need_min:
        raise ValueError(f"n_points must be >= {need_min} for requested order.")

    halfwidth = resolve_spacing(spacing, x0, base_abs)
    t = chebyshev_offsets(halfwidth, n_points, include_center=True)
    x = x0 + t
    return x, t, x.size, float(halfwidth), "chebyshev"


def chebyshev_offsets(halfwidth: float, n_points: int, include_center: bool = True) -> np.ndarray:
    """Generate Chebyshev-distributed offsets within [-halfwidth, halfwidth].

    This function generates `n_points` offsets based on the Chebyshev nodes,
    which are distributed to minimize interpolation error. The offsets lie
    within the interval `[-halfwidth, halfwidth]`. Optionally, the center point
    `0.0` can be included in the offsets if it is not already present.

    Args:
        halfwidth: Half the width of the interval (>0).
        n_points: Number of points to generate (>=1).
        include_center: If True, include 0.0 in the offsets if not already present.
                        Default is True.

    Returns:
        Array of offsets sorted in ascending order.
    """
    k = np.arange(1, n_points + 1)
    u = np.cos((2*k - 1) * np.pi / (2 * n_points))  # (-1,1]
    t = halfwidth * u
    if include_center and 0.0 not in t:
        t = np.sort(np.append(t, 0.0))
    else:
        t = np.sort(t)
    return t

