"""Utility functions for building grids of points."""

from __future__ import annotations

import numpy as np

from .spacing import resolve_spacing
from .transforms import (
    signed_log_forward,
    signed_log_to_physical,
    sqrt_domain_forward,
    sqrt_to_physical,
)

__all__ = ["make_offsets",
           "make_grid",
           "chebyshev_offsets",
           "make_domain_aware_chebyshev_grid",
           "ensure_min_samples_and_maybe_rebuild"
]


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

    This function generates ``n_points`` offsets based on the Chebyshev nodes,
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
        t = np.append(t, 0.0)
    t = np.sort(t)
    return t


def make_domain_aware_chebyshev_grid(
    x0: float,
    *,
    n_points: int,
    spacing: str | float,
    base_abs: float | None,
    domain: tuple[float | None, float | None] | None,
    max_cheby_points: int = 30,
) -> tuple[str, np.ndarray, np.ndarray, float, float | None]:
    """Build a Chebyshev grid around ``x0`` with optional domain-aware transforms.

    This constructs Chebyshev-distributed sample points around ``x0``. If a
    one-sided domain is supplied, it switches to a transform that respects the
    domain. When the domain is strictly non-negative or non-positive and
    ``x0 == 0``, a square-root coordinate centered at the boundary is used.
    When the domain is single-signed and a symmetric grid around ``x0`` would
    violate it, a signed-log coordinate is used.

    Args:
      x0: Expansion point about which to place the grid.
      n_points: Number of Chebyshev nodes to generate (center included).
      spacing: Sampling half-width control. Accepts ``"auto"``, a percentage
        string such as ``"2%"``, or a positive float half-width.
      base_abs: Absolute fallback scale used by ``"auto"`` (and transforms)
        near zero.
      domain: Optional bounds ``(lo, hi)``. Use ``None`` for an open end. If the
        domain is strictly non-negative or non-positive, a domain-aware transform
        may be applied as described above.
      max_cheby_points: Safety cap for the default Chebyshev node count.

    Returns:
      Tuple[str, np.ndarray, np.ndarray, float, float | None]: A 5-tuple
        ``(mode, x, t, spacing_resolved, sign_used)``. ``mode`` is one of
        ``"x"``, ``"signed_log"``, or ``"sqrt"``. ``x`` are physical samples.
        ``t`` are internal offsets in the fit coordinate. ``spacing_resolved``
        is the numeric half-width actually used. ``sign_used`` is ``+1`` or
        ``-1`` for ``"sqrt"`` mode and ``None`` otherwise.

    Raises:
      ValueError: If ``n_points`` exceeds ``max_cheby_points`` or if ``spacing``
        cannot be resolved to a positive finite half-width.

    Notes:
      The returned ``x`` always respects a one-sided domain. Chebyshev nodes
      include the center if not already present.
    """
    mode = "x"
    sign_used = None

    if n_points > max_cheby_points:
        raise ValueError(
            f"Too many points for default Chebyshev grid (n_points={n_points}, max={max_cheby_points}). "
            "Increase `spacing` (wider half-width) or pass an explicit grid."
        )

    halfwidth = resolve_spacing(spacing, float(x0), base_abs)
    t = chebyshev_offsets(halfwidth, n_points, include_center=True)
    x = x0 + t
    spacing_resolved = float(halfwidth)

    if domain is None:
        return "x", x, t, spacing_resolved, None

    lo, hi = domain
    pos_only = (lo is not None and lo >= 0.0) and (hi is None or hi >= 0.0)
    neg_only = (hi is not None and hi <= 0.0) and (lo is None or lo <= 0.0)
    single_sign = pos_only or neg_only

    if single_sign and x0 == 0.0:
        # sqrt-domain centered at 0 boundary, symmetric in u
        _u0, sgn = sqrt_domain_forward(x0)
        half_u = resolve_spacing(spacing, 0.0, base_abs=1e-3)
        if n_points > max_cheby_points:
            raise ValueError(
                f"Too many points for sqrt-mode Chebyshev (n_points={n_points}, max={max_cheby_points})."
            )
        tu = chebyshev_offsets(half_u, n_points, include_center=True)
        t = tu
        x = sqrt_to_physical(t, sgn)
        spacing_resolved = float(half_u)
        mode = "sqrt"
        sign_used = sgn

    elif single_sign:
        violates = ((lo is not None) and np.any(x < lo)) or ((hi is not None) and np.any(x > hi))
        if violates and x0 != 0.0:
            q0, sgn = signed_log_forward(x0)
            half_q = resolve_spacing(spacing, q0, base_abs=1e-3)
            if n_points > max_cheby_points:
                raise ValueError(
                    f"Too many points for signed-log Chebyshev (n_points={n_points}, max={max_cheby_points})."
                )
            tq = chebyshev_offsets(half_q, n_points, include_center=True)
            t = tq
            x = signed_log_to_physical(q0 + tq, sgn)
            spacing_resolved = float(half_q)
            mode = "signed_log"

    return mode, x, t, spacing_resolved, sign_used


def ensure_min_samples_and_maybe_rebuild(
    *,
    mode: str,
    x: np.ndarray,
    t: np.ndarray,
    spacing_resolved: float,
    sign_used: float | None,
    x0: float,
    order: int,
    n_points: int,
    spacing: str | float,
    base_abs: float = 1e-3,
    max_cheby_points: int = 30,
) -> tuple[str, np.ndarray, np.ndarray, float, float | None]:
    """Guarantee sufficient samples for the requested derivative; rebuild if needed.

    Computes the minimum number of samples required for a stable polynomial-fit
    derivative estimate and, when the current grid is a default Chebyshev grid
    (indicated by a finite ``spacing_resolved``), rebuilds it with more nodes
    if necessary. For explicit/physical grids (``spacing_resolved`` is not finite),
    a deficiency is treated as an error.

    The required sample count is:
      - ``min_pts = 2 * deg_req + 1``, where
      - ``deg_req = 2 * order`` for ``mode == "sqrt"`` (due to pullback), else ``deg_req = order``.

    Args:
      mode: Sampling mode, one of ``"x"``, ``"signed_log"``, or ``"sqrt"``.
      x: Physical sample locations, shape ``(n,)``.
      t: Internal offsets used for fitting, shape ``(n,)``.
      spacing_resolved: Numeric half-width used to generate the Chebyshev grid.
        Finite implies a default/rebuildable grid; non-finite implies an explicit grid.
      sign_used: For ``"sqrt"`` mode, the branch sign ``(+1 or -1)``; otherwise ``None``.
      x0: Expansion point about which derivatives are computed.
      order: Derivative order (``>= 1``).
      n_points: Target number of Chebyshev nodes if a rebuild is performed.
      spacing: Half-width control used when rebuilding (``"auto"``, percentage string, or positive float).
      base_abs: Absolute fallback scale for resolving ``spacing`` near zero. Defaults to ``1e-3``.
      max_cheby_points: Safety cap on Chebyshev node count when rebuilding.

    Returns:
      Tuple[str, np.ndarray, np.ndarray, float, float | None]:
        A 5-tuple ``(mode, x_out, t_out, spacing_out, sign_out)`` with the (possibly)
        rebuilt grid and associated metadata. The return types mirror the inputs.

    Raises:
      ValueError: If the grid is explicit (non-finite ``spacing_resolved``) and
        has fewer than the required samples.
      ValueError: If a rebuild would exceed ``max_cheby_points``.
      ValueError: If ``order < 1``.

    Notes:
      - When rebuilding in ``"signed_log"`` or ``"sqrt"`` mode, ``spacing`` is
        resolved in the transform coordinate (log/sqrt) space and mapped back
        to physical ``x``.
      - The center point is included for Chebyshev grids if not already present.
    """
    n_eff = len(t)
    deg_req = (2 * order) if (mode == "sqrt") else order
    min_pts = 2 * deg_req + 1
    if n_eff >= min_pts:
        return mode, x, t, spacing_resolved, sign_used

    # explicit grid → spacing_resolved will be NaN (we don't rebuild those)
    if not np.isfinite(spacing_resolved):
        raise ValueError(
            f"Not enough samples for order={order} (mode={mode}). Need ≥{min_pts}, got {n_eff}."
        )

    target = max(min_pts, n_points)
    if target > max_cheby_points:
        raise ValueError(
            f"Order={order} needs ≥{min_pts} points but cap is {max_cheby_points}. "
            "Increase `spacing`, reduce `order`, or provide an explicit grid."
        )

    if mode == "x":
        t = chebyshev_offsets(spacing_resolved, target, include_center=True)
        x = x0 + t
        return mode, x, t, spacing_resolved, sign_used

    if mode == "signed_log":
        q0, sgn = signed_log_forward(x0)
        half_q = resolve_spacing(spacing, q0, base_abs=base_abs)
        tq = chebyshev_offsets(half_q, target, include_center=True)
        x = signed_log_to_physical(q0 + tq, sgn)
        return mode, x, tq, float(half_q), sgn

    if mode == "sqrt":
        half_u = resolve_spacing(spacing, 0.0, base_abs=base_abs)
        tu = chebyshev_offsets(half_u, target, include_center=True)
        x = sqrt_to_physical(tu, sign_used)
        return mode, x, tu, float(half_u), sign_used

    return mode, x, t, spacing_resolved, sign_used
