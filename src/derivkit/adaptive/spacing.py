"""Convert a spacing spec ('auto', '<p>%', or number) into a positive step size."""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_spacing"]


def resolve_spacing(spacing, x0: float, base_abs: float | None) -> float:
    """Return a positive step size h around x0; 'auto' and '<p>%' scale with the absolute value of x0, numeric inputs are absolute, and base_abs sets the floor.

    Converts a user-facing spacing option into a numeric spacing h suitable for
    finite-difference or sampling routines. The 'auto' mode corresponds to 2% of
    the magnitude of x0 (i.e., 0.02 * abs(x0)) but never below the floor.

    If the scaled value is below the floor, the result is the floor (e.g., with the
    default 1e-3 and x0≈1e-6, 'auto' returns 1e-3); pass base_abs to choose a
    smaller floor. Numeric inputs are absolute and do not use the floor;
    the floor applies only to "auto" and "<p>%".

    Args:
        spacing: "auto", a percentage (e.g. a string representing a percentage "2%"), or a positive number.
        x0: Point at which the derivative is evaluated; scale reference for "auto"
            and percentages.
        base_abs: Absolute lower bound for h (defaults to 1e-3 if None).

    Returns:
        float: A positive, finite spacing value.

    Raises:
        ValueError: If spacing is invalid (e.g. non-positive/NaN number, malformed
            percent, or unsupported type).
    """
    floor = 1e-3 if base_abs is None else float(base_abs)

    # numeric absolute spacing (floor does not apply to explicit numbers)
    if isinstance(spacing, (int, float)):
        h = float(spacing)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("numeric spacing must be positive and finite.")
        return h

    # auto: scale with absolute value of x0 but never below floor
    if spacing == "auto":
        h = 0.02 * abs(float(x0))
        return float(max(h, floor))

    # percent like '2%'
    if isinstance(spacing, str) and spacing.strip().endswith("%"):
        s = spacing.strip()
        try:
            frac = float(s[:-1]) / 100.0
        except ValueError:
            raise ValueError(f"invalid percent spacing: {spacing!r}")
        if not np.isfinite(frac) or frac <= 0:
            raise ValueError("percent spacing must be > 0.")
        h = frac * abs(float(x0))
        # If x0 == 0 or too small, fall back to floor
        return float(max(h, floor))

    raise ValueError(
        "spacing must be 'auto', a percent like '2%', or a positive number."
    )
