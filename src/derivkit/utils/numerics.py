"""Numerical utilities."""

from __future__ import annotations

__all__ = [
    "central_difference_error_estimate",
]


def central_difference_error_estimate(step_size, order: int = 1):
    """Rule-of-thumb truncation error for central differences.

    This estimate comes from the leading term in the Taylor expansion of
    central-difference formulas. It gives the expected order of magnitude of
    the truncation error but is not an exact bound—hence “heuristic.”

    Args:
      step_size: Grid spacing.
      order: Derivative order (1–4 supported).

    Returns:
      Estimated truncation error scale.

    Raises:
      ValueError: If ``order`` is not in {1, 2, 3, 4}.
    """
    if order == 1:
        return step_size**2 / 6
    if order == 2:
        return step_size**2 / 12
    if order == 3:
        return step_size**2 / 20
    if order == 4:
        return step_size**2 / 30
    raise ValueError("Only derivative orders 1–4 are supported.")
