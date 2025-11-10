"""Numerical utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "central_difference_error_estimate",
    "richardson_extrapolate",
]


def central_difference_error_estimate(step_size, order: int = 1):
    """Heuristic size of the first omitted term (sub-leading) in central-difference stencils.

    Interpreted as an O(h^2) truncation-error scale for smooth functions on a
    uniform grid. Constants are rule-of-thumb (not strict bounds).

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


def richardson_extrapolate(
        base_values: Sequence[NDArray[np.float64] | float],
        p: int,
        r: float = 2.0,
) -> NDArray[np.float64] | float:
    """Richardson extrapolation on a sequence of approximations.

    Richardson extrapolation improves the accuracy of a sequence of
    numerical approximations that converge with a known leading-order error
    term. Given a sequence of approximations computed with decreasing step sizes,
    this method combines them to eliminate the leading error term, yielding
    a more accurate estimate of the true value.

    Args:
        base_values: Sequence of approximations at different step sizes.
            The step sizes are assumed to decrease by a factor of `r`
            between successive entries.
        p: The order of the leading error term in the approximations.
        r: The step-size reduction factor between successive entries
            (default is 2.0).

    Returns:
        The extrapolated value with improved accuracy.

    Raises:
        ValueError: If `base_values` has fewer than two entries.
    """
    # Work on float arrays for both scalar and vector cases
    n = len(base_values)
    if n < 2:
        raise ValueError("richardson_extrapolate requires at least two base values.")

    vals = [np.asarray(v, dtype=float) for v in base_values]

    for j in range(1, n):
        factor = r ** (p * j)
        for k in range(n - 1, j - 1, -1):
            vals[k] = (factor * vals[k] - vals[k - 1]) / (factor - 1.0)

    result = vals[-1]
    return float(result) if result.ndim == 0 else result
