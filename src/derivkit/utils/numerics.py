"""Numerical utilities."""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "central_difference_error_estimate",
    "relative_error",
]


def central_difference_error_estimate(step_size: float, order: int = 1) -> float:
    """Computes a general heuristic size of the first omitted term in central-difference stencils.

    Uses the general pattern h^2 / ((order + 1) * (order + 2)) as a
    rule-of-thumb O(h^2) truncation-error scale.

    Args:
        step_size: Grid spacing.
        order: Derivative order (positive integer).

    Returns:
        Estimated truncation error scale.
    """
    if order < 1:
        raise ValueError("order must be a positive integer.")

    # if order higher than 4 we do not support it, but we can still compute the estimate
    if order > 4:
        warnings.warn(
            "central_difference_error_estimate called with order > 4,"
            " which is not supported by finite_difference module.",
            UserWarning,
        )
    return step_size**2 / ((order + 1) * (order + 2))


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative error metric between a and b.

    This metric is defined as the maximum over all components of a and b of
    the absolute difference divided by the maximum of 1.0 and the absolute values of
    a and b.

    Args:
        a: First array-like input.
        b: Second array-like input.

    Returns:
        The relative error metric as a float.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))
