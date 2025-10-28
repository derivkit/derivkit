"""Utility functions for DerivKit package."""

from .linalg import (
    invert_covariance,
    normalize_covariance,
    solve_or_pinv,
)

__all__ = [
    "solve_or_pinv",
    "invert_covariance",
    "normalize_covariance",
]
