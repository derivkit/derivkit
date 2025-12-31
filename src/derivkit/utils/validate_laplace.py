"""Validation utilities for Laplace approximation code.

This module is intentionally standalone to avoid merge conflicts with the main
`derivkit.utils.validate` module while LaplaceApproximation is being developed.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray
import numpy as np

__all__ = [
    "validate_theta_1d_finite",
    "validate_square_matrix_finite",
]


def validate_theta_1d_finite(theta: ArrayLike, *, name: str = "theta") -> NDArray[np.float64]:
    """Validate a 1D finite parameter vector and return as float64.

    Args:
        theta: Array-like parameter vector.
        name: Name used in error messages.

    Returns:
        1D float64 NumPy array.

    Raises:
        ValueError: If not 1D, empty, or contains non-finite values.
    """
    t = np.asarray(theta, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {t.shape}.")
    if t.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{name} contains non-finite values.")
    return t.astype(np.float64, copy=False)


def validate_square_matrix_finite(
    a: ArrayLike, *, name: str = "matrix"
) -> NDArray[np.float64]:
    """Validate a finite 2D square matrix and return as float64.

    Args:
        a: Array-like matrix.
        name: Name used in error messages.

    Returns:
        2D float64 NumPy array.

    Raises:
        ValueError: If not 2D square or contains non-finite values.
    """
    m = np.asarray(a, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be 2D; got ndim={m.ndim}.")
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be square; got shape {m.shape}.")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} contains non-finite values.")
    return m.astype(np.float64, copy=False)
