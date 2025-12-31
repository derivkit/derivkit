"""Linear-algebra helpers for Laplace approximation.

This module is intentionally separate from `derivkit.utils.linalg` to avoid
merge conflicts while LaplaceApproximation is developed in parallel PRs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "symmetrize_matrix",
    "make_spd_by_jitter",
]


def symmetrize_matrix(a: Any) -> NDArray[np.float64]:
    """Symmetrizes a square matrix.

    Args:
        a: Array-like square matrix.

    Returns:
        Symmetric 2D float64 NumPy array.

    Raises:
        ValueError: If input is not a square 2D array.
    """
    m = np.asarray(a, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square 2D; got shape {m.shape}.")
    return (0.5 * (m + m.T)).astype(np.float64, copy=False)


def make_spd_by_jitter(
    matrix: Any,
    *,
    max_tries: int = 12,
    jitter_scale: float = 1e-10,
    jitter_floor: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    """Makes a symmetric matrix SPD by adding diagonal jitter if necessary.

    Args:
        matrix: Array-like square matrix.
        max_tries: Maximum number of jitter attempts (powers of 10).
        jitter_scale: Scale factor for jitter based on mean diagonal.
        jitter_floor: Minimum jitter to add.

    Returns:
        A tuple (spd_matrix, jitter_added), where spd_matrix is the SPD matrix
        and jitter_added is the amount of jitter added to the diagonal.

    Raises:
        np.linalg.LinAlgError: If the matrix cannot be made SPD within max_tries
    """
    h = symmetrize_matrix(matrix)
    n = h.shape[0]

    # attempt without jitter first
    try:
        np.linalg.cholesky(h)
        return h, 0.0
    except np.linalg.LinAlgError:
        pass

    diag_mean = float(np.mean(np.diag(h))) if n else 1.0
    if not np.isfinite(diag_mean) or diag_mean == 0.0:
        diag_mean = 1.0

    base = jitter_scale * abs(diag_mean) + jitter_floor
    eye = np.eye(n, dtype=np.float64)

    jitter = 0.0
    for k in range(max_tries):
        jitter = base * (10.0 ** k)
        h_try = h + jitter * eye
        try:
            np.linalg.cholesky(h_try)
            return h_try, float(jitter)
        except np.linalg.LinAlgError:
            continue

    evals = np.linalg.eigvalsh(h)
    min_eig = float(np.min(evals)) if evals.size else 0.0
    raise np.linalg.LinAlgError(
        "Hessian was not SPD and could not be regularized with diagonal jitter "
        f"(min_eig={min_eig:.2e}, last_jitter={jitter:.2e})."
    )
