"""Linear algebra helper functions with diagnostics and canonicalization."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def invert_covariance(
    cov: np.ndarray,
    *,
    rcond: float = 1e-12,
    warn_prefix: str = "",
) -> np.ndarray:
    """Return the inverse covariance with diagnostics; fall back to pseudoinverse when needed.

    This helper accepts a scalar (0D), a diagonal variance vector (1D), or a full
    covariance matrix (2D). Inputs are canonicalized to a 2D array before inversion.
    The function warns (but does not modify data) if the matrix is non-symmetric,
    warns on ill-conditioning, and uses a pseudoinverse when inversion is not viable.

    Args:
        cov: Covariance (scalar, diagonal vector, or full 2D matrix).
        rcond: Cutoff for small singular values used by ``np.linalg.pinv``.
        warn_prefix: Optional prefix included in warnings (e.g., a class or function name).

    Returns:
        A 2D NumPy array containing the inverse covariance.

    Raises:
        ValueError: If ``cov`` has more than 2 dimensions.
    """
    cov = np.asarray(cov, dtype=float)

    # Canonicalize to 2D
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    elif cov.ndim != 2:
        raise ValueError(f"`cov` must be 0D, 1D, or 2D; got ndim={cov.ndim}.")

    prefix = f"[{warn_prefix}] " if warn_prefix else ""

    # Symmetry check (warn only; do not symmetrize)
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
        warnings.warn(
            f"{prefix}`cov` is not symmetric; proceeding as-is (no symmetrization).",
            RuntimeWarning,
        )

    n = cov.shape[0]

    # Symmetry check (warn only; do not symmetrize)
    symmetric = np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
    if not symmetric:
        warnings.warn(
            f"{prefix}`cov` is not symmetric; proceeding as-is",
            RuntimeWarning,
        )

    # Ill-conditioning warning
    try:
        cond_val = np.linalg.cond(cov)
        if (not np.isfinite(cond_val)) or (cond_val > 1.0 / rcond):
            warnings.warn(
                f"{prefix}`cov` is ill-conditioned (cond≈{cond_val:.2e}); results may be unstable.",
                RuntimeWarning,
            )
    except np.linalg.LinAlgError:
        pass

    # Rank check
    try:
        rank = np.linalg.matrix_rank(cov)
    except np.linalg.LinAlgError:
        rank = n

    # Try exact inverse when full rank; otherwise pseudoinverse
    if rank == n:
        try:
            inv = np.linalg.inv(cov)
            return np.asarray(inv, dtype=float)
        except np.linalg.LinAlgError:
            # fall through to pinv with warning
            warnings.warn(
                f"{prefix}`cov` inversion failed; using pseudoinverse.",
                RuntimeWarning,
            )

    # Pseudoinverse path — IMPORTANT: hermitian = symmetric flag
    warnings.warn(
        f"{prefix}`cov` inversion failed; using pseudoinverse.",
        RuntimeWarning,
    )
    inv_cov = np.linalg.pinv(cov, rcond=rcond, hermitian=symmetric).astype(float, copy=False)
    return inv_cov


def normalize_covariance(
    cov: ArrayLike,
    n_parameters: int,
    *,
    asym_atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Return a canonicalized covariance matrix.

    Accepts a scalar (0D), a diagonal variance vector (1D), or a full covariance
    matrix (2D). Validates shapes and finiteness, symmetrizes full matrices,
    and returns a 2D array of shape (k, k).

    Args:
        cov: Covariance (scalar, diagonal vector, or full 2D matrix).
        n_parameters: Expected size of the covariance (number of parameters).
        asym_atol: Absolute tolerance for symmetry check of full matrices.

    Returns:
        A 2D NumPy array containing the canonicalized covariance matrix.

    Raises:
        ValueError: If ``cov`` has invalid shape, contains non-finite values,
            or is too asymmetric.
    """
    arr = np.asarray(cov, dtype=float)

    # scalar
    if arr.ndim == 0:
        if not np.isfinite(arr):
            raise ValueError("cov scalar must be finite.")
        return np.eye(n_parameters, dtype=float) * float(arr)

    # 1D diag
    if arr.ndim == 1:
        if arr.shape[0] != n_parameters:
            raise ValueError(f"cov vector length {arr.shape[0]} != k={n_parameters}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("cov diagonal contains non-finite values.")
        return np.diag(arr)

    # 2D full
    if arr.ndim == 2:
        if arr.shape != (n_parameters, n_parameters):
            raise ValueError(f"cov shape {arr.shape} != ({n_parameters},{n_parameters}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("cov matrix contains non-finite values.")
        a = arr.astype(float, copy=False)
        skew = a - a.T
        fro = np.linalg.norm(a)
        skew_fro = np.linalg.norm(skew)
        thresh = asym_atol * (fro if fro > 0.0 else 1.0)
        if skew_fro > thresh:
            raise ValueError(
                f"cov matrix too asymmetric (‖A-A^T‖_F={skew_fro:.2e} > {thresh:.2e})."
            )
        return 0.5 * (a + a.T)

    raise ValueError("cov must be scalar, 1D diag vector, or 2D (k,k) matrix.")


def solve_or_pinv(matrix: np.ndarray, vector: np.ndarray, *, rcond: float = 1e-12,
                  assume_symmetric: bool = True, warn_context: str = "linear solve") -> np.ndarray:
    """Solve ``matrix @ x = vector`` with pseudoinverse fallback.

    If ``assume_symmetric`` is True (e.g., Fisher matrices), attempt a
    Cholesky-based solve. If the matrix is not symmetric positive definite
    or is singular, emit a warning and fall back to
    ``np.linalg.pinv(matrix, rcond) @ vector``.

    Args:
      matrix: Coefficient matrix of shape ``(n, n)``.
      vector: Right-hand side vector or matrix of shape ``(n,)`` or ``(n, k)``.
      rcond: Cutoff for small singular values used by ``np.linalg.pinv``.
      assume_symmetric: If True, prefer a Cholesky solve
          (fast path for symmetric positive definite (SPD)/Hermitian).
      warn_context: Short label included in the warning message.

    Returns:
      Solution array ``x`` with shape matching ``vector`` (``(n,)`` or ``(n, k)``).

    Raises:
      ValueError: If shapes of ``matrix`` and ``vector`` are incompatible.
    """
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)

    # Shape checks
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square 2D; got shape {matrix.shape}.")
    n = matrix.shape[0]
    if vector.ndim not in (1, 2) or vector.shape[0] != n:
        raise ValueError(f"vector must have shape (n,) or (n,k) with n={n}; got {vector.shape}.")

    # Rank-deficient shortcut (ensures test captures a warning)
    try:
        rank = np.linalg.matrix_rank(matrix)
    except np.linalg.LinAlgError:
        rank = n
    if rank < n:
        warnings.warn(
            f"In {warn_context}, matrix is rank-deficient (rank={rank} < {n}); "
            f"falling back to pseudoinverse with rcond={rcond}.",
            RuntimeWarning,
        )
        hermitian = np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)
        return (np.linalg.pinv(matrix, rcond=rcond, hermitian=hermitian) @ vector).astype(float, copy=False)

    # Fast path: symmetric/Hermitian or general solve
    try:
        if assume_symmetric:
            l_factor = np.linalg.cholesky(matrix)
            y = np.linalg.solve(l_factor, vector)
            return np.linalg.solve(l_factor.T, y)
        else:
            return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        cond_msg = ""
        try:
            cond_val = np.linalg.cond(matrix)
            if np.isfinite(cond_val):
                cond_msg = f" (cond≈{cond_val:.2e})"
        except np.linalg.LinAlgError:
            pass

        warnings.warn(
            f"In {warn_context}, the matrix was not SPD or was singular; "
            f"falling back to pseudoinverse with rcond={rcond}{cond_msg}.",
            RuntimeWarning,
        )
        hermitian = np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)
        return (np.linalg.pinv(matrix, rcond=rcond, hermitian=hermitian) @ vector).astype(float, copy=False)
