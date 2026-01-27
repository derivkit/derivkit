"""This module provides small linear-algebra helpers.

The main features are:

1) Diagnostics: warn about non-symmetric inputs, ill-conditioning, and rank issues,
   and choose a safe fallback when a fast path fails.

2) Canonicalization: accept covariance inputs in multiple forms (scalar, 1D diagonal
   vector, or 2D matrix) and convert them into a consistent 2D array with validated
   shape and finite values. In other words, we normalize the input representation
   so downstream code always receives a well-formed (k x k) array.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.logger import derivkit_logger
from derivkit.utils.validate import validate_covariance_matrix_shape

CovSpec = NDArray[np.float64] | Mapping[str, Any]

__all__ = [
    "invert_covariance",
    "normalize_covariance",
    "solve_or_pinv",
    "symmetrize_matrix",
    "make_spd_by_jitter",
    "split_xy_covariance",
    "as_1d_data_vector",
]


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
        ValueError: If ``cov`` has more than 2 dimensions or is not square when 2D.
    """
    cov = np.asarray(cov, dtype=float)

    # Canonicalize to 2D
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    elif cov.ndim != 2:
        raise ValueError(f"`cov` must be 0D, 1D, or 2D; got ndim={cov.ndim}.")

    if cov.ndim == 2 and cov.shape[0] != cov.shape[1]:
        raise ValueError(f"`cov` must be square; got shape={cov.shape}.")

    prefix = f"[{warn_prefix}] " if warn_prefix else ""

    # Symmetry check (warn only; do not symmetrize)
    symmetric = np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
    if not symmetric:
        derivkit_logger.warning(
            f"{prefix}`cov` is not symmetric; proceeding as-is"
        )

    n = cov.shape[0]

    # Ill-conditioning warning
    try:
        cond_val = np.linalg.cond(cov)
        if (not np.isfinite(cond_val)) or (cond_val > 1.0 / rcond):
            derivkit_logger.warning(
                f"{prefix}`cov` is ill-conditioned (cond≈{cond_val:.2e});"
                "results may be unstable."
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
            pass  # fall through to pinv

    # Pseudoinverse path — IMPORTANT: hermitian = symmetric flag
    derivkit_logger.warning(
        f"{prefix}`cov` inversion failed; "
        "falling back to pseudoinverse "
        "(rcond={rcond})."
    )
    inv_cov = np.linalg.pinv(cov, rcond=rcond, hermitian=symmetric).astype(float, copy=False)
    return inv_cov


def normalize_covariance(
    cov: Any,
    n_parameters: int,
    *,
    asym_atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Return a canonicalized covariance matrix.

    Accepts a scalar (0D), a diagonal variance vector (1D), or a full covariance
    matrix (2D). Validates shapes and finiteness, symmetrizes full matrices,
    and returns a 2D array of shape ``(k, k)``.

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
      assume_symmetric: If ``True``, prefer a Cholesky solve
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
        derivkit_logger.warning(
            f"In {warn_context}, matrix is rank-deficient (rank={rank} < {n}); "
            f"falling back to pseudoinverse with rcond={rcond}."
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

        derivkit_logger.warning(
            f"In {warn_context}, the matrix was not SPD or was singular; "
            f"falling back to pseudoinverse with rcond={rcond}{cond_msg}."
        )
        hermitian = np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)
        return (np.linalg.pinv(matrix, rcond=rcond, hermitian=hermitian) @ vector).astype(float, copy=False)


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


def split_xy_covariance(
    cov: CovSpec,
    *,
    nx: int,
    atol_sym: float = 1e-12,
    rtol_sym: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validates and splits a stacked covariance for the concatenated vector ``[x, y]``.

    This function enforces the convention that the full covariance corresponds to a
    stacked data vector ordered as ``[x, y]``, where ``x`` has length ``nx`` and
    ``y`` has length ``n - nx``. It returns the covariance blocks ``(Cxx, Cxy, Cyy)``
    and raises informative errors if the input is not consistent with this convention.

    The input may be provided directly as a 2D covariance matrix, or as a dict-like
    specification that includes the covariance and optional metadata for enforcing or
    reordering the ``[x, y]`` convention.

    Args:
        cov: Full covariance for the stacked vector ``[x, y]``.

            Supported forms are:

            * A 2D array interpreted as already ordered ``[x, y]``.
            * A dict-like object with key ``"cov"`` containing the 2D array.

              The dict may include:

              * ``"order"``: Must be ``"xy"`` (clarifies the intended convention).
              * ``"x_idx"`` and ``"y_idx"``: Integer index arrays used to reorder an
                arbitrary covariance into ``[x, y]`` order before splitting.

        nx: Number of input components in ``x`` (length of ``x`` in the stacked vector).

        atol_sym: Absolute tolerance used for symmetry and cross-block consistency
            checks.

        rtol_sym: Relative tolerance used for symmetry and cross-block consistency
            checks.

    Returns:
        A tuple ``(cxx, cxy, cyy)`` where:

        * ``cxx`` has shape ``(nx, nx)``.
        * ``cxy`` has shape ``(nx, ny)``.
        * ``cyy`` has shape ``(ny, ny)``.

        Here ``ny = n - nx``.

    Raises:
        ValueError: If ``cov`` is not a valid square covariance matrix, contains
            non-finite values, is not symmetric within tolerance, cannot be split
            using ``nx``, or if the cross-blocks are inconsistent with the ``[x, y]``
            stacking convention. Also raised if a dict-like specification is missing
            required keys or uses an unsupported order value.
    """
    if isinstance(cov, Mapping):
        spec = cov
        cov_arr = np.asarray(spec["cov"], dtype=np.float64)

        # Optional explicit reordering
        if ("x_idx" in spec) or ("y_idx" in spec):
            if ("x_idx" not in spec) or ("y_idx" not in spec):
                raise ValueError("If using indices,"
                                 " you must provide both 'x_idx' and 'y_idx'.")
            x_idx = np.asarray(spec["x_idx"], dtype=np.int64)
            y_idx = np.asarray(spec["y_idx"], dtype=np.int64)
            cov_arr = _reorder_cov_to_xy(cov_arr, x_idx=x_idx, y_idx=y_idx)

        order = spec.get("order", "xy")
        if order != "xy":
            raise ValueError("Only order='xy' is supported."
                             " Use x_idx/y_idx to reorder explicitly.")
    else:
        cov_arr = np.asarray(cov, dtype=np.float64)

    validate_covariance_matrix_shape(cov_arr)

    if not np.all(np.isfinite(cov_arr)):
        raise ValueError("cov must contain only finite values.")

    if not np.allclose(cov_arr, cov_arr.T, atol=atol_sym, rtol=rtol_sym):
        max_asym = float(np.max(np.abs(cov_arr - cov_arr.T)))
        raise ValueError(
            "cov must be symmetric within tolerance. "
            f"max|cov-cov.T|={max_asym:g} (atol={atol_sym:g}, rtol={rtol_sym:g})."
        )

    n = int(cov_arr.shape[0])
    if not (0 < nx < n):
        raise ValueError(f"nx must satisfy 0 < nx < cov.shape[0];"
                         f" got nx={nx}, n={n}.")

    ny = n - nx

    cxx = cov_arr[:nx, :nx]
    cxy = cov_arr[:nx, nx:]
    cyy = cov_arr[nx:, nx:]

    # Block shape checks
    if cxx.shape != (nx, nx):
        raise ValueError(f"cxx must have shape ({nx},{nx}); got {cxx.shape}.")
    if cxy.shape != (nx, ny):
        raise ValueError(f"cxy must have shape ({nx},{ny}); got {cxy.shape}.")
    if cyy.shape != (ny, ny):
        raise ValueError(f"cyy must have shape ({ny},{ny}); got {cyy.shape}.")

    # Cross-block consistency: Cxy == Cyx^T
    cyx = cov_arr[nx:, :nx]
    if not np.allclose(cxy, cyx.T, atol=atol_sym, rtol=rtol_sym):
        max_cross = float(np.max(np.abs(cxy - cyx.T)))
        raise ValueError(
            "Cross-covariance blocks inconsistent with [x,y] stacking: "
            "expected cov[:nx,nx:] == cov[nx:,:nx].T within tolerance. "
            f"max diff={max_cross:g} (atol={atol_sym:g}, rtol={rtol_sym:g})."
        )

    return cxx, cxy, cyy


def _reorder_cov_to_xy(
    cov: NDArray[np.float64],
    *,
    x_idx: NDArray[np.int64],
    y_idx: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Reorders a full covariance matrix into the stacked ``[x, y]`` convention.

    This helper reindexes the input covariance so that the returned matrix is
    ordered as ``[x, y]``, where the first block corresponds to indices ``x_idx``
    and the second block corresponds to indices ``y_idx``. It is intended to
    support cases where the original covariance uses a different ordering than
    the required ``[x, y]`` stacking convention.

    Args:
        cov: Full 2D covariance matrix to reorder.
        x_idx: Integer indices selecting the x components in the original ordering.
        y_idx: Integer indices selecting the y components in the original ordering.

    Returns:
        A reordered covariance matrix with the same shape as cov, where rows and
        columns are permuted so the stacked order is ``[x, y]``.

    Raises:
        ValueError: If cov is not a square 2D matrix, if indices are not 1D, are
            out of range, overlap, or do not cover all covariance dimensions
            exactly once.
    """
    cov = np.asarray(cov, dtype=np.float64)
    validate_covariance_matrix_shape(cov)

    x_idx = np.asarray(x_idx, dtype=np.int64).ravel()
    y_idx = np.asarray(y_idx, dtype=np.int64).ravel()

    idx = np.concatenate([x_idx, y_idx])
    n = int(cov.shape[0])

    if idx.size != n:
        raise ValueError(
            "x_idx and y_idx must partition cov dimension exactly: "
            f"len(x_idx)+len(y_idx)={idx.size} vs cov.shape[0]={n}."
        )
    if idx.min(initial=0) < 0 or idx.max(initial=0) >= n:
        raise ValueError("x_idx/y_idx contain out-of-range indices.")
    if np.unique(idx).size != n:
        raise ValueError("x_idx and y_idx must be disjoint and"
                         " cover all indices exactly once.")

    return cov[np.ix_(idx, idx)]


def as_1d_data_vector(y: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Converts a model output into a 1D data vector.

    This function standardizes model outputs so downstream code can treat them as a
    single data vector. Scalars are converted to length-1 arrays. Array outputs are
    returned as 1D arrays, flattening higher-rank inputs in row-major ("C") order.

    Args:
        y: Model output to convert. May be a scalar or an array-like object of any
            shape.

    Returns:
        A 1D float64 NumPy array representing the model output as a single data
        vector.
    """
    arr = np.asarray(y, dtype=np.float64)

    if arr.ndim == 0:
        return arr[None]
    if arr.ndim == 1:
        return arr
    return arr.ravel(order="C")
