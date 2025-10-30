"""Utility functions for Gaussian Process regression."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.linalg import LinAlgError, cholesky, solve

from derivkit.gaussian_process.kernels.base import Kernel

__all__ = [
    "neg_log_marginal_likelihood",
    "standardize_targets",
    "take_local_window",
    "try_cholesky",
    "chol_solve",
    "to_psd",
    "default_grid",
    "resolve_half_width",
    "clamp_params_for_span",
    "ensure_min_noise",
    "standardize_xy",
    "span_width",
]


def neg_log_marginal_likelihood(
    theta_log: np.ndarray,
    training_inputs: np.ndarray,
    targets: np.ndarray,
    kernel: Kernel,
    init_params: dict,
    normalize: bool = True,
) -> float:
    """Computes the negative log marginal likelihood (NLML) for a GP with isotropic or ARD length scale.

    Packs hyperparameters in log-space and evaluates the GP evidence term given
    training inputs, targets, and a kernel. If ``init_params['length_scale']`` is
    an array (ARD), that length scale is held fixed and only ``output_scale`` and
    ``noise`` are read from ``theta_log``; otherwise an isotropic ``length_scale``
    is optimized alongside ``output_scale`` and ``noise``. Optionally mean-centers
    targets when ``normalize=True``.

    Args:
      theta_log: Log-parameters to evaluate. For isotropic: ``[log_length_scale, log_output_scale, log_noise]``.
        For ARD: ``[ignored, log_output_scale, log_noise]`` (indexing kept for consistency).
      training_inputs: Training design matrix with shape ``(n_samples, d)``.
      targets: Training targets with shape ``(n_samples,)`` or ``(n_samples, 1)``.
      kernel: Kernel implementing ``cov_value_value(X, Xp, params) -> ndarray``.
      init_params: Baseline hyperparameters (e.g., ``{"length_scale": 1.0, "output_scale": 1.0}``).
        If ``length_scale`` is an array, ARD mode is assumed and that array is held fixed.
      normalize: If ``True``, subtracts the mean of ``targets`` before evaluation.

    Returns:
      float: The NLML value (lower is better). Returns ``np.inf`` if Cholesky
      decomposition fails for the implied kernel matrix.

    Notes:
      - The kernel matrix is ``K = k(X, X; params) + noise * I`` with ``noise = exp(theta_log[idx])``.
      - Uses a stabilized Cholesky factorization; failure yields ``np.inf`` to steer optimizers away.
      - ``output_scale`` multiplies the kernel output; it does not square it internally here.

    Raises:
      ValueError: If shapes are incompatible or parameters are ill-formed (may be raised by NumPy/LA ops).
    """
    x = np.atleast_2d(training_inputs)
    y = np.asarray(targets).reshape(-1)

    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("training_inputs must be 2D and targets 1D.")
    if x.shape[0] != y.size:
        raise ValueError("Row count of training_inputs must match length of targets.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Inputs and targets must be finite.")

    # theta_log size checks
    if np.ndim(init_params.get("length_scale", 1.0)) > 0:
        if theta_log.size < 3:
            raise ValueError("theta_log must have at least 3 entries for ARD: [*, log_output_scale, log_noise].")
    else:
        if theta_log.size != 3:
            raise ValueError(
                "theta_log must be length 3 for isotropic: [log_length_scale, log_output_scale, log_noise].")

    # validate init_params length_scale positivity (scalar or array)
    ls0 = np.asarray(init_params.get("length_scale", 1.0), float)
    if not np.all(ls0 > 0):
        raise ValueError("init_params['length_scale'] must be positive (scalar or array).")

    y_use = y - np.mean(y) if normalize else y

    # Resolve params
    ard = np.ndim(init_params.get("length_scale", 1.0)) > 0
    if ard:
        output_scale = float(np.exp(theta_log[1]))
        noise = float(np.exp(theta_log[2]))
        params = {"length_scale": init_params["length_scale"], "output_scale": output_scale}
    else:
        length_scale = float(np.exp(theta_log[0]))
        output_scale = float(np.exp(theta_log[1]))
        noise = float(np.exp(theta_log[2]))
        params = {"length_scale": length_scale, "output_scale": output_scale}

    # resolve noise; ensure nonnegative
    if noise < 0.0 or not np.isfinite(noise):
        raise ValueError("noise (variance) must be finite and nonnegative.")

    # ensure kernel output is finite and symmetric to numerical tolerance
    k_xx = kernel.cov_value_value(x, x, params)
    if not np.all(np.isfinite(k_xx)):
        raise ValueError("Kernel covariance returned non-finite values.")
    if k_xx.shape[0] != k_xx.shape[1]:
        raise ValueError("Kernel covariance must be square.")
    k_xx = 0.5 * (k_xx + k_xx.T)  # symmetrize
    k_xx = k_xx + noise * np.eye(k_xx.shape[0])
    try:
        l_factor = cholesky(k_xx)
    except LinAlgError:
        return float("inf")

    alpha = solve(l_factor.T, solve(l_factor, y_use))
    logdet = 2.0 * np.sum(np.log(np.diag(l_factor)))
    n = y_use.size
    return 0.5 * y_use @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)


def standardize_targets(targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean-center a 1D target vector and returns the centered values and original mean.

    Converts the input to a 1D array, computes its mean, and subtracts it to
    produce zero-mean targets—useful for numerically stable GP fitting.

    Args:
      targets: Array-like targets of shape ``(n,)`` or ``(n, 1)``.

    Returns:
      tuple: ``(y_centered, mean)`` where ``y_centered`` has
      shape ``(n,)`` and ``mean`` is the scalar average of the input.
    """
    y = np.asarray(targets).reshape(-1)
    if y.size == 0:
        raise ValueError("targets must be non-empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("targets must be finite.")

    mu = float(np.mean(y))
    return y - mu, mu


def take_local_window(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    *,
    frac_span: float = 0.35,
    min_pts: int = 9,
):
    """Selects samples within a local span around a reference point.

    Keeps only training points whose x-values lie within ``frac_span`` times the
    total span of the data, centered on ``x0``. This focuses the GP fit on a
    neighborhood around the expansion point. If too few points fall inside, it
    falls back to taking the ``min_pts`` nearest samples.

    Args:
      x: Input samples with shape ``(n_samples, d)``.
      y: Corresponding target values with shape ``(n_samples,)``.
      x0: Reference location around which to select the window.
      frac_span: Fraction of the total x-span to include on each side of ``x0``.
        Defaults to ``0.35``.
      min_pts: Minimum number of samples to keep, even if the span-based window
        would include fewer.

    Returns:
      tuple[np.ndarray, np.ndarray]: Filtered ``(x_window, y_window)`` arrays,
      preserving order from the original inputs.
    """
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x must be 2D (n_samples, d).")
    if x.shape[0] != y.size:
        raise ValueError("Length of y must match number of rows in x.")
    if x.shape[1] < 1:
        raise ValueError("x must have at least one column.")

    x_col = x[:, 0]
    span = float(x_col.max() - x_col.min()) or 1.0
    half = max(frac_span * span, 1e-12)

    mask = np.abs(x_col - x0) <= half
    idx = np.nonzero(mask)[0]

    if idx.size < min_pts:
        order = np.argsort(np.abs(x_col - x0))
        idx = order[:max(min_pts, min(len(x_col), min_pts))]

    return x[idx], y[idx]


def try_cholesky(
    mat: np.ndarray, *, jitter0: float = 1e-12, max_tries: int = 6
) -> Tuple[np.ndarray, float]:
    """Computes a Cholesky factor with automatic jitter escalation for numerical stability.

    Tries to factorize ``mat`` by adding a diagonal jitter term that grows by
    ×10 each failure, up to ``max_tries`` attempts. This is useful when small
    negative eigenvalues from rounding cause PSD violations.

    Args:
      mat: Square matrix expected to be symmetric (approximately PSD).
      jitter0: Initial diagonal jitter to add (in matrix units).
      max_tries: Maximum number of jitter escalation attempts (each ×10).

    Returns:
      tuple[np.ndarray, float]: ``(L, jitter)`` where ``L`` is the lower-triangular
      Cholesky factor of ``mat + jitter*I``, and ``jitter`` is the final value used.

    Raises:
      LinAlgError: If the final attempt still fails to factorize.

    Notes:
      - Symmetry is not enforced here; call ``0.5*(A + A.T)`` beforehand if needed.
      - The returned ``jitter`` can inform downstream noise floors or diagnostics.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat must be a square 2D array.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("mat must be finite.")
    if jitter0 <= 0:
        raise ValueError("jitter0 must be positive.")
    if max_tries < 0:
        raise ValueError("max_tries must be >= 0.")

    eye_mat = np.eye(mat.shape[0], dtype=mat.dtype)
    jitter = float(jitter0)
    for _ in range(max_tries):
        try:
            chol_lower = cholesky(mat + jitter * eye_mat)
            return chol_lower, jitter
        except LinAlgError:
            jitter *= 10.0
    chol_lower = cholesky(mat + jitter * eye_mat)
    return chol_lower, jitter


def chol_solve(chol_lower: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solves a linear system using a Cholesky factor.

    Given a lower-triangular Cholesky factor of a positive-definite matrix,
    this solves the system for the unknowns by forward and backward substitution.

    Args:
      chol_lower: Lower-triangular Cholesky factor (from factoring the system matrix).
      b: Right-hand side vector or matrix.

    Returns:
      np.ndarray: Solution array with the same trailing shape as ``b``.

    Notes:
      - ``chol_lower`` should be from a successful Cholesky factorization of the
        system matrix. No symmetry checks are performed here.
      - Raises ``LinAlgError`` if the inputs are incompatible (via ``numpy.linalg.solve``).
    """
    if chol_lower.ndim != 2 or chol_lower.shape[0] != chol_lower.shape[1]:
        raise ValueError("chol_lower must be a square 2D array.")
    if chol_lower.shape[0] != b.shape[0]:
        raise ValueError("Incompatible shapes: chol_lower and b.")
    if not (np.all(np.isfinite(chol_lower)) and np.all(np.isfinite(b))):
        raise ValueError("chol_lower and b must be finite.")

    y = solve(chol_lower, b)
    x = solve(chol_lower.T, y)
    return x


def to_psd(mat: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Returns a symmetric, positive-semidefinite copy of a matrix by eigenvalue clipping.

    The input is first symmetrized as ``0.5*(A + A.T)`` to remove small asymmetries,
    then eigen-decomposed; any negative (or too-small) eigenvalues are clipped to
    ``floor`` and the matrix is reconstructed. This is useful to sanitize nearly-PSD
    kernels or covariance matrices affected by round-off.

    Args:
      mat: Square array to sanitize (approximately symmetric).
      floor: Minimum eigenvalue after clipping. Use a small positive value to
        enforce strict PSD in numerically noisy settings.

    Returns:
      np.ndarray: Symmetric PSD matrix of the same shape as ``mat``.

    Notes:
      - For large matrices, this costs an eigen-decomposition (``O(n^3)``).
      - If exact symmetry is required, the output is guaranteed symmetric by construction.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat must be a square 2D array.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("mat must be finite.")
    if floor < 0:
        raise ValueError("floor must be >= 0.")

    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, float(floor))
    return (eigvecs * eigvals) @ eigvecs.T


def default_grid(x0: float, half_width: float, n_points: int) -> np.ndarray:
    """Generates a symmetric 1D grid of sample points around a reference value.

    Creates evenly spaced coordinates centered on ``x0`` and extending
    ``±half_width`` on both sides. The result is formatted as a column vector
    for consistency with Gaussian Process training input shapes.

    Args:
      x0: Center of the grid.
      half_width: Half of the total span to cover on each side of ``x0``.
      n_points: Number of grid points to generate (must be ≥ 2).

    Returns:
      np.ndarray: Grid array of shape ``(n_points, 1)`` containing the sampled positions.

    Example:
      >>> default_grid(0.0, 0.5, 5).ravel()
      array([-0.5 , -0.25,  0.  ,  0.25,  0.5])
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if half_width <= 0 or not np.isfinite(half_width):
        raise ValueError("half_width must be positive and finite.")
    if not np.isfinite(x0):
        raise ValueError("x0 must be finite.")

    return (np.linspace(-half_width, half_width, n_points) + x0)[:, None]


def resolve_half_width(x0: float, spacing: float | str, base_abs: float) -> float:
    """Determines a reasonable half-width for building a local sampling grid.

    If ``spacing`` is a numeric value, it is returned directly. Otherwise,
    a heuristic is applied that scales with the magnitude of ``x0`` to ensure
    the grid covers a meaningful neighborhood even for large or small values.

    Args:
      x0: Reference point around which the grid will be centered.
      spacing: Either a numeric half-width or the string ``"auto"`` to trigger
        heuristic scaling.
      base_abs: Baseline minimum half-width to use when ``spacing="auto"``.

    Returns:
      float: The resolved half-width for constructing a symmetric grid.
    """
    if isinstance(spacing, (int, float)):
        spacing = float(spacing)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("numeric spacing must be positive and finite.")
        return spacing
    if base_abs <= 0 or not np.isfinite(base_abs):
        raise ValueError("base_abs must be positive and finite.")
    if not np.isfinite(x0):
        raise ValueError("x0 must be finite.")

    scale = max(1.0, abs(x0))
    return max(base_abs, 0.25 * scale)


def standardize_xy(x_mat: np.ndarray, y_vec: np.ndarray):
    """Standardizes inputs and targets and return the standardized data plus their means and scales.

    Computes the mean and standard deviation of the first input column and the
    targets, applies z-scoring to each, and returns both standardized arrays.
    If a computed standard deviation is zero or not finite, it falls back to 1.0
    to avoid division by zero.

    Args:
      x_mat: Input design matrix of shape ``(n_samples, d)``. Only the first
        column is standardized; other columns are copied unchanged.
      y_vec: Target values of shape ``(n_samples,)`` or ``(n_samples, 1)``.

    Returns:
      tuple: ``(x_std, y_std, mu_x, sigma_x, mu_y, sigma_y)`` where:
        - ``x_std`` is ``x_mat`` with column 0 standardized.
        - ``y_std`` is the standardized target vector.
        - ``mu_x`` and ``sigma_x`` are the mean and scale used for column 0.
        - ``mu_y`` and ``sigma_y`` are the mean and scale used for ``y_vec``.

    Notes:
      - Standardizing only the first column matches common 1D GP setups where
        the derivative is taken with respect to that coordinate.
      - The fallbacks to a scale of 1.0 keep the transformation well-defined
        even for constant inputs or targets.
    """
    x_mat = np.asarray(x_mat, float)
    y_vec = np.asarray(y_vec, float).reshape(-1)
    if x_mat.ndim != 2 or x_mat.shape[0] != y_vec.size:
        raise ValueError("x_mat must be (n_samples, d) and y_vec length must match n_samples.")
    if x_mat.shape[1] < 1:
        raise ValueError("x_mat must have at least one column.")
    if not (np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec))):
        raise ValueError("x_mat and y_vec must be finite.")

    x_col = x_mat[:, 0]
    mu_x = float(np.mean(x_col))
    sigma_x = float(np.std(x_col))
    if not np.isfinite(sigma_x) or sigma_x == 0.0:
        sigma_x = 1.0

    mu_y = float(np.mean(y_vec))
    sigma_y = float(np.std(y_vec))
    if not np.isfinite(sigma_y) or sigma_y == 0.0:
        sigma_y = 1.0

    x_std = x_mat.copy()
    x_std[:, 0] = (x_col - mu_x) / sigma_x
    y_std = (y_vec - mu_y) / sigma_y
    return x_std, y_std, mu_x, sigma_x, mu_y, sigma_y


def span_width(x: np.ndarray) -> float:
    """Returns half the data range.

    Converts the input to a float array, finds the maximum and the minimum,
    subtracts the minimum from the maximum, and divides the result by two.

    Args:
      x: sequence of numeric values.

    Returns:
      float: half of the range.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite.")
    xmax = np.max(x).item()
    xmin = np.min(x).item()
    return 0.5 * (xmax - xmin)


def clamp_params_for_span(
    kernel_params: dict,
    span: float,
    *,
    ls_min_frac: float = 0.05,
    ls_max_frac: float = 3.0,
) -> dict:
    """Clamps kernel hyperparameters to ranges tied to the data span (standardized units).

    Produces a copy of ``kernel_params`` with the ``length_scale`` clipped to
    ``[ls_min_frac * span, ls_max_frac * span]``. This prevents degenerate GP fits:
    a small lower bound helps capture high-frequency behavior, while an upper
    bound avoids overly smooth models. The ``output_scale`` is set to 1.0,
    assuming targets are standardized.

    Args:
      kernel_params: Base kernel parameters (e.g., {"length_scale": 1.0, "output_scale": 1.0}).
        The ``length_scale`` may be a scalar (isotropic) or a 1D array (ARD).
      span: Data span in standardized x units (typically half the total range or similar).
      ls_min_frac: Minimum length scale as a fraction of ``span``. Defaults to 0.05.
      ls_max_frac: Maximum length scale as a fraction of ``span``. Defaults to 3.0.

    Returns:
      dict: A new parameters dictionary with:
        - ``length_scale`` clipped to the specified range.
        - ``output_scale`` set to 1.0 (for standardized targets).

    Notes:
      - If ``length_scale`` is an array (ARD), clipping is applied elementwise.
      - A tiny absolute floor of 1e-8 is enforced on the lower bound to avoid zero.
    """
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be positive and finite.")
    if ls_min_frac <= 0 or ls_max_frac < ls_min_frac:
        raise ValueError("Require 0 < ls_min_frac <= ls_max_frac.")

    out = dict(kernel_params)
    lo = max(ls_min_frac * span, 1e-8)
    hi = max(ls_max_frac * span, lo * 1.01)

    length_scale = np.asarray(out.get("length_scale", 1.0), float)
    if not np.all(np.isfinite(length_scale)):
        raise ValueError("length_scale must be finite.")

    out["length_scale"] = (
        float(np.clip(length_scale, lo, hi)) if length_scale.ndim == 0 else np.clip(length_scale, lo, hi)
    )
    out["output_scale"] = 1.0  # keep standardized
    return out


def ensure_min_noise(noise_var_std: float) -> float:
    """Enforces a small floor on the noise variance in standardized space.

    Clamps the provided variance to at least ``1e-6`` to avoid numerical issues
    (e.g., singular kernel matrices) when fitting or predicting with a GP.

    Args:
      noise_var_std: Noise variance in standardized target units.

    Returns:
      float: ``max(noise_var_std, 1e-6)``.
    """
    if not np.isfinite(noise_var_std):
        raise ValueError("noise_var_std must be finite.")

    return float(max(noise_var_std, 1e-6))
