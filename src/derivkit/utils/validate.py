"""Validation utilities for DerivativeKit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.utils.sandbox import get_partial_function

__all__ = [
    "is_finite_and_differentiable",
    "check_scalar_valued",
    "validate_tabulated_xy",
    "validate_covariance_matrix_shape",
    "validate_symmetric_psd",
]

def is_finite_and_differentiable(
    function: Callable[[float], Any],
    x: float,
    delta: float = 1e-5,
) -> bool:
    """Check that ``function`` is finite at ``x`` and ``x + delta``.

    Evaluates without exceptions and returns finite values at both points.

    Args:
      function: Callable ``f(x)`` returning a scalar or array-like.
      x: Probe point.
      delta: Small forward step.

    Returns:
      True if finite at both points; otherwise False.
    """
    f0 = np.asarray(function(x))
    f1 = np.asarray(function(x + delta))
    return np.isfinite(f0).all() and np.isfinite(f1).all()


def check_scalar_valued(function, theta0: np.ndarray, i: int, n_workers: int):
    """Helper used by ``build_gradient`` and ``build_hessian``.

    Args:
        function (callable): The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return a scalar observable value.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)

    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "build_gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )


def validate_tabulated_xy(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Validates and converts tabulated ``x`` and ``y`` arrays into NumPy arrays.

    Requirements:
      - ``x`` is 1D and strictly increasing.
      - ``y`` has at least 1 dimension.
      - ``y.shape[0] == x.shape[0]``, but ``y`` may have arbitrary trailing
        dimensions (scalar, vector, or ND output).

    Args:
        x: 1D array-like of x values (must be strictly increasing).
        y: Array-like of y values with ``y.shape[0] == len(x)``.

    Returns:
        Tuple of (x_array, y_array) as NumPy arrays.

    Raises:
        ValueError: If input arrays do not meet the required conditions.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1D.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length along axis 0.")
    if not np.all(np.diff(x_arr) > 0):
        raise ValueError("x must be strictly increasing.")
    if y_arr.ndim < 1:
        raise ValueError("y must be at least 1D.")

    return x_arr, y_arr


def validate_covariance_matrix_shape(cov: ArrayLike) -> NDArray[np.floating]:
    """Validates covariance input shape: allows 0D/1D/2D; if 2D requires square."""
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim > 2:
        raise ValueError(f"cov must be at most two-dimensional; got ndim={cov_arr.ndim}.")
    if cov_arr.ndim == 2 and cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"cov must be square; got shape={cov_arr.shape}.")
    return cov_arr


def validate_symmetric_psd(
    matrix: ArrayLike,
    *,
    sym_atol: float = 1e-12,
    psd_atol: float = 1e-12,
) -> NDArray[np.floating]:
    """Validates that an input is a symmetric positive semidefinite (PSD) matrix.

    This is intended for strict validation (e.g., inputs passed to GetDist, or any
    code path where an indefinite "covariance-like" matrix should hard-fail). This
    is an important valdaition because many algorithms assume PSD inputs, and
    invalid inputs can lead to silent failures or nonsensical results.

    Policy:
      - Requires 2D square shape.
      - Requires near-symmetry within ``sym_atol`` (raises if violated).
      - Checks PSD by computing eigenvalues of the symmetrized matrix
        ``S = 0.5 * (A + A.T)`` and requiring ``min_eig >= -psd_atol``.

    Args:
        matrix: Array-like input expected to be a covariance-like matrix.
        sym_atol: Absolute tolerance for symmetry check. If ``max(|A-A^T|) > sym_atol``,
            this raises ``ValueError``.
        psd_atol: Absolute tolerance for PSD check. Allows small negative eigenvalues
            down to ``-psd_atol`` (useful for roundoff).

    Returns:
        A NumPy array view/copy of the input, converted to ``float`` (same values as input).
        Note: this function does not modify the returned matrix (it does *not* symmetrize it);
        the PSD check is performed on the symmetrized form only.

    Raises:
        ValueError: If ``matrix`` is not 2D square, is too asymmetric, contains non-finite
            values, or is not PSD within tolerance.
    """
    a = np.asarray(matrix, dtype=float)

    if a.ndim != 2:
        raise ValueError(f"matrix must be 2D; got ndim={a.ndim}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square; got shape={a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix contains non-finite values.")

    # Symmetry check (strict)
    skew = a - a.T
    max_abs_skew = float(np.max(np.abs(skew))) if skew.size else 0.0
    if max_abs_skew > sym_atol:
        raise ValueError(
            f"matrix must be symmetric within sym_atol={sym_atol:.2e}; "
            f"max(|A-A^T|)={max_abs_skew:.2e}."
        )

    # PSD check (numerically robust): eigenvalues of symmetrized matrix
    s = 0.5 * (a + a.T)
    try:
        evals = np.linalg.eigvalsh(s)
    except np.linalg.LinAlgError as e:
        raise ValueError("eigenvalue check failed for matrix (LinAlgError).") from e

    min_eig = float(np.min(evals)) if evals.size else 0.0
    if min_eig < -psd_atol:
        raise ValueError(
            f"matrix is not PSD within psd_atol={psd_atol:.2e}; min eigenvalue={min_eig:.2e}."
        )

    return a
