"""Utilities for polynomial fitting and evaluation."""

from __future__ import annotations

from math import factorial

import numpy as np

__all__ = [
    "choose_degree",
    "scale_offsets",
    "fit_multi_power",
    "extract_derivative",
]


def _vandermonde(t: np.ndarray, deg: int) -> np.ndarray:
    """Return the Vandermonde matrix for 1D inputs in the power basis.

    Args:
        t: 1D array of shape (n_points,).
        deg: Polynomial degree.

    Returns:
        np.ndarray: Matrix of shape (n_points, deg+1) with columns [1, t, t**2, ..., t**deg].

    Raises:
        ValueError: If `t` is not 1D or `deg` < 0.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1D.")
    if deg < 0:
        raise ValueError("deg must be >= 0.")

    return np.vander(t, N=deg + 1, increasing=True)


def choose_degree(order: int, n_pts: int, extra: int = 5) -> int:
    """Choose a polynomial degree given derivative order and sample size.

    Selects ``min(order + extra, n_pts - 1)`` to avoid underdetermined fits while
    allowing some flexibility beyond the target derivative order.

    Args:
        order: Derivative order (>= 0).
        n_pts: Number of available points (>= 1).
        extra: Extra degrees beyond ``order`` (>= 0). Default is 5.

    Returns:
        int: Chosen polynomial degree.

    Raises:
        ValueError: If ``order < 0``, ``n_pts < 1``, or ``extra < 0``.
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    if n_pts < 1:
        raise ValueError("n_pts must be >= 1")
    if extra < 0:
        raise ValueError("extra must be >= 0")

    return min(order + extra, n_pts - 1)


def scale_offsets(t: np.ndarray) -> tuple[np.ndarray, float]:
    """Rescale offsets to improve numerical stability.

    Converts offsets `t` to `u = t/s`, where `s = max(|t|)` (or `1` if `t` is
    empty or all zeros). This mitigates instability in polynomial fitting and
    differentiation, where powers of `t` can become very large or very small.

    Args:
        t: 1D array of offsets (can be empty).

    Returns:
        u: Scaled offsets, same shape as `t`.
        s: Positive scaling factor.

    Raises:
        ValueError: If `t` is not 1D.
    """
    t = np.asarray(t, dtype=float)
    s = float(np.max(np.abs(t))) if t.size else 1.0
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    return t / s, s


def fit_multi_power(
    u: np.ndarray, y: np.ndarray, deg: int, ridge: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a least-squares polynomial fit in the power basis for multiple components.

    This is a vectorized version of `fit_and_rel_rms_multi` using the Vandermonde matrix.

    Args:
        u: 1D array of scaled independent variable values (n_pts,).
        y: 2D array of dependent variable values (n_pts, n_comp).
        deg: Degree of polynomial to fit (integer, >= 0).
        ridge: Optional ridge regularization parameter (default 0.0).

    Returns:
        C: Array of shape (deg+1, n_comp) with power-basis coefficients.
        rrms: Array of shape (n_comp,) with relative RMS errors.

    Raises:
        ValueError: If inputs have wrong shapes/lengths or degree is invalid.
        TypeError: If `deg` is not an integer.
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)

    if u.ndim != 1:
        raise ValueError("u must be 1D.")
    if y.ndim != 2:
        raise ValueError("y must be 2D (n_pts, n_comp).")

    if y.shape[0] != u.size:
        raise ValueError("len(u) must match y.shape[0].")
    if deg < 0 or deg >= u.size:
        raise ValueError("deg must be in [0, n_pts-1].")

    vander = np.vander(u, N=deg + 1, increasing=True)
    u, s, vt = np.linalg.svd(vander, full_matrices=False)
    if ridge and ridge > 0.0:
        s_filtered = s / (s * s + ridge)
    else:
        s_filtered = np.where(s > 0, 1.0 / s, 0.0)
    coeffs = (vt.T * s_filtered) @ (u.T @ y)

    res = y - vander @ coeffs
    rms = np.sqrt(np.mean(res * res, axis=0))
    yc = y - np.mean(y, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(yc * yc, axis=0)) + 1e-15
    rrms = rms / scale
    return coeffs, rrms


def extract_derivative(
    coeffs: np.ndarray, order: int, scale: float
) -> np.ndarray:
    """Extract the derivative of given order from power-basis coefficients.

    Args:
        coeffs: array of shape (deg+1, n_comp) with power-basis coefficients
        order: derivative order (>= 0)
        scale: scaling factor used in offsets (s > 0)

    Returns:
        deriv: array of shape (n_comp,) with the estimated derivative values

    Raises:
        ValueError: if order < 0 or scale <= 0 or C has invalid shape
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("scale must be > 0 and finite.")

    a_m = coeffs[order, :]
    return (factorial(order) * a_m) / (scale**order)
