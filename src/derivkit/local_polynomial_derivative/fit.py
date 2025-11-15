"""Local polynomial fitting with outlier trimming."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from derivkit.local_polynomial_derivative.local_poly_config import (
    LocalPolyConfig,
)

__all__ = ["design_matrix", "trimmed_polyfit"]


def design_matrix(
        x0: float,
        config: LocalPolyConfig,
        sample_points: np.ndarray,
        degree: int) -> np.ndarray:
    """Builds a Vandermonde design matrix.

    This method constructs the Vandermonde matrix for polynomial fitting based
    on whether centering around x0 is specified in the config.

    Args:
        x0:
            The center point for polynomial fitting.
        config:
            LocalPolyConfig instance with fitting settings.
        sample_points:
            An array of sample points (shape (n_samples,)).
        degree:
            The degree of the polynomial to fit.

    Returns:
        A Vandermonde matrix (shape (n_samples, degree + 1)).
    """
    if config.center:
        z = sample_points - x0
    else:
        z = sample_points
    return np.vander(z, N=degree + 1, increasing=True)


def trimmed_polyfit(
    x0: float,
    config: LocalPolyConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Returns a polynomial fit with trimmed outliers.

    Args:
        x0:
            The center point for polynomial fitting.
        config:
            LocalPolyConfig instance with fitting settings.
        xs:
            An array of sample points (shape (n_samples,)).
        ys:
            An array of function evaluations (shape (n_samples, n_components)).
        degree:
            The degree of the polynomial to fit.

    Returns:
        coeffs : (degree+1, n_comp)
        used_mask : (n_samples,) bool
        ok : bool (True if residuals within tolerances on final mask)
    """
    n_samples, n_comp = ys.shape
    keep = np.ones(n_samples, dtype=bool)
    n_trim = 0

    last_coeffs = None
    last_keep = keep.copy()
    last_ok = False

    needed = max(config.min_samples, degree + 1)

    while keep.sum() >= needed and n_trim <= config.max_trim:
        idx = np.where(keep)[0]
        x_use = xs[idx]
        y_use = ys[idx]

        mat = design_matrix(x0, config, x_use, degree)
        coeffs, *_ = np.linalg.lstsq(mat, y_use, rcond=None)

        y_fit = mat @ coeffs
        denom = np.maximum(np.abs(y_use), config.tol_abs)
        err = np.abs(y_fit - y_use) / denom

        bad_rows = (err > config.tol_rel).any(axis=1)
        if not bad_rows.any():
            last_coeffs = coeffs
            last_keep = keep.copy()
            last_ok = True
            break

        bad_idx_all = idx[bad_rows]
        leftmost, rightmost = idx[0], idx[-1]
        trimmed = False

        # shave edges only if we keep enough for this degree
        if bad_idx_all[0] == leftmost and keep.sum() - 1 >= needed:
            keep[leftmost] = False
            trimmed = True
        if bad_idx_all[-1] == rightmost and keep.sum() - 1 >= needed:
            keep[rightmost] = False
            trimmed = True

        if not trimmed:
            last_coeffs = coeffs
            last_keep = keep.copy()
            last_ok = False
            break

        last_coeffs = coeffs
        last_keep = keep.copy()
        last_ok = False
        n_trim += 1

    if last_coeffs is None:
        last_coeffs = np.zeros((degree + 1, n_comp), dtype=float)
        last_keep = keep.copy()
        last_ok = False

    return last_coeffs, last_keep, last_ok
