"""Diagnostics for local polynomial derivative estimation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from derivkit.local_polynomial_derivative.fit import design_matrix
from derivkit.local_polynomial_derivative.local_poly_config import LocalPolyConfig

__all = ["make_diag"]


def make_diag(
    x0: float,
    config: LocalPolyConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    keep: np.ndarray,
    coeffs: np.ndarray,
    degree: int,
    order: int,
    ok: bool,
) -> Dict[str, Any]:
    """Builds diagnostics dictionary.

    Args:
        x0:
            The center point for polynomial fitting.
        config:
            LocalPolyConfig instance with fitting settings.
        xs:
            An array of sample points (shape (n_samples,)).
        ys:
            An array of function evaluations (shape (n_samples, n_components)).
        keep:
            A boolean array indicating which samples were used (shape (n_samples,)).
        coeffs:
            The polynomial coefficients (shape (degree + 1, n_components)).
        degree:
            The degree of the polynomial fit.
        order:
            The order of the derivative being estimated.
        ok:
            Whether the fit met the residual tolerances.

    Returns:
        A diagnostics dictionary.
    """
    used_x = xs[keep]
    used_y = ys[keep]

    if used_x.size:
        mat = design_matrix(x0, config, used_x, degree)
        y_fit = mat @ coeffs
        denom = np.maximum(np.abs(used_y), config.tol_abs)
        err = np.abs(y_fit - used_y) / denom
        max_err = float(err.max())
    else:
        max_err = float("nan")

    diag: Dict[str, Any] = {
        "ok": bool(ok),
        "x0": float(x0),
        "degree": int(degree),
        "order": int(order),
        "n_all": int(xs.size),
        "n_used": int(keep.sum()),
        "x_used": used_x.tolist(),
        "max_rel_err_used": max_err,
        "tol_rel": float(config.tol_rel),
        "tol_abs": float(config.tol_abs),
        "min_samples": int(config.min_samples),
        "max_trim": int(config.max_trim),
        "center": bool(config.center),
        "coeffs": coeffs.tolist(),
    }

    if not ok:
        diag["note"] = (
            "No interval fully satisfied residual tolerances; derivative is taken "
            "from the last polynomial fit and should be treated with caution."
        )

    return diag
