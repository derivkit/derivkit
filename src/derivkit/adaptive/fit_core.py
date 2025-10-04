"""Core polynomial-fitting utilities used by the adaptive estimator.

All fits are performed in normalized coordinates. The input values are
shifted by the expansion point and then divided by the maximum absolute
deviation, so that the normalized range is typically between minus one
and one. This normalization improves numerical stability when fitting
polynomials.

Derivatives with respect to the original variable can be recovered from
the fitted polynomial in normalized space by rescaling with the same
normalization factor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .weights import inverse_distance_weights

__all__ = [
    "normalize_coords",
    "polyfit_u",
    "residuals_relative",
    "fit_once",
    "derivative_at_x0",
    "residual_to_signal",
]


def normalize_coords(x_vals: np.ndarray, x0: float) -> tuple[np.ndarray, float]:
    """Normalize coordinates around a given center point.

    The input values are shifted by ``x0`` and then divided by the maximum
    absolute deviation from that point. This produces normalized coordinates
    that typically range between minus one and one. The method also returns
    the scaling factor that was applied, with a small lower bound to avoid
    division by zero.

    Normalization improves the numerical stability of polynomial fits and
    provides a straightforward way to convert derivatives from the normalized
    space back into the original variable.

    Args:
      x_vals: The sample input values.
      x0: The center point for normalization.

    Returns:
      tuple[np.ndarray, float]: A pair consisting of the normalized coordinates
      and the scaling factor that was applied.
    """
    t = np.asarray(x_vals, dtype=float) - float(x0)
    h = float(np.max(np.abs(t))) if t.size else 0.0
    h = max(h, 1e-12)
    return t / h, h


def polyfit_u(
    u_vals: np.ndarray,
    y_vals: np.ndarray,
    order: int,
    weights: np.ndarray,
) -> Optional[np.poly1d]:
    """Fit a weighted polynomial in normalized coordinates.

    A polynomial is fit to the data in normalized coordinates using
    ``np.polyfit`` and optional per-sample weights. The polynomial is
    defined in the normalized space, so its coefficients are not directly
    in terms of the original input values. To obtain derivatives with
    respect to the original variable at the expansion point, you need
    to rescale using the normalization factor returned by
    ``normalize_coords``.

    Args:
      u_vals: The normalized x-coordinates (usually scaled to lie between
        minus one and one).
      y_vals: The corresponding y-values of the samples.
      order: The degree of the polynomial to fit.
      weights: Per-sample weights to apply in the fit.

    Returns:
      np.poly1d | None: A polynomial model in the normalized coordinates
      if the fit succeeds, or ``None`` if the system is singular or the
      fit fails.
    """
    try:
        coeffs = np.polyfit(
            np.asarray(u_vals, float),
            np.asarray(y_vals, float),
            deg=order,
            w=np.asarray(weights, float),
        )
        return np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        return None


def residuals_relative(
    y_fit: np.ndarray,
    y_true: np.ndarray,
    floor: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Compute elementwise relative residuals and their maximum.

    Each residual is measured as the absolute difference between the
    predicted and observed values, divided by the larger of the absolute
    observed value and a small floor value. This prevents the ratio from
    blowing up when the observed value is very close to zero.

    Args:
      y_fit: Model predictions at the sample points.
      y_true: Observed values at the sample points.
      floor: Small positive number used as a lower bound in the denominator
        to avoid division by values near zero.

    Returns:
      tuple[np.ndarray, float]: A pair consisting of:
        - an array of elementwise relative residuals,
        - the maximum residual across all elements (or ``0.0`` if the array
          is empty).
    """
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    safe = np.maximum(np.abs(y_true), floor)
    resid = np.abs(y_fit - y_true) / safe
    rel_error = float(np.max(resid)) if resid.size else 0.0
    return resid, rel_error


def fit_once(
    x0: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    order: int,
    *,
    weight_eps_frac: float = 1e-3,
) -> Dict[str, Any]:
    """Perform one weighted polynomial fit in normalized coordinates.

    Steps:
      1) Normalize: compute ``u = (x − x0) / h`` and record the scale ``h``.
      2) Weight: build inverse-distance weights around ``x0``.
      3) Fit: obtain ``poly_u(u)`` with ``np.polyfit`` in normalized space.
      4) Diagnose: compute fitted values and relative residuals.

    Note:
      Derivatives in the original variable are obtained via
      ``d^m y/dx^m = poly_u^(m)(0) / h**m``.

    Args:
      x0: Expansion point used for normalization.
      x_vals: Sample abscissae.
      y_vals: Sample ordinates.
      order: Polynomial degree (also the derivative order extracted later).
      weight_eps_frac: Epsilon fraction for inverse-distance weights.

    Returns:
      Dict[str, Any]: Keys include:
        - ``ok`` (bool): Fit succeeded.
        - ``reason`` (str | None): Failure reason if any.
        - ``h`` (float): Normalization scale.
        - ``poly_u`` (np.poly1d | None): Polynomial in normalized coords.
        - ``y_fit`` (np.ndarray | None): Fitted values at ``u_vals``.
        - ``residuals`` (np.ndarray | None): Relative residuals.
        - ``rel_error`` (float): Maximum relative residual.
    """
    u_vals, h = normalize_coords(x_vals, x0)
    weights = inverse_distance_weights(x_vals, x0, eps_frac=weight_eps_frac)
    poly_u = polyfit_u(u_vals, y_vals, order, weights)
    if poly_u is None:
        return {"ok": False, "reason": "singular_normal_equations"}
    y_fit = poly_u(u_vals)
    resid, rel_error = residuals_relative(y_fit, y_vals, floor=1e-8)
    return {
        "ok": True,
        "reason": None,
        "h": h,
        "poly_u": poly_u,
        "y_fit": y_fit,
        "residuals": resid,
        "rel_error": rel_error,
    }


def derivative_at_x0(poly_u: np.poly1d, h: float, order: int) -> float:
    """Return d^order y/dx^order at x0 from poly in normalized coords."""
    return float(poly_u.deriv(m=order)(0.0) / (h ** order))


def residual_to_signal(y_fit: np.ndarray, y_true: np.ndarray, *, floor: float = 1e-12) -> tuple[float, float, float]:
    """Return (rho, rms_resid, signal_scale) with a robust local signal scale."""
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    diff = y_fit - y_true
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    signal = float(np.maximum(np.median(np.abs(y_true)), floor))
    rho = 0.0 if signal == 0.0 else (rms / signal)
    return rho, rms, signal
