"""Utilities for polynomial fitting and evaluation."""

from __future__ import annotations

from math import factorial

import numpy as np
import numpy.linalg as npl

from .transforms import pullback_sqrt_at_zero, signed_log_derivatives_to_x

__all__ = [
    "choose_degree",
    "scale_offsets",
    "fit_multi_power",
    "extract_derivative",
    "assess_polyfit_quality",
    "fit_with_headroom_and_maybe_minimize",
    "pullback_derivative_from_fit",
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


def assess_polyfit_quality(
    u: np.ndarray,
    y: np.ndarray,
    coeffs: np.ndarray,
    deg: int,
    ridge: float = 0.0,
    factor: float = 1.0,
    order: int = 0,
) -> tuple[dict[str, float | dict[str, float]], list[str]]:
    """Assess numerical quality of a power-basis polynomial fit.

    Computes several diagnostics for a polynomial fit evaluated on scaled offsets
    ``u``:

    - Relative RMS residual (``rrms_rel``)
    - Leave-one-out (LOO) relative RMSE via the ridge hat matrix (``loo_rel``)
    - Condition number of the Vandermonde design matrix (``cond_vdm``)
    - Relative change of the target derivative compared to a degree-1 refit
      (``deriv_rel``), when feasible

    It also returns human-readable suggestions to improve a poor fit. The thresholds
    are heuristic and may be tuned for your application. The LOO estimate is derived
    from the ridge hat-matrix diagonal, which is a fast approximation and works well
    for spotting overfit/outliers.

    Args:
        u (ndarray): Scaled offsets, shape ``(n,)``.
        y (ndarray): Function values at the sample points, shape ``(n, m)``.
        coeffs (ndarray): Power-basis coefficients of the fit with columns per
            component, shape ``(deg+1, m)``. Column ``k`` corresponds to the
            coefficient of ``u**k``.
        deg (int): Polynomial degree used in the fit (``>= 0``).
        ridge (float): Ridge regularization used in the fit (``>= 0``).
        factor (float): Scaling factor such that original offsets ``t = u * factor``.
            Used only when extracting derivatives in the refit comparison.
        order (int): Derivative order of interest (``>= 0``).

    Returns:
        tuple[dict, list[str]]: A tuple ``(metrics, suggestions)`` where:

            - ``metrics`` is a dict with keys:
                - ``"rrms_rel"`` (float)
                - ``"loo_rel"`` (float)
                - ``"cond_vdm"`` (float)
                - ``"deriv_rel"`` (float)
                - ``"thresholds"`` (dict): contains the threshold values used for
                  each metric (same keys as above)
            - ``suggestions`` is a list of textual recommendations to improve the fit.

    Notes:
        Large values of any metric indicate potential instability. Consider widening
        the sampling window (``spacing``), modestly increasing the number of points,
        or adding light ridge regularization.
    """
    # build design matrix [1, u, u^2, ...]
    design = np.vstack([u ** k for k in range(deg + 1)]).T  # (n, deg+1)

    # predictions and residuals
    y_hat = design @ coeffs  # (n, m)
    resid = y - y_hat
    rrms = np.sqrt(np.mean(resid**2, axis=0))  # (m,)
    y_scale = np.median(np.abs(y), axis=0) + 1e-12
    rrms_rel = float(np.max(rrms / y_scale))

    # ridge hat-matrix diagonal for LOO residuals
    gram = design.T @ design + ridge * np.eye(deg + 1)
    gram_inv = npl.pinv(gram) if ridge == 0.0 else npl.inv(gram)
    design_gram_inv = design @ gram_inv
    h_diag = np.sum(design_gram_inv * design, axis=1)  # (n,)
    loo_resid = resid / (1.0 - h_diag)[:, None]
    loo_rmse = np.sqrt(np.mean(loo_resid**2, axis=0))  # (m,)
    loo_rel = float(np.max(loo_rmse / y_scale))

    # condition number of the design (svd-based, no ridge)
    sing_vals = npl.svd(design, compute_uv=False)
    cond_vdm = float((sing_vals[0] / sing_vals[-1]) if sing_vals[-1] > 0 else np.inf)

    # derivative stability vs one-degree-lower refit
    deriv_rel = 0.0
    if deg >= max(order, 1) + 1:
        design_m = design[:, :deg]  # degree-1 design
        gram_m = design_m.T @ design_m + ridge * np.eye(deg)
        gram_m_inv = npl.pinv(gram_m) if ridge == 0.0 else npl.inv(gram_m)
        coeffs_m = (gram_m_inv @ design_m.T @ y)  # (deg, m)

        deriv_full = extract_derivative(coeffs, order, factor)  # shape (m,)
        deriv_minus = extract_derivative(coeffs_m, order, factor)  # shape (m,)
        num = np.max(np.abs(deriv_full - deriv_minus))
        den = np.max(np.abs(deriv_full)) + 1e-12
        deriv_rel = float(num / den)

    # heuristic thresholds
    th = {"rrms_rel": 5e-4, "loo_rel": 1e-3, "cond_vdm": 1e8, "deriv_rel": 5e-3}

    # suggestions
    suggestions: list[str] = []
    if rrms_rel > th["rrms_rel"] or loo_rel > th["loo_rel"]:
        suggestions.append("Increase sampling half-width via `spacing` to spread nodes.")
        suggestions.append("Add a few points (n_points) up to the Chebyshev cap, or pass an explicit grid.")
    if cond_vdm > th["cond_vdm"]:
        suggestions.append("Increase `ridge` (e.g., ×10) to stabilize the fit.")
        suggestions.append("Widen `spacing` to reduce node crowding near zero.")
    if deriv_rel > th["deriv_rel"]:
        suggestions.append("Use a slightly higher degree (add 2–4 points) or widen `spacing`.")
    if not suggestions:
        suggestions.append("Fit looks numerically healthy.")

    metrics: dict[str, float | dict[str, float]] = {
        "rrms_rel": rrms_rel,
        "loo_rel": loo_rel,
        "cond_vdm": cond_vdm,
        "deriv_rel": deriv_rel,
        "thresholds": th,
    }
    return metrics, suggestions


def fit_with_headroom_and_maybe_minimize(
    u: np.ndarray, y: np.ndarray, *, order: int, mode: str, ridge: float, factor: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Perform a polynomial fit with headroom and optionally prefer minimal degree.

    Fits a polynomial of degree ``deg_hi = deg_req + headroom`` to the data and, if the
    lower-degree fit (``deg_req``) yields effectively identical derivatives, switches
    to the minimal degree for stability and exactness. The method ensures that exact
    polynomials or smooth functions yield consistent derivatives without overfitting.

    Args:
      u: Scaled independent variable values (offsets), shape ``(n_points,)``.
      y: Function values evaluated at the grid points, shape ``(n_points, n_components)``.
      order: Derivative order to compute (``>= 1``).
      mode: Sampling mode — one of ``"x"``, ``"signed_log"``, or ``"sqrt"``.
        Determines whether additional pullback corrections are applied when comparing fits.
      ridge: Ridge regularization parameter applied in the least-squares fit.
      factor: Scaling factor relating physical offsets to scaled ones (``t = u * factor``).

    Returns:
      Tuple[np.ndarray, np.ndarray, int]:
        A 3-tuple ``(coeffs, rrms, deg_used)`` where:
          - ``coeffs``: Power-basis polynomial coefficients, shape ``(deg+1, n_components)``.
          - ``rrms``: Relative RMS residuals per component, shape ``(n_components,)``.
          - ``deg_used``: Polynomial degree actually adopted (either ``deg_req`` or ``deg_hi``).

    Raises:
      ValueError: If the fit fails due to invalid input dimensions or degree constraints.

    Notes:
      - ``deg_req`` equals ``2 * order`` for ``"sqrt"`` mode, else ``order``.
      - Headroom is set to +4 for second-order sqrt mode, otherwise +2.
      - The switch to minimal degree occurs only if both:
          (a) the lower-degree fit has negligible residuals (``rrms < 5e-15``), and
          (b) its derivatives match the higher-degree fit within absolute tolerance 1e-9.
    """
    n_eff = u.size
    deg_req = (2 * order) if (mode == "sqrt") else order
    extra_need = 4 if (mode == "sqrt" and order == 2) else 2
    deg_hi = min(deg_req + extra_need, (n_eff - 1) // 2)

    c_hi, rrms_hi = fit_multi_power(u, y, deg_hi, ridge=ridge)
    deg_used = deg_hi

    if deg_hi > deg_req and order >= 3:
        c_min, rrms_min = fit_multi_power(u, y, deg_req, ridge=ridge)
        if np.all(rrms_min < 5e-15):
            return c_min, rrms_min, deg_req

        pull_hi = pullback_derivative_from_fit(
            mode=mode, order=order, coeffs=c_hi, factor=factor,
            x0=0.0, sign_used=(+1.0 if mode == "sqrt" else None)
        )
        pull_min = pullback_derivative_from_fit(
            mode=mode, order=order, coeffs=c_min, factor=factor,
            x0=0.0, sign_used=(+1.0 if mode == "sqrt" else None)
        )

        if np.allclose(pull_hi, pull_min, rtol=0.0, atol=1e-9):
            return c_min, rrms_min, deg_req

    return c_hi, rrms_hi, deg_used


def pullback_derivative_from_fit(
    *, mode: str, order: int, coeffs: np.ndarray, factor: float, x0: float, sign_used: float | None
) -> np.ndarray:
    """Extract the derivative at ``x0`` with mode-specific pullbacks.

    Interprets the power-basis polynomial coefficients in the internal coordinate
    and converts the requested derivative to the physical ``x`` domain. In
    ``"x"`` mode, the derivative is read directly from the power basis. In
    transformed modes, an analytic pullback is applied: the signed-log chain
    rule or the boundary-centered square-root mapping. Note that in ``"sqrt"`` mode,
    the first derivative in ``x`` uses the internal 2nd coefficient,
    and the second derivative uses the internal 4th coefficient.

    Args:
      mode: Sampling/transform mode (``"x"``, ``"signed_log"``, or ``"sqrt"``).
      order: Derivative order to return (``>= 1``). For transformed modes, only
        orders 1 and 2 are supported.
      coeffs: Power-basis coefficients with columns per component, shape
        ``(deg+1, n_components)``.
      factor: Positive scaling factor such that physical offsets satisfy
        ``t = u * factor``.
      x0: Physical expansion point where the derivative is evaluated.
      sign_used: For ``"sqrt"`` mode, the branch sign (``+1`` or ``-1``). Ignored
        for other modes.

    Returns:
      np.ndarray: The requested derivative at ``x0`` with shape ``(n_components,)``.

    Raises:
      NotImplementedError: If ``mode`` is ``"signed_log"`` or ``"sqrt"`` and
        ``order`` is not 1 or 2.
    """
    if mode == "signed_log":
        d1 = extract_derivative(coeffs, 1, factor)
        if order == 1:
            return signed_log_derivatives_to_x(1, x0, d1)
        d2 = extract_derivative(coeffs, 2, factor)
        return signed_log_derivatives_to_x(2, x0, d1, d2)

    if mode == "sqrt":
        s = +1.0 if (sign_used is None) else float(sign_used)
        if order == 1:
            g2 = extract_derivative(coeffs, 2, factor)
            return pullback_sqrt_at_zero(1, s, g2=g2)
        g4 = extract_derivative(coeffs, 4, factor)
        return pullback_sqrt_at_zero(2, s, g4=g4)

    return extract_derivative(coeffs, order, factor)
