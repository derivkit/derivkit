"""Single-fit polynomial estimator with acceptance gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ComponentOutcome:
    """Per-component result of the adaptive fit.

    Holds the estimated derivative for one output component along with optional
    diagnostics captured during the fit (e.g., points used, residual metrics,
    conditioning indicators).
    """

    value: float
    x_used: Optional[np.ndarray]
    y_used: Optional[np.ndarray]
    y_fit: Optional[np.ndarray]
    residuals: Optional[np.ndarray]
    status: dict


def estimate_component(
    *,
    x0: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
    order: int,
    tau_res: float = 5e-2,
    kappa_max: float = 1e8,
) -> ComponentOutcome:
    """Estimate the derivative of a single component via a single weighted polynomial fit.

    The fit is performed in normalized coordinates around x0, and the derivative

    is extracted from the fitted polynomial. Two acceptance gates are applied:
    a residual-to-signal gate and a conditioning gate on the Vandermonde matrix.

    Args:
        x0: The point at which the derivative is evaluated.
        x_values: The x-coordinates of the samples.
        y_values: The y-coordinates of the samples (same length as x_values).
        order: The order of the derivative to estimate (and polynomial degree).
        tau_res: The maximum acceptable residual-to-signal ratio (default 5e-2
            for a 5% threshold).
        kappa_max: The maximum acceptable condition number of the Vandermonde
            matrix (default 1e8).

    Returns:
        A ComponentOutcome containing the derivative estimate, fit diagnostics,
        and acceptance status.

    Raises:
        ValueError: If inputs are invalid.
    """
    from derivkit.adaptive.fit_core import (
        fit_once,
        normalize_coords,
        residual_to_signal,
    )

    x = np.asarray(x_values, float)
    y = np.asarray(y_values, float)

    # Perform the single weighted polynomial fit in normalized coords
    fit = fit_once(x0, x, y, order, weight_eps_frac=1e-3)
    if not fit.get("ok", False):
        import warnings
        warnings.warn(
            "[AdaptiveFitDerivative] Fit failed; returned NaN. "
            "Consider increasing `min_samples` or loosening `acceptance`.",
            RuntimeWarning,
        )
        return ComponentOutcome(
            value=float("nan"),
            x_used=None,
            y_used=None,
            y_fit=None,
            residuals=None,
            status={"accepted": False, "reason": "fit_failed", "n_points": int(x.size),
                    "tau_res": float(tau_res), "kappa_max": float(kappa_max)},
        )

    # residual-to-signal gate
    rho = fit.get("rho")
    if rho is None:
        _, _h = normalize_coords(x, x0)
        rho, _, signal = residual_to_signal(fit["y_fit"], y)
        if not np.isfinite(rho) or signal <= 1e-12:
            rho = 0.0

    # conditioning gate on normalized Vandermonde
    u, _ = normalize_coords(x, x0)
    V = np.vander(u, N=order + 1, increasing=False)
    if V.size == 0 or min(V.shape) == 0:
        kappa = np.inf
    else:
        s = np.linalg.svd(V, compute_uv=False, hermitian=False)
        kappa = float(s[0] / max(s[-1], 1e-14))

    # derivative at x0 from the fitted poly
    val = float(fit.get("deriv_at_x0", fit["poly_u"].deriv(m=order)(0.0) / (fit["h"] ** order)))

    accepted = (rho <= tau_res) and (kappa <= kappa_max)
    reason = "accepted" if accepted else ("rho_gate" if rho > tau_res else "conditioning_gate")
    if not accepted:
        import warnings
        warnings.warn(
            "[AdaptiveFitDerivative] Acceptance gates not satisfied; returned polynomial "
            f"estimate (reason={reason}, rho≈{rho:.3g}, cond≈{kappa:.3g}). "
            "Consider increasing `min_samples` or loosening `acceptance`.",
            RuntimeWarning,
        )

    return ComponentOutcome(
        value=val,
        x_used=None,
        y_used=None,
        y_fit=fit.get("y_fit"),
        residuals=fit.get("residuals"),
        status={
            "accepted": bool(accepted),
            "reason": reason,
            "rho": float(rho),
            "kappa": float(kappa),
            "tau_res": float(tau_res),
            "kappa_max": float(kappa_max),
            "n_points": int(x.size),
        },
    )
