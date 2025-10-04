"""Diagnostics utilities for the adaptive estimator."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def hints_from_reasons(*, reasons: List[str], include_zero: bool) -> List[str]:
    """Convert estimator reasons into short, user-facing suggestions.

    Args:
        reasons: List of reason strings from the estimator outcomes.
        include_zero: Whether zero was included in the grid.

    Returns:
        List of human hints.
    """
    rset = set(reasons)
    hints = []
    if "rho_gate" in rset:
        hints.append("Increase `min_samples` to enlarge the local fit window.")
        hints.append("Or loosen `acceptance` (e.g., 'loose' or 'very_loose').")
    if "conditioning_gate" in rset:
        hints.append("Increase `min_samples` to improve numerical conditioning.")
        hints.append("Or loosen `acceptance` slightly.")
        if not include_zero:
            hints.append("Set `include_zero=True` for a symmetric, anchored grid.")
    if "fit_failed" in rset:
        hints.append("Increase `min_samples` and ensure the function returns finite values on the grid.")
    return hints


def make_diagnostics(
    *,
    outcomes,
    x_all: np.ndarray,
    y_all: np.ndarray,
    order: int,
    min_samples: int,
    include_zero: bool,
    tau_res: float,
    kappa_max: float,
) -> Dict[str, Any]:
    """Daignostics dictionary for an adaptive fit operation.

    Args:
        outcomes: List of ComponentOutcome objects from the fit.
        x_all: Full grid of x offsets used (1D array).
        y_all: Full grid of function values (2D array, shape [n_points,
            n_components]).
        order: Derivative order requested.
        min_samples: Minimum samples requested.
        include_zero: Whether zero was included in the grid.
        tau_res: Residual-to-signal threshold used.
        kappa_max: Conditioning cap used.

    Returns:
        Dictionary with diagnostics and human hints.
    """
    reasons = [o.status.get("reason") for o in outcomes]
    any_not_accepted = any(not o.status.get("accepted", True) for o in outcomes)

    n_components = int(y_all.shape[1])
    x_used = [x_all.copy() for _ in range(n_components)]
    y_used = [y_all[:, i].copy() for i in range(n_components)]
    used_mask = [np.ones_like(x_all, dtype=bool) for _ in range(n_components)]

    return {
        "any_not_accepted": bool(any_not_accepted),
        "component_reasons": reasons,
        "tau_res": float(tau_res),
        "kappa_max": float(kappa_max),
        "order": int(order),
        "min_samples": int(min_samples),
        "include_zero": bool(include_zero),
        # full grid & per-component mirrors (compat fields)
        "x_all": x_all.copy(),
        "y_all": y_all.copy(),
        "x_used": x_used,
        "y_used": y_used,
        "used_mask": used_mask,
        # per-component raw status for deeper inspection if needed
        "component_status": [o.status for o in outcomes],
        # human hints
        "hints": hints_from_reasons(reasons=reasons, include_zero=include_zero),
    }
