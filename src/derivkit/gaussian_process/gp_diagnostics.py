"""Diagnostics for Gaussian-Process-based derivative estimates."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "make_gp_diag",
    "format_gp_diagnostics",
    "print_gp_diagnostics",
    "gp_fit_is_obviously_bad",
]


def make_gp_diag(
    *,
    x0: float,
    order: int,
    axis: int,
    x_full: NDArray[np.floating],
    y_full: NDArray[np.floating],
    x_win: NDArray[np.floating],
    y_win: NDArray[np.floating],
    mu_x: float,
    sigma_x: float,
    mu_y: float,
    sigma_y: float,
    kernel_name: str,
    kernel_params_before: Dict[str, Any],
    kernel_params_after_opt: Dict[str, Any] | None,
    kernel_params_after_clamp: Dict[str, Any],
    noise_variance_before: float,
    noise_variance_after_opt: float | None,
    jitter_used: float,
    variance_floor: float,
    optimize: bool,
    frac_window: float,
    spacing: Any,
    n_points: int,
    base_abs: float | None,
    state: Dict[str, Any],
    fd_estimate: float | None,
    dmean_std: float,
    dvar_std: float,
    dmean: float,
    dvar: float,
    suspicious: bool,
    kernel_cond: float | None,
    kernel_min_eig: float | None,
    kernel_max_eig: float | None,
) -> Dict[str, Any]:
    """Assemble a GP diagnostics dictionary suitable for logging/printing.

    Args:
        x0: Expansion point.
        order: Derivative order.
        axis: Axis along which derivative is computed.
        x_full: Full input grid used for GP fitting.
        y_full: Function values at full input grid.
        x_win: Local input window used for GP fitting.
        y_win: Function values at local input window.
        mu_x: Mean of x values used for standardization.
        sigma_x: Stddev of x values used for standardization.
        mu_y: Mean of y values used for standardization.
        sigma_y: Stddev of y values used for standardization.
        kernel_name: Name of the kernel used.
        kernel_params_before: Kernel hyperparameters before optimization.
        kernel_params_after_opt: Kernel hyperparameters after optimization.
        kernel_params_after_clamp: Kernel hyperparameters after clamping.
        noise_variance_before: Noise variance before optimization.
        noise_variance_after_opt: Noise variance after optimization.
        jitter_used: Jitter added to the kernel diagonal.
        variance_floor: Variance floor used in GP.
        optimize: Whether hyperparameter optimization was performed.
        frac_window: Fractional window size used for local fitting.
        spacing: Spacing strategy or value used for grid construction.
        n_points: Number of points in the grid.
        base_abs: Baseline absolute half-width for "auto" spacing.
        state: Internal GP state dictionary.
        fd_estimate: Finite-difference reference estimate for the derivative.
        dmean_std: Mean of the standardized derivative estimate.
        dvar_std: Variance of the standardized derivative estimate.
        dmean: Mean of the derivative estimate.
        dvar: Variance of the derivative estimate.
        suspicious: Whether the derivative estimate is considered suspicious.
        kernel_cond: Condition number of the kernel matrix.
        kernel_min_eig: Minimum eigenvalue of the kernel matrix.
        kernel_max_eig: Maximum eigenvalue of the kernel matrix.

    Returns:
        A dictionary containing GP diagnostics.
    """
    return {
        "x0": float(x0),
        "order": int(order),
        "axis": int(axis),
        "n_points": int(n_points),
        "spacing": spacing,
        "base_abs": None if base_abs is None else float(base_abs),
        "frac_window": float(frac_window),
        "optimize": bool(optimize),

        "x_full_preview": np.asarray(x_full).reshape(-1),
        "y_full_preview": np.asarray(y_full).reshape(-1, 1) if np.ndim(y_full) == 1 else np.asarray(y_full),

        "x_window_preview": np.asarray(x_win).reshape(-1),
        "y_window_preview": np.asarray(y_win).reshape(-1, 1) if np.ndim(y_win) == 1 else np.asarray(y_win),

        "standardization": {
            "mu_x": float(mu_x),
            "sigma_x": float(sigma_x),
            "mu_y": float(mu_y),
            "sigma_y": float(sigma_y),
        },

        "kernel": {
            "name": kernel_name,
            "params_before": dict(kernel_params_before),
            "params_after_opt": None if kernel_params_after_opt is None else dict(kernel_params_after_opt),
            "params_after_clamp": dict(kernel_params_after_clamp),
            "noise_before": float(noise_variance_before),
            "noise_after_opt": None if noise_variance_after_opt is None else float(noise_variance_after_opt),
            "jitter_used": float(jitter_used),
            "variance_floor": float(variance_floor),
            "K_cond": None if kernel_cond is None else float(kernel_cond),
            "K_min_eig": None if kernel_min_eig is None else float(kernel_min_eig),
            "K_max_eig": None if kernel_max_eig is None else float(kernel_max_eig),
        },

        "derivative": {
            "fd_reference": None if fd_estimate is None else float(fd_estimate),
            "mean_standardized": float(dmean_std),
            "var_standardized": float(dvar_std),
            "mean": float(dmean),
            "var": float(dvar),
            "sigma": float(np.sqrt(max(dvar, 0.0))),
            "suspicious": bool(suspicious),
        },

        # pass through a compact public subset of state (safe to log)
        "state_summary": {
            "normalize": bool(state.get("normalize", False)),
            "jitter": float(state.get("jitter", 0.0)),
            "variance_floor": float(state.get("variance_floor", 0.0)),
            "n_train": int(np.asarray(state.get("training_inputs", [])).shape[0]
                           if "training_inputs" in state else 0),
        },
    }


def format_gp_diagnostics(
    diag: Dict[str, Any],
    *,
    decimals: int = 4,
    max_rows: int = 12,
) -> str:
    """Formats GP diagnostics dictionary into a human-readable string.

    Args:
        diag: GP diagnostics dictionary as produced by ``make_gp_diag``.
        decimals: Number of decimal places for floating-point output.
        max_rows: Maximum number of rows to display for array previews.

    Returns:
        A formatted string summarizing the GP diagnostics.
    """
    if not isinstance(diag, dict):
        return "‹gp diagnostics unavailable›"

    with np.printoptions(precision=decimals, suppress=True):
        lines = ["=== GP Derivative Diagnostics ==="]

        lines.append("Meta:")
        for k in ("x0", "order", "axis", "n_points", "spacing", "base_abs", "frac_window", "optimize"):
            if k in diag:
                lines.append(f"  {k}={diag[k]}")
        lines.append("")

        # Grid previews
        xf = np.asarray(diag.get("x_full_preview", []), float)
        xw = np.asarray(diag.get("x_window_preview", []), float)
        lines += [
            "Design:",
            f"  x_full (preview): {_preview_1d(xf, max_rows)}",
            f"  x_win  (preview): {_preview_1d(xw, max_rows)}",
            "",
        ]

        # Standardization
        std = diag.get("standardization", {})
        if isinstance(std, dict) and std:
            lines.append("Standardization:")
            lines.append(
                "  mu_x={mu_x:.4g}, sigma_x={sigma_x:.4g}, mu_y={mu_y:.4g}, sigma_y={sigma_y:.4g}".format(**std)
            )
            lines.append("")

        # Kernel
        ker = diag.get("kernel", {})
        if isinstance(ker, dict) and ker:
            lines.append("Kernel:")
            lines.append(f"  name={ker.get('name')}")
            lines.append(f"  params_before={ker.get('params_before')}")
            if ker.get("params_after_opt") is not None:
                lines.append(f"  params_after_opt={ker.get('params_after_opt')}")
            lines.append(f"  params_after_clamp={ker.get('params_after_clamp')}")
            lines.append(
                "  noise_before={:.3e}, noise_after_opt={}".format(
                    ker.get("noise_before", float("nan")),
                    "None" if ker.get("noise_after_opt") is None else f"{ker.get('noise_after_opt'):.3e}",
                )
            )
            lines.append(
                "  jitter_used={:.3e}, variance_floor={:.3e}".format(
                    ker.get("jitter_used", float("nan")), ker.get("variance_floor", float("nan"))
                )
            )
            if ker.get("K_cond") is not None:
                lines.append(
                    "  K: cond≈{:.3e}, min_eig≈{:.3e}, max_eig≈{:.3e}".format(
                        ker.get("K_cond"), ker.get("K_min_eig"), ker.get("K_max_eig")
                    )
                )
            lines.append("")

        # Derivative
        der = diag.get("derivative", {})
        if isinstance(der, dict) and der:
            lines.append("Derivative at x0:")
            lines.append(
                "  mean={:.6e}, sigma={:.3e}, suspicious={}".format(
                    der.get("mean", float("nan")), der.get("sigma", float("nan")), der.get("suspicious", False)
                )
            )
            if der.get("fd_reference") is not None:
                lines.append(f"  fd_reference={der['fd_reference']:+.6e}")
            lines.append("")

        # Suggestion rules (purely textual)
        suggestions = _suggestions_from_diag(diag)
        if suggestions:
            lines.append("Suggestions:")
            for s in suggestions:
                lines.append(f"  - {s}")

        return "\n".join(lines)


def print_gp_diagnostics(diag: Dict[str, Any]) -> None:
    """Prints GP diagnostics in a human-readable format.

    Args:
        diag: GP diagnostics dictionary as produced by ``make_gp_diag``.

    Returns:
        None
    """
    print(format_gp_diagnostics(diag))


def gp_fit_is_obviously_bad(diag: Dict[str, Any]) -> tuple[bool, str]:
    """Heuristically flags obviously bad GP fits from diagnostics.

    Args:
        diag: GP diagnostics dictionary as produced by ``make_gp_diag``.

    Returns:
        A tuple (is_bad, message) where is_bad is True if the fit is obviously bad,
        and message is a human-readable explanation.
    """
    ker = diag.get("kernel", {}) if isinstance(diag, dict) else {}
    der = diag.get("derivative", {}) if isinstance(diag, dict) else {}

    cond = ker.get("K_cond", None)
    min_eig = ker.get("K_min_eig", None)
    suspicious = bool(der.get("suspicious", False))

    too_ill = cond is not None and cond > 1e12
    negish = min_eig is not None and min_eig < -1e-10

    is_bad = suspicious or too_ill or negish
    msg_bits = []
    if suspicious:
        msg_bits.append("GP vs finite-difference mismatch")
    if too_ill:
        msg_bits.append(f"K condition≈{cond:.2e} (ill-conditioned)")
    if negish:
        msg_bits.append(f"K min_eig≈{min_eig:.2e} (not PSD)")
    return is_bad, "; ".join(msg_bits)


def _suggestions_from_diag(diag: Dict[str, Any]) -> list[str]:
    """Generate human-readable tuning suggestions from diagnostics.

    Args:
        diag: GP diagnostics dictionary as produced by ``make_gp_diag``.

    Returns:
        A list of suggestion strings.
    """
    out: list[str] = []

    ker = diag.get("kernel", {}) if isinstance(diag, dict) else {}
    der = diag.get("derivative", {}) if isinstance(diag, dict) else {}

    cond = ker.get("K_cond", None)
    suspicious = bool(der.get("suspicious", False))
    frac = float(diag.get("frac_window", 0.35))
    n_points = int(diag.get("n_points", 13))
    spacing = diag.get("spacing", "auto")
    optimize = bool(diag.get("optimize", False))

    # Rules
    if suspicious:
        out.append("Increase local window (e.g., frac_window → max(0.5, 1.5×current)).")
        out.append("Enable or keep hyperparameter optimization.")
        out.append("Prefer Chebyshev spacing for second derivatives.")
    if cond is not None and cond > 1e12:
        out.append("Reduce effective model complexity: increase noise_variance or jitter slightly.")
        out.append("Consider a larger length_scale or clamp length_scale bounds to the window span.")
    if isinstance(spacing, str) and spacing != "chebyshev":
        out.append("Use spacing='chebyshev' for curvature (order=2).")
    if n_points < 61:
        out.append("Increase n_points (e.g., ≥61) to stabilize curvature estimates.")
    if frac < 0.35:
        out.append("Use at least frac_window≈0.35 to avoid ultra-tight windows.")
    if not optimize:
        out.append("Set optimize=True for robust local hyperparameter selection.")

    # de-duplicate while preserving order
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def _preview_1d(a: np.ndarray, max_rows: int) -> np.ndarray:
    """Returns a preview of a 1D array with at most max_rows entries.

    Args:
        a: Input array.
        max_rows: Maximum number of rows to display.

    Returns:
        A 1D array with at most max_rows entries, with NaN in the middle if truncated.
    """
    a = np.asarray(a)
    if a.ndim != 1 or a.size <= max_rows:
        return a
    k = max_rows // 2
    return np.concatenate([a[:k], np.array([np.nan]), a[-k:]])


def _preview_2d_rows(a: np.ndarray, max_rows: int) -> np.ndarray:
    """Returns a preview of a 2D array with at most max_rows rows.

    Args:
        a: Input array.
        max_rows: Maximum number of rows to display.

    Returns:
        A 2D array with at most max_rows rows, with a row of NaNs in the middle if truncated.
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] <= max_rows:
        return a
    k = max_rows // 2
    return np.vstack([a[:k], np.full((1, a.shape[1]), np.nan), a[-k:]])
