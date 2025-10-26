"""Diagnostics for derivative approximations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from derivkit.adaptive.polyfit_utils import assess_polyfit_quality

__all__ = [
    "format_derivative_diagnostics",
    "print_derivative_diagnostics",
    "make_derivative_diag",
    "fit_is_obviously_bad"
]


def format_derivative_diagnostics(
    diag: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    decimals: int = 4,
    max_rows: int = 12,
) -> str:
    """Format derivative diagnostics into a human-readable string.

    Args:
      diag: Diagnostics dictionary as returned by ``make_derivative_diag``.
      meta: Optional metadata dictionary to include in the output.
      decimals: Number of decimal places for floating-point numbers.
      max_rows: Maximum number of rows to display for arrays; larger arrays are truncated.

    Returns:
      A formatted string summarizing the diagnostics.
    """
    if not isinstance(diag, dict):
        return "‹diagnostics unavailable›"

    x = np.asarray(diag.get("x", []), float)
    t = np.asarray(diag.get("t", []), float)
    y = np.asarray(diag.get("y", []), float)
    degree = diag.get("degree", None)

    step_min = step_max = None
    if t.size >= 2:
        dt = np.diff(np.sort(t))
        step_min = float(np.min(dt))
        step_max = float(np.max(dt))
    uniformish = (
        step_min is not None
        and step_max is not None
        and abs(step_max - step_min) <= 1e-12 * max(1.0, step_max)
    )

    with np.printoptions(precision=decimals, suppress=True):
        lines = ["=== Derivative Diagnostics ==="]
        if meta:
            lines.append("Meta:")
            wanted = [
                "x0",
                "order",
                "n_points",
                "spacing",
                "base_abs",
                "spacing_resolved",
                "n_workers",
                "domain",
                "mode",
                "ridge",
            ]
            for k in wanted:
                if k in meta:
                    lines.append(f"  {k}={meta[k]}")
            for k, v in meta.items():
                if k not in wanted:
                    lines.append(f"  {k}={v}")
            lines.append("")

        lines += [
            "Grid:",
            f"  t offsets (preview): {_preview_1d(t, max_rows)}",
            f"  u offsets (preview): {_preview_1d(np.asarray(diag.get('u', []), float), max_rows)}",
            f"  x points  (preview): {_preview_1d(x, max_rows)}",
        ]
        if step_min is not None:
            lines.append(
                f"step_min={step_min:.{decimals}g}, "
                f"step_max={step_max:.{decimals}g}, "
                f"uniformish={uniformish}"
            )
        lines.append("")

        lines.append("Samples y (rows correspond to x/t):")
        lines.append(f"{_preview_2d_rows(y, max_rows)}")
        lines.append("")

        lines.append("Fit:")
        lines.append(f"  chosen degree(s): {degree}")
        rrms = diag.get("rrms", None)
        if rrms is not None:
            lines.append(f"  rrms: {rrms}")

        fq = diag.get("fit_quality", None)
        fs = diag.get("fit_suggestions", None)
        if isinstance(fq, dict):
            lines.append("")
            lines.append("Fit quality:")
            lines.append(
                "  rrms_rel={:.2e}, loo_rel={:.2e}, cond_vdm={:.2e}, deriv_rel={:.2e}".format(
                    fq.get("rrms_rel", float("nan")),
                    fq.get("loo_rel", float("nan")),
                    fq.get("cond_vdm", float("nan")),
                    fq.get("deriv_rel", float("nan")),
                )
            )
            th = fq.get("thresholds", {})
            if not isinstance(th, dict):
                th = {}
            lines.append(
                "  thresholds: rrms_rel={:.1e}, loo_rel={:.1e}, cond_vdm={:.1e}, deriv_rel={:.1e}".format(
                    th.get("rrms_rel", float("nan")),
                    th.get("loo_rel", float("nan")),
                    th.get("cond_vdm", float("nan")),
                    th.get("deriv_rel", float("nan")),
                )
            )
            if isinstance(fs, (list, tuple)) and len(fs) > 0:
                lines.append("  suggestions:")
                for s in fs:
                    lines.append(f"    - {s}")

        return "\n".join(lines)


def print_derivative_diagnostics(
    diag: Dict[str, Any], *, meta: Optional[Dict[str, Any]] = None
) -> None:
    """Print derivative diagnostics to standard output.

    Args:
      diag: Diagnostics dictionary as returned by ``make_derivative_diag``.
      meta: Optional metadata dictionary to include in the output.

    Returns:
      None
    """
    print(format_derivative_diagnostics(diag, meta=meta))


def _preview_1d(a: np.ndarray, max_rows: int) -> np.ndarray:
    """Return a preview of a 1D array, truncating with NaN if too long.

    Args:
      a: Input 1D array.
      max_rows: Maximum number of rows to display.

    Returns:
      A 1D array with at most ``max_rows`` elements, with NaN in the middle if truncated.
    """
    a = np.asarray(a)
    if a.ndim != 1 or a.size <= max_rows:
        return a
    k = max_rows // 2
    return np.concatenate([a[:k], np.array([np.nan]), a[-k:]])


def _preview_2d_rows(a: np.ndarray, max_rows: int) -> np.ndarray:
    """Return a preview of a 2D array by rows.

    The preview is truncated with a NaN row if there are too many elements.

    Args:
      a: Input 2D array.
      max_rows: Maximum number of rows to display.

    Returns:
      A 2D array with at most ``max_rows`` rows, with a NaN row in the middle if truncated.
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] <= max_rows:
        return a
    k = max_rows // 2
    return np.vstack([a[:k], np.full((1, a.shape[1]), np.nan), a[-k:]])


def make_derivative_diag(
    *,
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    degree: int | list[int],
    spacing_resolved: float | None = None,
    rrms: Optional[NDArray[np.floating]] = None,
    coeffs: Optional[NDArray[np.floating]] = None,
    ridge: float | None = None,
    order: int | None = None,
) -> dict:
    """Builds a lightweight diagnostics for a local polynomial derivative fit.

    This assembles the core quantities used in plotting/printing diagnostics and,
    when enough inputs are provided, augments them with polynomial-fit quality
    metrics and human-readable suggestions.

    Args:
        x: Absolute sample locations, shape ``(n_points,)``.
        t: Offsets relative to ``x0`` (``t = x - x0``), shape ``(n_points,)``.
        u: Scaled offsets used in the polynomial basis (typically ``u = t / s``),
           shape ``(n_points,)``.
        y: Function evaluations at ``x``, shape ``(n_points, n_obs)``.
        degree: Final polynomial degree used. May be an ``int`` or a per-observable
            list of ints (length ``n_obs``).
        spacing_resolved: Resolved spacing descriptor for the default grid (numeric
            half-width or ``None`` if not applicable).
        rrms: Relative RMS residuals of the fit, shape ``(n_obs,)`` (optional).
        coeffs: Polynomial coefficients in the scaled basis, shape ``(deg+1, n_obs)`` (optional).
        ridge: Ridge regularization strength used in the fit (optional).
        order: Derivative order of interest (optional).

    Returns:
        dict: A plain dictionary with fields suited for logging/printing/plotting.

        Always present:
            - ``"x"`` : ``np.ndarray`` with shape ``(n_points,)``
            - ``"t"`` : ``np.ndarray`` with shape ``(n_points,)``
            - ``"u"`` : ``np.ndarray`` with shape ``(n_points,)``
            - ``"y"`` : ``np.ndarray`` with shape ``(n_points, n_obs)``
            - ``"degree"`` : ``int`` or ``list[int]``

        Included when available:
            - ``"spacing_resolved"`` : ``float | None``
            - ``"rrms"`` : ``np.ndarray`` or ``float``

        Included when quality inputs are provided (``coeffs``, ``ridge``, ``order``):
            - ``"fit_quality"`` : ``dict`` with keys like ``"rrms_rel"``, ``"loo_rel"``,
              ``"cond_vdm"``, ``"deriv_rel"``, and a nested ``"thresholds"`` dict.
            - ``"fit_suggestions"`` : ``list[str]`` with human-readable hints.
    """
    out: Dict[str, Any] = {
        "x": x,
        "t": t,
        "u": u,
        "y": y,
        "degree": degree,
    }
    if spacing_resolved is not None:
        out["spacing_resolved"] = float(spacing_resolved)
    if rrms is not None:
        out["rrms"] = rrms if not (rrms.ndim == 1 and rrms.size == 1) else float(rrms[0])

    have_quality_args = (
        coeffs is not None
        and ridge is not None
        and order is not None
    )
    if have_quality_args:
        metrics, suggestions = assess_polyfit_quality(
            u=u,
            y=y,
            coeffs=coeffs,
            deg=(degree if isinstance(degree, int) else int(degree[0])),
            ridge=float(ridge),
            order=int(order),
        )
        out["fit_quality"] = metrics
        out["fit_suggestions"] = suggestions

    return out


def fit_is_obviously_bad(metrics: dict) -> tuple[bool, str]:
    """Heuristically flag a clearly unstable polynomial fit and return a brief reason.

    This inspects scalar diagnostics (from ``assess_polyfit_quality``) against amplified
    thresholds. If any metric is far beyond its nominal limit, the fit is flagged.

    Args:
        metrics: Dictionary with keys like:
            - ``"rrms_rel"``, ``"loo_rel"``, ``"cond_vdm"``, ``"deriv_rel"``
            - ``"thresholds"`` : ``dict`` of nominal per-metric thresholds

    Returns:
        tuple[bool, str]: ``(is_bad, message)`` where:
            - ``is_bad`` is ``True`` if any metric exceeds its amplified threshold
              (×5 for ``rrms_rel``, ``loo_rel``, ``deriv_rel``; ×10 for ``cond_vdm``),
              otherwise ``False``.
            - ``message`` is a short human-readable summary when ``is_bad`` is ``True``,
              otherwise ``""``.

    Notes:
        This is a soft, non-fatal screen for diagnostics/logging. Callers decide how to
        react (warn, rebuild grid, widen spacing, add samples, increase ridge, etc.).
    """
    th = metrics["thresholds"]
    is_bad = (
        metrics["rrms_rel"] > 5 * th["rrms_rel"]
        or metrics["loo_rel"] > 5 * th["loo_rel"]
        or metrics["cond_vdm"] > 10 * th["cond_vdm"]
        or metrics["deriv_rel"] > 5 * th["deriv_rel"]
    )
    msg = ""
    if is_bad:
        msg = (
            "Polynomial fit looks unstable: "
            f"rrms_rel={metrics['rrms_rel']:.2e}, "
            f"loo_rel={metrics['loo_rel']:.2e}, "
            f"cond_vdm={metrics['cond_vdm']:.2e}, "
            f"deriv_rel={metrics['deriv_rel']:.2e}."
        )
    return is_bad, msg
