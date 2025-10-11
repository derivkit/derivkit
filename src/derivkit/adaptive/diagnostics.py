"""Diagnostics for derivative approximations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "format_derivative_diagnostics",
    "print_derivative_diagnostics",
    "make_derivative_diag",
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
        diag: Diagnostics dictionary as returned by `make_derivative_diag`.
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
        step_min = float(dt.min())
        step_max = float(dt.max())
    uniformish = (
        step_min is not None
        and step_max is not None
        and abs(step_max - step_min) <= 1e-12 * max(1.0, step_max)
    )

    old = np.get_printoptions()
    np.set_printoptions(precision=decimals, suppress=True)
    try:
        lines = ["=== Derivative Diagnostics ==="]
        if meta:
            lines.append("Meta:")
            wanted = [
                "x0",
                "order",
                "n_points",
                "direction",
                "spacing",
                "base_abs",
                "spacing_resolved",
                "n_workers",
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
            f"  x points  (preview): {_preview_1d(x, max_rows)}",
        ]
        if step_min is not None:
            lines.append(
                f"  step_min={step_min:.{decimals}g}, step_max={step_max:.{decimals}g}, uniformish={uniformish}"
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

        return "\n".join(lines)
    finally:
        np.set_printoptions(**old)


def print_derivative_diagnostics(
    diag: Dict[str, Any], *, meta: Optional[Dict[str, Any]] = None
) -> None:
    """Print derivative diagnostics to standard output.

    Args:
        diag: Diagnostics dictionary as returned by `make_derivative_diag`.
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
        A 1D array with at most max_rows elements, with NaN in the middle if truncated.
    """
    a = np.asarray(a)
    if a.ndim != 1 or a.size <= max_rows:
        return a
    k = max_rows // 2
    return np.concatenate([a[:k], np.array([np.nan]), a[-k:]])


def _preview_2d_rows(a: np.ndarray, max_rows: int) -> np.ndarray:
    """Return a preview of a 2D array by rows, truncating with NaN row if too many.

    Args:
        a: Input 2D array.
        max_rows: Maximum number of rows to display.

    Returns:
        A 2D array with at most max_rows rows, with a NaN row in the middle if truncated.
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
    s: float,
    y: np.ndarray,
    degree: int | list[int],
    spacing_resolved: float | None = None,
    rrms: np.ndarray | None = None,
) -> dict:
    """Create a diagnostics dictionary for derivative approximations.

    Args:
        x: Physical sample points.
        t: Offsets from expansion point (x - x0).
        u: Scaled offsets used for polynomial fitting.
        s: Scaling factor applied to offsets.
        y: Function values at sample points.
        degree: Degree of the polynomial fit (int or list of int for multi-component).
        spacing_resolved: Resolved spacing used (None if not applicable).
        rrms: Residual root-mean-square error of the polynomial fit (None if not applicable).

    Returns:
        A dictionary containing the diagnostics information.
    """
    out = {
        "x": x,
        "t": t,
        "u": u,
        "scale_s": float(s),
        "y": y,
        "degree": degree,
    }
    if spacing_resolved is not None:
        out["spacing_resolved"] = float(spacing_resolved)
    if rrms is not None:
        out["rrms"] = (
            rrms if not (rrms.ndim == 1 and rrms.size == 1) else float(rrms[0])
        )
    return out
