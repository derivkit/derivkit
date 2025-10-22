"""Adaptive polynomial-fit derivatives for estimating derivatives from function samples spaced around x0."""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import (
    make_derivative_diag,
    print_derivative_diagnostics,
)
from derivkit.adaptive.grid import chebyshev_offsets
from derivkit.adaptive.polyfit_utils import (
    assess_polyfit_quality,
    extract_derivative,
    fit_multi_power,
    scale_offsets,
)
from derivkit.adaptive.spacing import resolve_spacing
from derivkit.adaptive.transforms import (
    pullback_signed_log,
    pullback_sqrt_at_zero,
    signed_log_forward,
    signed_log_to_physical,
    sqrt_domain_forward,
    sqrt_to_physical,
)


class AdaptiveFitDerivative:
    """Derivative estimation via a single local polynomial fit around x0."""

    def __init__(self, func, x0: float):
        """Initialize the estimator.

        Args:
            func: Callable mapping a float to a scalar or 1D array-like output.
            x0: Expansion point about which derivatives are computed.
        """
        self.func = func
        self.x0 = float(x0)

    def differentiate(
            self,
            order: int,
            *,
            n_points: int = 10,
            spacing: float | str | None = "auto",
            base_abs: float | None = None,
            n_workers: int = 1,
            grid: tuple[str, np.ndarray] | None = None,  # ('offsets'|'absolute', array)
            domain: "tuple[float | None, float | None] | None" = None,
            ridge: float = 0.0,
            diagnostics: bool = False,
            meta: dict | None = None,
    ):
        """Compute the derivative of specified order at x0 using an adaptive polynomial fit.

        Sampling strategy:
          - grid=None → symmetric Chebyshev offsets around x0 with half-width from `spacing`.
          - grid=("offsets", arr) → explicit offsets t; samples at x = x0 + t (0 inserted if missing).
          - grid=("absolute", arr) → explicit absolute x positions; samples at x = arr.

        Args:
            order: Derivative order (>=1).
            n_points: Number of sample points when building the default grid. Default is 10.
            spacing: Scale for default grid: float, percentage string e.g. "2%", or "auto".
            base_abs: Absolute spacing floor used by "auto"/percentage near x0≈0 (default 1e-3).
            n_workers: Parallel workers for batched function evals (1 = serial).
            grid: Either ('offsets', array) or ('absolute', array), or None for default.
            domain: Optional (lo, hi) used to trigger domain-aware transforms in default mode.
            ridge: Ridge regularization for polynomial fit. Defaults to 0.0.
            diagnostics: If True, return (derivative, diagnostics_dict).
            meta: Extra metadata to carry in diagnostics.

        Returns:
            Derivative at x0 (scalar or 1D array). If diagnostics=True, also returns a dict.

        Raises:
            ValueError: If inputs are invalid or not enough samples are provided.
        """
        if order < 1:
            raise ValueError("order must be >= 1")

        max_cheby_points = 30  # guard for ill-conditioning / too-dense default grids

        mode = "x"  # "x" or "signed_log" or "sqrt"
        sign_used = None
        spacing_resolved = np.nan

        # 1) Choose sample locations x,t
        if grid is not None:
            if not (isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], str)):
                raise ValueError("grid must be ('offsets'|'absolute', numpy_array) or None.")
            kind, arr = grid
            arr = np.asarray(arr, dtype=float).ravel()

            if kind == "offsets":
                # Ensure center point present; sort for stability
                t = np.sort(np.unique(np.append(arr, 0.0)))
                x = self.x0 + t

            elif kind == "absolute":
                x = np.sort(arr)
                t = x - self.x0

            else:
                raise ValueError("grid kind must be 'offsets' or 'absolute'.")

        else:
            # Default: symmetric Chebyshev offsets around x0
            if n_points > max_cheby_points:
                raise ValueError(
                    f"Too many points for default Chebyshev grid (n_points={n_points}, max={max_cheby_points}). "
                    "If you want a denser sampling, increase the local half-width via `spacing` "
                    "(e.g. a larger float or percentage) so points spread out, or pass an explicit grid: "
                    "grid=('offsets', offsets_array) or grid=('absolute', x_array)."
                )
            half_width = resolve_spacing(spacing, float(self.x0), base_abs)
            t = chebyshev_offsets(half_width, n_points, include_center=True)
            x = self.x0 + t
            spacing_resolved = float(half_width)

            # 1a) Optional domain-aware transform selection
            if domain is not None:
                lo, hi = domain
                single_pos = (lo is not None and lo >= 0.0) and (hi is None or hi >= 0.0)
                single_neg = (hi is not None and hi <= 0.0) and (lo is None or lo <= 0.0)
                single_sign = single_pos or single_neg

                if single_sign and self.x0 == 0.0:
                    # Boundary at zero → sqrt-domain with symmetric u-grid
                    _u0, sign_used = sqrt_domain_forward(self.x0, +1.0 if single_pos else -1.0)
                    half_width_u = resolve_spacing(spacing, 0.0, base_abs=1e-3)
                    if n_points > max_cheby_points:
                        raise ValueError(
                            f"Too many points for default Chebyshev grid in sqrt-mode "
                            f"(n_points={n_points}, max={max_cheby_points}). "
                            "Increase `spacing` to spread points or provide an explicit grid."
                        )
                    tu = chebyshev_offsets(half_width_u, n_points, include_center=True)
                    t = tu  # internal u-offsets
                    x = sqrt_to_physical(t, sign_used)  # physical x
                    spacing_resolved = float(half_width_u)
                    mode = "sqrt"

                elif single_sign:
                    # If symmetric x-grid violates domain and x0 != 0 then use signed-log
                    violates = ((lo is not None) and np.any(x < lo)) or ((hi is not None) and np.any(x > hi))
                    if violates and self.x0 != 0.0:
                        q0, sgn = signed_log_forward(self.x0)
                        half_width_q = resolve_spacing(spacing, q0, base_abs=1e-3)
                        if n_points > max_cheby_points:
                            raise ValueError(
                                f"Too many points for default Chebyshev grid in signed-log mode "
                                f"(n_points={n_points}, max={max_cheby_points}). "
                                "Increase `spacing` to spread points or provide an explicit grid."
                            )
                        tq = chebyshev_offsets(half_width_q, n_points, include_center=True)
                        t = tq  # internal q-offsets
                        x = signed_log_to_physical(q0 + tq, sgn)
                        spacing_resolved = float(half_width_q)
                        mode = "signed_log"

        # 1b) Ensure enough samples for desired derivative
        n_eff = len(t)
        deg_req = (2 * order) if (mode == "sqrt") else order
        min_pts = 2 * deg_req + 1  # ensure (n_eff - 1)//2 ≥ deg_req

        if n_eff < min_pts:
            if grid is not None:
                raise ValueError(
                    f"Not enough samples for order={order} (mode={mode}). Need ≥{min_pts} points, got {n_eff}."
                )
            # We control the default grid → rebuild with more points
            target = max(min_pts, n_points)
            if target > max_cheby_points:
                raise ValueError(
                    f"Requested derivative/order requires at least {min_pts} points, "
                    f"but the default Chebyshev grid is capped at {max_cheby_points}. "
                    "Increase `spacing` (larger half-width spreads Chebyshev nodes), reduce `order`, "
                    "or supply an explicit grid via grid=('offsets', ...) or grid=('absolute', ...)."
                )
            if mode == "x":
                t = chebyshev_offsets(spacing_resolved, target, include_center=True)
                x = self.x0 + t
            elif mode == "signed_log":
                q0, sgn = signed_log_forward(self.x0)
                half_width_q = resolve_spacing(spacing, q0, base_abs=1e-3)
                tq = chebyshev_offsets(half_width_q, target, include_center=True)
                t = tq
                x = signed_log_to_physical(q0 + tq, sgn)
                spacing_resolved = float(half_width_q)
            elif mode == "sqrt":
                half_width_u = resolve_spacing(spacing, 0.0, base_abs=1e-3)
                tu = chebyshev_offsets(half_width_u, target, include_center=True)
                t = tu
                x = sqrt_to_physical(t, sign_used)
                spacing_resolved = float(half_width_u)
            n_eff = len(t)

        # 2) Evaluate function on a grid
        ys = eval_function_batch(self.func, x, n_workers=n_workers)
        if ys.ndim == 1:
            ys = ys[:, None]
        n_components = ys.shape[1]

        # 3) Poly fit
        offsets, factor = scale_offsets(t)
        extra_need = 4 if (mode == "sqrt" and order == 2) else 2
        deg_req = (2 * order) if (mode == "sqrt") else order
        deg_hi = min(deg_req + extra_need, (n_eff - 1) // 2)
        # First, fit with headroom(default path)
        coeffs, rrms = fit_multi_power(offsets, ys, deg_hi, ridge=ridge)
        deg = deg_hi  # current choice

        # Heuristic gate for “essentially exact polynomial”
        y_scale = max(1.0, float(np.nanmax(np.abs(ys))))
        if (deg_hi > deg_req) and (order >= 3) and float(np.nanmax(rrms)) <= 1e-12 * y_scale:
            # Refit at minimal degree
            coeffs_min, rrms_min = fit_multi_power(offsets, ys, deg_req, ridge=ridge)

            # Compare derivatives at x0
            if mode == "signed_log":
                dfdq_hi = extract_derivative(coeffs, 1, factor)
                dfdq_min = extract_derivative(coeffs_min, 1, factor)
                if order == 1:
                    deriv_hi = pullback_signed_log(1, self.x0, dfdq_hi)
                    deriv_min = pullback_signed_log(1, self.x0, dfdq_min)
                else:  # order == 2
                    d2fdq2_hi = extract_derivative(coeffs, 2, factor)
                    d2fdq2_min = extract_derivative(coeffs_min, 2, factor)
                    deriv_hi = pullback_signed_log(2, self.x0, dfdq_hi, d2fdq2_hi)
                    deriv_min = pullback_signed_log(2, self.x0, dfdq_min, d2fdq2_min)
            elif mode == "sqrt":
                if order == 1:
                    g2_hi = extract_derivative(coeffs, 2, factor)
                    g2_min = extract_derivative(coeffs_min, 2, factor)
                    deriv_hi = pullback_sqrt_at_zero(1, sign_used, g2=g2_hi)
                    deriv_min = pullback_sqrt_at_zero(1, sign_used, g2=g2_min)
                else:  # order == 2
                    g4_hi = extract_derivative(coeffs, 4, factor)
                    g4_min = extract_derivative(coeffs_min, 4, factor)
                    deriv_hi = pullback_sqrt_at_zero(2, sign_used, g4=g4_hi)
                    deriv_min = pullback_sqrt_at_zero(2, sign_used, g4=g4_min)
            else:
                deriv_hi = extract_derivative(coeffs, order, factor)
                deriv_min = extract_derivative(coeffs_min, order, factor)

            # Prefer minimal degree if derivatives are indistinguishable
            if np.allclose(deriv_hi, deriv_min, rtol=0.0, atol=1e-9):
                coeffs = coeffs_min
                rrms = rrms_min
                deg = deg_req

        # Evaluate fit quality (component-wise worst-case metrics)
        metrics, suggestions = assess_polyfit_quality(
            offsets,  # u (scaled offsets)
            ys,  # y (n_pts, n_comp)
            coeffs,  # (deg+1, n_comp)
            deg,
            ridge=ridge,
            factor=factor,
            order=order,
        )

        # define a “clearly bad” gate (tighter/looser if you like)
        bad = (
                metrics["rrms_rel"] > 5 * metrics["thresholds"]["rrms_rel"]
                or metrics["loo_rel"] > 5 * metrics["thresholds"]["loo_rel"]
                or metrics["cond_vdm"] > 10 * metrics["thresholds"]["cond_vdm"]
                or metrics["deriv_rel"] > 5 * metrics["thresholds"]["deriv_rel"]
        )

        # Surface suggestions when it looks poor; keep running (non-fatal)
        if bad:
            print(
                "Polynomial fit looks unstable: "
                f"rrms_rel={metrics['rrms_rel']:.2e}, "
                f"loo_rel={metrics['loo_rel']:.2e}, "
                f"cond_vdm={metrics['cond_vdm']:.2e}, "
                f"deriv_rel={metrics['deriv_rel']:.2e}. "
                "Suggestions: " + " ".join(suggestions)
            )

        # 4) Derivative
        if mode == "signed_log":
            dfdq = extract_derivative(coeffs, 1, factor)
            if order == 1:
                deriv = pullback_signed_log(1, self.x0, dfdq)
            elif order == 2:
                d2fdq2 = extract_derivative(coeffs, 2, factor)
                deriv = pullback_signed_log(2, self.x0, dfdq, d2fdq2)
            else:
                raise NotImplementedError("signed-log path supports orders 1 and 2.")
        elif mode == "sqrt":
            if order == 1:
                g2 = extract_derivative(coeffs, 2, factor)
                deriv = pullback_sqrt_at_zero(1, sign_used, g2=g2)
            elif order == 2:
                g4 = extract_derivative(coeffs, 4, factor)
                deriv = pullback_sqrt_at_zero(2, sign_used, g4=g4)
            else:
                raise NotImplementedError("sqrt path supports orders 1 and 2.")
        else:
            deriv = extract_derivative(coeffs, order, factor)

        out = deriv.item() if n_components == 1 else deriv
        if not diagnostics:
            return out

        # 5) Diagnostics (optional)
        degree_out = int(deg) if n_components == 1 else [int(deg)] * n_components
        diag = make_derivative_diag(
            x=x,
            t=t,
            u=offsets,
            s=factor,
            y=ys,
            degree=degree_out,
            spacing_resolved=spacing_resolved,
            rrms=rrms,
            coeffs=coeffs,
            ridge=ridge,
            factor=factor,
            order=order,
        )

        meta_payload = {
            "x0": self.x0,
            "order": order,
            "n_points": len(x),
            "spacing": spacing,
            "base_abs": base_abs,
            "spacing_resolved": spacing_resolved,
            "n_workers": n_workers,
            "domain": domain,
            "mode": mode,
            "ridge": ridge,
            **(meta or {}),
        }
        print_derivative_diagnostics(diag, meta=meta_payload)
        return out, {**diag, "x0": self.x0, "meta": meta_payload}
