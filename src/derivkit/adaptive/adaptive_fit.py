"""Adaptive polynomial-fit derivatives for estimating derivatives from function samples spaced around x0."""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import (
    make_derivative_diag,
    print_derivative_diagnostics,
)

from derivkit.adaptive.polyfit_utils import (
    extract_derivative,
    fit_multi_power,
    scale_offsets,
)

from derivkit.adaptive.grid import make_grid, chebyshev_offsets
from derivkit.adaptive.transforms import (
    signed_log_forward, signed_log_to_physical, pullback_signed_log,
    sqrt_domain_forward, sqrt_to_physical, pullback_sqrt_at_zero,
)

from derivkit.adaptive.spacing import resolve_spacing


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
        use_physical_grid: bool = False,
        n_workers: int = 1,
        grid: np.ndarray | None = None,
        domain: "tuple[float | None, float | None] | None" = None,
        ridge: float = 1e-8,
        diagnostics: bool = False,
        meta: dict | None = None,
    ):
        """Compute the derivative of specified order at x0 using an adaptive polynomial fit.

        This method samples the function at points around x0, fits a polynomial to those
        samples, and extracts the requested derivative from the fitted coefficients. It
        supports scalar or vector-valued functions and selects a degree consistent with
        the number of points and the derivative order.
        Unlike finite-difference methods, the spacing here controls how far sample
        points are placed from x0 for the polynomial fit, not the step size used in
        a finite-difference stencil.

        Args:
            order: The derivative order to compute (>= 1).
            n_points: Number of sample points around x0 used for the fit. Default is 10.
            spacing: Controls how far sample points lie from x0:
                    - positive float → fixed absolute distance,
                    - percentage string like "1%" → relative to the magnitude of x0,
                    - "auto" → 2% of the magnitude of x0, with a minimum floor,
                    - NumPy array → explicit offsets when `use_physical_grid=True`.
            base_abs: Absolute spacing floor used by "auto" and percentage modes near x0≈0.
                If None, defaults to 1e-3.
            use_physical_grid: If True, `spacing` must be an array of explicit offsets.
            n_workers: Number of worker processes for parallel evaluation (1 = serial).
            diagnostics: If True, also return a diagnostics dictionary.
            meta: Optional metadata to include in diagnostics.

        Returns:
            The derivative at x0. For vector-valued functions, returns a 1D NumPy array.
            If `diagnostics=True`, returns `(derivative, diagnostics_dict)`.

        Raises:
            ValueError: If `order < 1` or spacing/direction parameters are invalid.
        """
        if order < 1:
            raise ValueError("order must be >= 1")

            # ---------------------------
            # 1) Build an initial grid
            # ---------------------------
        if grid is not None:
            # explicit OFFSETS around x0 (expert mode)
            t = np.asarray(grid, dtype=float).ravel()
            if 0.0 not in t:
                t = np.sort(np.append(t, 0.0))
            else:
                t = np.sort(t)
            x = self.x0 + t
            spacing_resolved = np.nan
            mode = "x"
            sign_used = None

        elif use_physical_grid:
            # 'spacing' is an array of physical x-samples in this mode
            provisional_min = 1  # we validate properly after mode selection
            x, t, _n_pts, spacing_resolved, _ = make_grid(
                self.x0,
                n_points=n_points,
                spacing=spacing,
                base_abs=base_abs,
                need_min=provisional_min,
                use_physical_grid=True,
            )
            mode = "x"
            sign_used = None

        else:
            # default symmetric Chebyshev OFFSETS around x0
            H = resolve_spacing(spacing, float(self.x0), base_abs)
            t = chebyshev_offsets(H, n_points, include_center=True)
            x = self.x0 + t
            spacing_resolved = float(H)
            mode = "x"
            sign_used = None

            # ------------------------------------------------
            # 1a) Optional domain-aware transform selection
            # ------------------------------------------------
        single_sign = False
        single_pos = False  # initialize to avoid NameErrors

        if domain is not None:
            lo, hi = domain
            single_pos = (lo is not None and lo >= 0.0) and (hi is None or hi >= 0.0)
            single_neg = (hi is not None and hi <= 0.0) and (lo is None or lo <= 0.0)
            single_sign = single_pos or single_neg

            if single_sign and self.x0 == 0.0:
                # Boundary at zero → sqrt-domain with symmetric u-grid
                _u0, sign_used = sqrt_domain_forward(self.x0, +1.0 if single_pos else -1.0)
                Hu = resolve_spacing(spacing, 0.0, base_abs=1e-3)
                tu = chebyshev_offsets(Hu, n_points, include_center=True)
                t = tu  # internal u-offsets
                x = sqrt_to_physical(t, sign_used)  # physical x
                spacing_resolved = float(Hu)
                mode = "sqrt"

            elif single_sign:
                # If symmetric x-grid violates domain and x0 != 0 → use signed-log
                violates = ((lo is not None) and np.any(x < lo)) or ((hi is not None) and np.any(x > hi))
                if violates and self.x0 != 0.0:
                    q0, sgn = signed_log_forward(self.x0)
                    Hq = resolve_spacing(spacing, q0, base_abs=1e-3)
                    tq = chebyshev_offsets(Hq, n_points, include_center=True)
                    t = tq  # internal q-offsets
                    x = signed_log_to_physical(q0 + tq, sgn)
                    spacing_resolved = float(Hq)
                    mode = "signed_log"

        # ----------------------------------------------------------
        # 1b) Ensure we have enough samples for the chosen mode/order
        #     - If user supplied grid/physical points → raise if too few
        #     - If default Chebyshev → auto-bump to the minimum needed
        # ----------------------------------------------------------
        n_eff = len(t)
        # internal derivative order needed: sqrt needs g^(2*order) at 0
        deg_req = (2 * order) if (mode == "sqrt") else order
        min_pts = 2 * deg_req + 1  # ensure (n_eff - 1)//2 ≥ deg_req

        if n_eff < min_pts:
            if grid is not None or use_physical_grid:
                raise ValueError(
                    f"Not enough samples for order={order} (mode={mode}). "
                    f"Need ≥{min_pts} points, got {n_eff}."
                )
            # We control the default Chebyshev grid → rebuild with more points
            target = max(min_pts, n_points)
            if mode == "x":
                t = chebyshev_offsets(spacing_resolved, target, include_center=True)
                x = self.x0 + t
            elif mode == "signed_log":
                q0, sgn = signed_log_forward(self.x0)
                Hq = resolve_spacing(spacing, q0, base_abs=1e-3)
                tq = chebyshev_offsets(Hq, target, include_center=True)
                t = tq
                x = signed_log_to_physical(q0 + tq, sgn)
                spacing_resolved = float(Hq)
            elif mode == "sqrt":
                # reuse sign_used from selection above
                Hu = resolve_spacing(spacing, 0.0, base_abs=1e-3)
                tu = chebyshev_offsets(Hu, target, include_center=True)
                t = tu
                x = sqrt_to_physical(t, sign_used)
                spacing_resolved = float(Hu)
            n_eff = len(t)

        # ------------------------------------
        # 2) Evaluate function on chosen grid
        # ------------------------------------
        ys = eval_function_batch(self.func, x, n_workers=n_workers)
        if ys.ndim == 1:
            ys = ys[:, None]
        n_components = ys.shape[1]

        # --------------------------
        # 3) Fit polynomial (stable)
        # --------------------------
        offsets, factor = scale_offsets(t)
        # Degree with a little elbow room; sqrt(order=2) needs up to 4th in u
        extra_need = 4 if (mode == "sqrt" and order == 2) else 2
        deg = min(deg_req + extra_need, (n_eff - 1) // 2)
        coeffs, rrms = fit_multi_power(offsets, ys, deg, ridge=ridge)

        # --------------------------
        # 4) Extract derivative
        # --------------------------
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

        # ---------------------------------
        # 5) Diagnostics & pretty printing
        # ---------------------------------
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