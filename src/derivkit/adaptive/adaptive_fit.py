"""Adaptive polynomial-fit derivatives for estimating derivatives from function samples spaced around x0."""

from __future__ import annotations

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import (
    make_derivative_diag,
    print_derivative_diagnostics,
)
from derivkit.adaptive.grid import make_grid
from derivkit.adaptive.polyfit_utils import (
    choose_degree,
    extract_derivative,
    fit_multi_power,
    scale_offsets,
)


class AdaptiveFitDerivative:
    """Estimate first- or k-th derivatives by fitting a single local polynomial around x0; here 'higher-order' refers to the polynomial's degree, not necessarily the derivative order."""

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
        direction: str = "both",
        base_abs: float | None = None,
        use_physical_grid: bool = False,
        n_workers: int = 1,
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
            direction: Sampling side relative to x0: "both", "pos", or "neg".
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
        need_min = max(5, order + 2)

        # 1) First create a grid of points
        x, t, n_pts, spacing_resolved, direction_used = make_grid(
            self.x0,
            n_points=n_points,
            spacing=spacing,
            direction=direction,
            base_abs=base_abs,
            need_min=need_min,
            use_physical_grid=use_physical_grid,
        )

        # 2) Then evaluate the function at those points
        ys = eval_function_batch(self.func, x, n_workers=n_workers)
        if ys.ndim == 1:
            ys = ys[:, None]
        n_components = ys.shape[1]

        # 3) Fit polynomial
        offsets, factor = scale_offsets(t)
        deg = choose_degree(order, n_pts, extra=5)
        coeffs, rrms = fit_multi_power(offsets, ys, deg)

        # 4) Compute derivative
        deriv = extract_derivative(coeffs, order, factor)
        out = deriv.item() if n_components == 1 else deriv

        if not diagnostics:
            return out

        # 5) If diagnostics requested, prepare diagnostics info
        degree_out = (
            int(deg) if n_components == 1 else [int(deg)] * n_components
        )
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
            "n_points": n_pts if use_physical_grid else n_points,
            "direction": direction_used,
            "spacing": spacing,  # may be ndarray if physical grid
            "base_abs": base_abs,
            "spacing_resolved": spacing_resolved,
            "n_workers": n_workers,
            **(meta or {}),
        }
        print_derivative_diagnostics(diag, meta=meta_payload)
        return out, {**diag, "x0": self.x0, "meta": meta_payload}
