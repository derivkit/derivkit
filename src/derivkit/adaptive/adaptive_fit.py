"""Adaptive polynomial-fit derivatives for estimating derivatives from function samples spaced around x0."""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import (
    fit_is_obviously_bad,
    make_derivative_diag,
    print_derivative_diagnostics,
)
from derivkit.adaptive.grid import (
    ensure_min_samples_and_maybe_rebuild,
    make_domain_aware_chebyshev_grid,
)
from derivkit.adaptive.polyfit_utils import (
    assess_polyfit_quality,
    fit_with_headroom_and_maybe_minimize,
    pullback_derivative_from_fit,
    scale_offsets,
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
            - grid=None: symmetric Chebyshev offsets around x0 with half-width from `spacing`.
            - grid=("offsets", arr): explicit offsets t; samples at x = x0 + t (0 inserted if missing).
            - grid=("absolute", arr): explicit absolute x positions; samples at x = arr.

        Args:
            order: Derivative order (>=1).
            n_points: Number of sample points when building the default grid. Default is 10.
            spacing: Scale for the default symmetric grid around ``x0`` (ignored when ``grid`` is provided).

                Accepted forms:

                - float: interpreted as an absolute half-width ``h``; samples in ``[x0 - h, x0 + h]``.
                - "<pct>%": percentage string; ``h`` is that fraction of a local scale
                  set by ``abs(x0)`` with a floor ``base_abs`` near zero.
                - "auto": choose ``h`` adaptively. DerivKit picks a half-width based on
                  the local scale of ``x0`` with a minimum of ``base_abs``; if ``domain``
                  is given, the interval is clipped to stay inside ``(lo, hi)``. The
                  default grid uses Chebyshev nodes on that interval and always includes
                  the center point.

            base_abs: Absolute spacing floor used by "auto"/percentage near x0≈0. Defaults to ``1e-3`` if not set.
            n_workers: Parallel workers for batched function evals (1 = serial).
            grid: Either ("offsets", array) or ('absolute', array), or None for default.

                This lets the user supply their own sampling points instead of using the
                automatically built Chebyshev grid. With ``("offsets", arr)``, the array
                gives relative offsets from ``x0`` (samples at ``x = x0 + t``). With
                ``('absolute', arr)``, the array gives absolute ``x`` positions. If
                ``None``, the method builds a symmetric default grid around ``x0``.

            domain: Optional (lo, hi) used to trigger domain-aware transforms in default mode.
            ridge: Ridge regularization for polynomial fit. Defaults to 0.0.

                This term adds a small penalty to the fit to keep the coefficients from
                becoming too large when the Vandermonde matrix is nearly singular.
                Increasing ``ridge`` makes the fit more stable but slightly smoother;
                setting it to 0 disables the regularization. Default is 0.0.

            diagnostics: If True, return (derivative, diagnostics_dict).
            meta: Extra metadata to carry in diagnostics.

        Returns:
            Derivative at x0 (scalar or 1D array). If diagnostics=True, also returns a dict.

        Raises:
            ValueError: If inputs are invalid or not enough samples are provided.
        """
        if order < 1:
            raise ValueError("order must be >= 1")

        # 1) Choose sample locations (x, t)
        if grid is not None:
            if not (isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], str)):
                raise ValueError("grid must be ('offsets'|'absolute', numpy_array) or None.")
            kind, arr = grid
            arr = np.asarray(arr, dtype=float).ravel()
            match kind:
                case "offsets":
                    t = np.sort(np.unique(np.append(arr, 0.0)))  # ensure center; sorted for stability
                    x = self.x0 + t
                case "absolute":
                    x = np.sort(arr)
                    t = x - self.x0
                case _:
                    raise ValueError("grid kind must be 'offsets' or 'absolute'.")
            mode, spacing_resolved, sign_used = "x", float("nan"), None
        else:
            mode, x, t, spacing_resolved, sign_used = make_domain_aware_chebyshev_grid(
                self.x0,
                n_points=n_points,
                spacing=spacing,
                base_abs=base_abs,
                domain=domain,
                max_cheby_points=30,
            )

        # 1b) Ensure enough samples (rebuild default Chebyshev grids if needed)
        mode, x, t, spacing_resolved, sign_used = ensure_min_samples_and_maybe_rebuild(
            mode=mode,
            x=x,
            t=t,
            spacing_resolved=spacing_resolved,
            sign_used=sign_used,
            x0=self.x0,
            order=order,
            n_points=n_points,
            spacing=spacing,
            base_abs=base_abs,
            max_cheby_points=30,
        )

        # 2) Evaluate function on the grid
        ys = eval_function_batch(self.func, x, n_workers=n_workers)
        if ys.ndim == 1:
            ys = ys[:, None]
        n_components = ys.shape[1]

        # 3) Polynomial fit (scaled offsets) with headroom + optional minimal-degree swap
        u, factor = scale_offsets(t)
        coeffs, rrms, deg = fit_with_headroom_and_maybe_minimize(
            u, ys, order=order, mode=mode, ridge=ridge, factor=factor
        )

        # 3b) Fit quality (soft warnings only)
        metrics, suggestions = assess_polyfit_quality(
            u, ys, coeffs, deg, ridge=ridge, factor=factor, order=order
        )
        bad, msg = fit_is_obviously_bad(metrics)
        if bad:
            pretty_suggestions = "\n  ".join(suggestions)
            print(
                msg
                + "\nTo improve this derivative, try:\n  "
                + pretty_suggestions
            )

        # 4) Derivative (mode-aware pullback)
        deriv = pullback_derivative_from_fit(
            mode=mode, order=order, coeffs=coeffs, factor=factor, x0=self.x0, sign_used=sign_used
        )
        out = deriv.item() if n_components == 1 else deriv
        if not diagnostics:
            return out

        # 5) Diagnostics (optional)
        degree_out = int(deg) if n_components == 1 else [int(deg)] * n_components
        diag = make_derivative_diag(
            x=x,
            t=t,
            u=u,
            y=ys,
            degree=degree_out,
            spacing_resolved=spacing_resolved,
            rrms=rrms,
            coeffs=coeffs,
            ridge=ridge,
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
