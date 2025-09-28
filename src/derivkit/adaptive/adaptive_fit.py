"""Public API: AdaptiveFitDerivative (thin orchestrator)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Optional

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import DiagnosticsRecorder
from derivkit.adaptive.estimator import estimate_component
from derivkit.adaptive.fallback import fallback_fd as _default_fallback_fn
from derivkit.adaptive.fit_core import fit_once as _default_fit_once_fn
from derivkit.adaptive.grid import build_x_offsets
from derivkit.adaptive.validate import validate_inputs
from derivkit.adaptive.weights import (
    inverse_distance_weights as _default_weight_fn,
)

warnings.simplefilter("once", category=RuntimeWarning)


class AdaptiveFitDerivative:
    """Adaptive polynomial-fit derivative estimator."""

    def __init__(
        self,
        function,
        x0: float,
        *,
        fit_once_fn: Optional[Callable] = None,
        fallback_fn: Optional[Callable] = None,
        weight_fn: Optional[Callable] = None,
    ):
        """Initialize AdaptiveFitDerivative.

        Args:
            function: Callable mapping a float to a scalar or 1D array-like.
            x0: Expansion point for the derivative.
            fit_once_fn: Optional override for the single-fit routine. Signature:
                ``(x0, x_vals, y_vals, order, *, weight_eps_frac=...) -> dict``.
            fallback_fn: Optional override for the fallback path when the fit is
                rejected or unavailable. Signature:
                ``(function, x0, order, n_workers) -> float | np.ndarray``.
            weight_fn: Optional override for the per-sample weighting function.
                Signature: ``(x_vals, x0, eps_frac) -> np.ndarray``.
        """
        self.function = function
        self.x0 = float(x0)
        self.diagnostics_data = None
        self.min_used_points = 5  # floor for usable samples in a fit

        # Publicly injectable hooks (default to your current implementations)
        self._fit_once_fn = fit_once_fn or _default_fit_once_fn
        self._fallback_fn = fallback_fn or _default_fallback_fn
        self._weight_fn = weight_fn or _default_weight_fn

    def differentiate(
        self,
        order: int = 1,
        min_samples: int = 7,
        diagnostics: bool = False,
        fallback_mode: str = "finite_difference",
        fit_tolerance: float = 0.05,
        include_zero: bool = True,
        floor_accept_multiplier: float = 2.0,
        n_workers: int = 1,
        *,
        # Optional per-call overrides (also public) if you want
        fit_once_fn: Optional[Callable] = None,
        fallback_fn: Optional[Callable] = None,
        weight_fn: Optional[Callable] = None,
    ):
        """Estimate the derivative at ``x0`` by adaptive polynomial fitting.

        Args:
            order: Derivative order to estimate.
            min_samples: Minimum number of total samples to build the grid.
            diagnostics: If True, return a tuple of (value, diagnostics_dict).
            fallback_mode: Strategy when at the sample-count floor and the fit
                misses tolerance (e.g., ``"finite_difference"``, ``"auto"``,
                ``"poly_at_floor"``).
            fit_tolerance: Maximum allowed relative residual to accept a fit.
            include_zero: Whether to include the center point in the grid.
            floor_accept_multiplier: Extra headroom on the max residual for
                auto-acceptance at the floor.
            n_workers: Parallelism hint for function evaluation / fallback.
            fit_once_fn: Per-call override of the fit hook.
            fallback_fn: Per-call override of the fallback hook.
            weight_fn: Per-call override of the weight function.

        Returns:
            If ``diagnostics`` is False, returns a float or ``np.ndarray``.
            If ``diagnostics`` is True, returns ``(value, diagnostics_dict)``.
        """
        validate_inputs(order, min_samples, self.min_used_points)

        # 1) Build sampling grid around x0
        x_offsets, required_points = build_x_offsets(
            x0=self.x0,
            order=order,
            include_zero=include_zero,
            min_samples=min_samples,
            min_used_points=self.min_used_points,
        )
        x_values = self.x0 + x_offsets

        # 2) Evaluate function on the grid (shape: (n_points, n_components))
        Y = eval_function_batch(self.function, x_values, n_workers)
        n_components = Y.shape[1]
        derivs = np.zeros(n_components, dtype=float)

        # 3) Diagnostics recorder
        rec = DiagnosticsRecorder(enabled=diagnostics, x_all=x_values, y_all=Y)

        # Resolve hooks (per-call overrides win over ctor defaults)
        weight_fn = weight_fn or self._weight_fn
        fit_once_fn = fit_once_fn or self._fit_once_fn
        fallback_fn = fallback_fn or self._fallback_fn

        # Prepare closures with your existing default parameters
        ffit = (  # noqa: E731
            lambda xv, yv: fit_once_fn(self.x0, xv, yv, order, weight_eps_frac=1e-3)
        )
        ffallback = lambda: fallback_fn(  # noqa: E731
            self.function, self.x0, order, n_workers
        )

        # 4) Per-component estimation (accept / prune / fallback handled internally)
        for i in range(n_components):
            outcome = estimate_component(
                x0=self.x0,
                x_values=x_values,
                y_values=Y[:, i],
                order=order,
                required_points=required_points,
                fit_tolerance=fit_tolerance,
                fallback_mode=fallback_mode,
                floor_accept_multiplier=floor_accept_multiplier,
                fit_once_fn=ffit,
                fallback_fn=ffallback,
            )
            derivs[i] = outcome.value
            rec.add(outcome)

        value = derivs.item() if derivs.size == 1 else derivs
        return (value, rec.build()) if diagnostics else value
