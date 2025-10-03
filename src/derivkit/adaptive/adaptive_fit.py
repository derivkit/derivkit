"""Public API: Adaptive polynomial-fit derivative estimator with gate-based acceptance."""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch
from derivkit.adaptive.diagnostics import make_diagnostics
from derivkit.adaptive.estimator import estimate_component
from derivkit.adaptive.grid import build_x_offsets
from derivkit.adaptive.validate import validate_inputs


class AdaptiveFitDerivative:
    """Adaptive polynomial-fit derivative estimator with gate-based acceptance.

    Fits a local polynomial around ``x0``, checks residual-to-signal and
    conditioning gates, and estimates derivatives component-wise.

    Attributes:
        function: Forward model called as ``function(x)``.
        x0: Expansion point about which the local polynomial is fit.
        min_used_points: Hard lower bound on usable samples after pruning.
    """
    def __init__(self, function, x0: float):
        """Initialize the adaptive derivative estimator.

        Args:
            function: The function to differentiate. Must accept a single
                float argument and return either a scalar or a 1D numpy array.
            x0: The point at which to estimate the derivative.
        """
        self.function = function
        self.x0 = float(x0)
        self.min_used_points = 5  # hard lower bound on min_samples

    def differentiate(
            self,
            order: int = 1,
            min_samples: int = 7,
            include_zero: bool = True,
            acceptance: str | float = "balanced",
            n_workers: int = 1,
            *,
            diagnostics: bool = False,
    ):
        """Estimate the derivative at ``x0`` with gate-based acceptance.

        Constructs a local grid around ``x0``, evaluates the function in batch,
        and fits a polynomial sufficient for the requested derivative order.
        Acceptance thresholds for residuals (``tau_res``) and conditioning
        (``kappa_max``) are derived from the ``acceptance`` setting.

        Args:
            order: Derivative order to estimate. Defaults to 1.
            min_samples: Number of grid points to evaluate. Acts as an evaluation
                budget; the fitter may use a subset but never fewer than
                ``min_used_points``. Defaults to 7.
            include_zero: Whether to include ``x0`` (zero offset) in the evaluation
                grid. Defaults to True.
            acceptance: Preset string (``"strict"``, ``"balanced"``, ``"loose"``,
                ``"very_loose"``) or a float in ``0 < a < 1`` controlling thresholds
                via geometric interpolation. Defaults to "balanced".
            n_workers: Number of workers for batched evaluations. Defaults to 1.
            diagnostics: If True, also return a diagnostics dictionary with grid
                data and per-component outcomes. Defaults to False.

        Returns:
            The derivative estimate at ``x0``. A scalar is returned for scalar
            functions; otherwise a 1D array of per-component estimates. If
            ``diagnostics=True``, returns a tuple ``(estimate, diag)`` where
            ``diag`` is a diagnostics dictionary with keys described in
             :func:`make_diagnostics`.

        Raises:
            ValueError: If inputs are invalid (unsupported ``order``,
                insufficient samples, or invalid ``acceptance``) or if function
                outputs are inconsistent across the grid (e.g., wrong shape or
                non-finite values).
        """
        validate_inputs(order, min_samples, self.min_used_points)

        # First, resolve the acceptance setting and map it to (tau_res, kappa_max).
        tau, kappa_cap = self._resolve_acceptance(acceptance)

        # 1) Build the grid offsets and absolute x values.
        x_offsets, _ = build_x_offsets(
            x0=self.x0,
            order=order,
            include_zero=include_zero,
            min_samples=min_samples,
            min_used_points=self.min_used_points,
        )
        x_values = self.x0 + x_offsets

        # 2) Evaluate the function in batch mode.
        y = eval_function_batch(self.function, x_values, n_workers)
        n_components = y.shape[1]
        derivs = np.empty(n_components, dtype=float)

        outcomes = []

        # 3) Loop over components and estimate each one.
        for i in range(n_components):
            outcome = estimate_component(
                x0=self.x0,
                x_values=x_values,
                y_values=y[:, i],
                order=order,
                tau_res=tau,
                kappa_max=kappa_cap,
            )
            derivs[i] = outcome.value
            outcomes.append(outcome)

        result = derivs.item() if derivs.size == 1 else derivs

        if not diagnostics:
            return result

        # Build diagnostics dictionary.
        diag = make_diagnostics(
            outcomes=outcomes,
            x_all=x_values,
            y_all=y,
            order=order,
            min_samples=min_samples,
            include_zero=include_zero,
            tau_res=tau,
            kappa_max=kappa_cap,
        )
        return result, diag

    def _differentiate(
        self,
        order: int = 1,
        min_samples: int = 7,
        include_zero: bool = True,
        tau_res: float = 5e-2,
        n_workers: int = 1,
    ):
        """Internal variant that exposes ``tau_res`` directly.

        Bypasses the acceptance resolver and supplies the residual-to-signal threshold
        explicitly. Uses a fixed conditioning cap internally.

        Args:
            order: Derivative order to estimate.
            min_samples: Number of grid points to evaluate.
            include_zero: Whether to include ``x0`` in the grid.
            tau_res: Residual-to-signal threshold (smaller is stricter).
            n_workers: Number of workers for batched evaluations.

        Returns:
            The derivative estimate at ``x0`` (scalar or per-component).
        """
        validate_inputs(order, min_samples, self.min_used_points)

        x_offsets, _ = build_x_offsets(
            x0=self.x0,
            order=order,
            include_zero=include_zero,
            min_samples=min_samples,
            min_used_points=self.min_used_points,
        )
        x_values = self.x0 + x_offsets

        y = eval_function_batch(self.function, x_values, n_workers)
        n_components = y.shape[1]
        derivs = np.empty(n_components, dtype=float)

        for i in range(n_components):
            outcome = estimate_component(
                x0=self.x0,
                x_values=x_values,
                y_values=y[:, i],
                order=order,
                tau_res=float(tau_res),
                kappa_max=1e8,  # sensible default cap
            )
            derivs[i] = outcome.value

        return derivs.item() if derivs.size == 1 else derivs

    def _resolve_acceptance(self, acceptance) -> tuple[float, float]:
        """Convert the acceptance setting into residual and conditioning thresholds.

        Args:
            acceptance: Either one of the preset strings ("strict", "balanced",
                "loose", "very_loose") or a floating-point number between zero and
                one (exclusive). Smaller values mean stricter thresholds; larger
                values mean looser thresholds.

        Returns:
            A pair (tau_res, kappa_max) where:
              - tau_res is the residual-to-signal threshold used by the estimator.
              - kappa_max is the maximum allowed conditioning value for the fit.

        Raises:
            ValueError: If the preset is not recognized or if a floating-point
                value outside the open interval (zero, one) is provided.
        """
        tau_min, tau_max = 0.03, 0.20     # residual-to-signal gate
        kappa_min, kappa_max = 1e7, 1e10  # conditioning gate

        if isinstance(acceptance, str):
            key = acceptance.strip().lower()
            preset_a = {
                "strict": 0.0,
                "balanced": 0.35,
                "loose": 0.70,
                "very_loose": 1.0,
            }
            if key not in preset_a:
                raise ValueError(
                    f"unknown acceptance preset '{acceptance}'. "
                    f"choose one of {list(preset_a.keys())} or pass a float in (0,1)."
                )
            a = preset_a[key]
        else:
            a = float(acceptance)
            if not (0.0 < a < 1.0):
                raise ValueError("acceptance as float must be in (0, 1).")

        # geometric interpolation to keep ratios sensible
        tau = float(tau_min * (tau_max / tau_min) ** a)
        kappa = float(kappa_min * (kappa_max / kappa_min) ** a)
        return tau, kappa
