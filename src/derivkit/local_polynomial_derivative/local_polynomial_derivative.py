#!/usr/bin/env python3
"""Local polynomial-regression derivative estimator."""

from __future__ import annotations

import math
from typing import Any, Callable

from .diagnostics import make_diagnostics
from .fit import trimmed_polyfit
from .local_poly_config import LocalPolyConfig
from .sampling import build_samples


class LocalPolynomialDerivative:
    """Estimates derivatives via trimmed local polynomial regression around x0."""

    def __init__(
        self,
        func: Callable[[float], Any],
        x0: float,
        config: LocalPolyConfig | None = None,
    ):
        """Initializes the LocalPolynomialDerivative instance.

        Args:
            func:
                The function for which to estimate derivatives. It should take a float
                and return a scalar or np.ndarray.
            x0:
                The point at which to estimate the derivative.
            config:
                An optional LocalPolyConfig instance with configuration settings.
        """
        self.func = func
        self.x0 = float(x0)
        self.config = config or LocalPolyConfig()

    def differentiate(
        self,
        order: int = 1,
        degree: int | None = None,
        n_workers: int = 1,
        diagnostics: bool = False,
    ):
        """Estimates the derivative of specified order at x0 using local polynomial regression.

        This method fits a local polynomial to samples of the function around x0,
        trims outliers based on residuals, and extracts the derivative from the fitted polynomial.

        Args:
            order:
                The order of the derivative to estimate (must be >= 1).
            degree:
                The degree of the polynomial fit. If None, it is set to max(order +
                2, 3) but capped by config.max_degree.
            n_workers:
                The number of parallel workers for function evaluation (must be >= 1).
            diagnostics:
                If True, returns a diagnostics dictionary along with the derivative estimate.

        Returns:
            If diagnostics is False:
                The estimated derivative (float or np.ndarray).
            If diagnostics is True:
                A tuple (derivative, diagnostics_dict).

        Raises:
            ValueError:
                If order < 1, n_workers < 1, or degree < order.
        """
        if order < 1:
            raise ValueError("order must be >= 1.")
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1.")

        # Choose polynomial degree with a bit of headroom.
        if degree is None:
            degree = max(order + 2, 3)
        degree = int(min(degree, self.config.max_degree))
        if degree < order:
            raise ValueError("degree must be >= order.")

        xs, ys = build_samples(self.func, self.x0, self.config, n_workers=n_workers)
        coeffs, used_mask, ok = trimmed_polyfit(self.x0, self.config, xs, ys, degree)

        n_comp = coeffs.shape[1]
        if order > degree:
            raise ValueError("Internal error: order > degree in derivative extraction.")

        factorial = math.factorial(order)
        a_k = coeffs[order]
        deriv = factorial * a_k
        deriv_out = float(deriv[0]) if n_comp == 1 else deriv

        if not diagnostics:
            return deriv_out

        diag = make_diagnostics(
            self.x0,
            self.config,
            xs,
            ys,
            used_mask,
            coeffs,
            degree,
            order,
            ok,
        )
        diag["n_workers"] = int(n_workers)
        return deriv_out, diag
