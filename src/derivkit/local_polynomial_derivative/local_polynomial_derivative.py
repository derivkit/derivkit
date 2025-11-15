"""Local polynomial-regression derivative estimator."""

from __future__ import annotations

import math
from typing import Any, Callable

from derivkit.local_polynomial_derivative.diagnostics import make_diagnostics
from derivkit.local_polynomial_derivative.fit import trimmed_polyfit
from derivkit.local_polynomial_derivative.local_poly_config import (
    LocalPolyConfig,
)
from derivkit.local_polynomial_derivative.sampling import build_samples


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
                Function to differentiate. It should take a float and return either
                a scalar or a NumPy array (vector or tensor); derivatives are
                computed componentwise with the same output shape.
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
        """Local polynomial-regression derivative estimator.

        This class estimates derivative at ``x0`` by sampling the function in a
        small neighborhood around that point, fitting a polynomial to those samples,
        and trimming away samples whose residuals are inconsistent with the fit.
        Once a stable local polynomial is obtained, the k-th derivative is read off
        directly from the coefficient of the fitted polynomial (``k! * a_k``). The
        method works for scalar or vector/tensor-valued functions, and can optionally return
        a diagnostics dictionary showing which samples were used, how trimming
        behaved, and whether the final fit passed all internal checks.

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
            raise ValueError(f"order must be at least 1 but is {order}.")
        if n_workers < 1:
            raise ValueError(f"n_workers must be at least 1 but is {n_workers}.")

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
