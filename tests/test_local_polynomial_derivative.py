"""Unit tests for LocalPolynomialDerivative."""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest

from derivkit.local_polynomial_derivative.local_poly_config import (
    LocalPolyConfig,
)
from derivkit.local_polynomial_derivative.local_polynomial_derivative import (
    LocalPolynomialDerivative,
)


def _rel_err(a: float, b: float) -> float:
    """Computes relative error between a and b."""
    d = max(1.0, abs(a), abs(b))
    return abs(a - b) / d


def eval_poly_derivative(k: int, x: float, coeffs, degree: int) -> float:
    """Analytic-like k-th derivative of p(x) = sum_j coeffs[j] x^j at x."""
    val = 0.0
    for j in range(k, degree + 1):
        factor = 1.0
        for m in range(j - k + 1, j + 1):
            factor *= m
        val += coeffs[j] * factor * (x ** (j - k))
    return float(val)


def poly_function(x: float, coeffs, degree: int) -> float:
    """Polynomial with given coefficients."""
    xx = float(x)
    return sum(coeffs[j] * xx**j for j in range(degree + 1))


def noisy_edge_sine(x: float, x0: float) -> float:
    """A noisy sine function with perturbations at the edges."""
    val = math.sin(x)
    if abs(x - x0) > 0.1:
        val += 5e-2  # strong perturbation at edges
    return val


def linear_function(x: float, slope: float = 3.5, intercept: float = -2.0) -> float:
    """Simple linear function."""
    return slope * x + intercept


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("x0", [-1.3, -0.5, 0.0, 0.7, 1.2])
def test_polynomial_exact_derivatives(degree, x0):
    """Tests that LocalPolynomialDerivative recovers exact derivatives of polynomials."""
    rng = np.random.default_rng(42 + degree)
    coeffs = rng.normal(size=degree + 1)

    f = partial(poly_function, coeffs=coeffs, degree=degree)

    config = LocalPolyConfig(
        rel_steps=(0.01, 0.02, 0.04, 0.08),
        max_degree=degree,
        min_samples=degree + 3,
        center=True,
    )

    for order in range(1, min(5, degree) + 1):
        lp = LocalPolynomialDerivative(f, float(x0), config=config)
        est, diag = lp.differentiate(order=order, degree=degree, diagnostics=True)

        truth = eval_poly_derivative(order, float(x0), coeffs, degree)
        err = _rel_err(float(est), truth)

        # With exact polynomial and sufficient degree, expect near machine precision.
        assert err < 1e-8, (
            f"poly deg={degree}, order={order}, x0={x0}: "
            f"est={est}, truth={truth}, err={err}, diag={diag}"
        )
        assert diag["ok"], f"Expected ok=True on exact polynomial fit, got diag={diag}"


@pytest.mark.parametrize(
    "name,f,df2,df3",
    [
        (
            "sin",
            np.sin,
            lambda x: -np.sin(x),  # 2nd derivative
            lambda x: -np.cos(x),  # 3rd derivative
        ),
        (
            "exp",
            np.exp,
            np.exp,
            np.exp,
        ),
    ],
)


@pytest.mark.parametrize("x0", [-2.0, -0.7, 0.0, 0.5, 1.3, 2.0])
def test_smooth_functions_orders_1_to_3(name, f, df2, df3, x0):
    """LocalPolynomialDerivative should be accurate on smooth functions."""
    config = LocalPolyConfig(
        rel_steps=(0.01, 0.02, 0.04, 0.08),
        max_degree=6,
        min_samples=8,
        center=True,
        tol_rel=1e-2,
    )

    if name == "sin":
        d1_true = math.cos(x0)
    else:  # exp
        d1_true = math.exp(x0)

    d2_true = float(df2(x0))
    d3_true = float(df3(x0))

    lp = LocalPolynomialDerivative(f, float(x0), config=config)

    d1, diag1 = lp.differentiate(order=1, diagnostics=True)
    d2, diag2 = lp.differentiate(order=2, diagnostics=True)
    d3, diag3 = lp.differentiate(order=3, diagnostics=True)

    for est, truth, diag, order in [
        (d1, d1_true, diag1, 1),
        (d2, d2_true, diag2, 2),
        (d3, d3_true, diag3, 3),
    ]:
        est = float(est)
        assert math.isfinite(est), f"{name} order {order} @ x0={x0}: non-finite est, diag={diag}"
        err = _rel_err(est, truth)
        # Baseline target: comfortably better than 1e-4
        assert err < 1e-4, (
            f"{name} order {order} @ x0={x0}: est={est}, truth={truth}, "
            f"err={err}, diag={diag}"
        )
        if diag["ok"]:
            # If we claim ok=True, be a bit stricter.
            assert err < 5e-5, (
                f"{name} order {order} @ x0={x0}: ok=True but err={err} too large, diag={diag}"
            )


def test_trimming_shaves_bad_edges():
    """Tests trimming handles bad edge samples without breaking the estimate."""
    config = LocalPolyConfig(
        rel_steps=(0.01, 0.02, 0.04, 0.08, 0.16),
        max_degree=4,
        min_samples=6,
        max_trim=10,
        center=True,
        tol_rel=5e-3,
    )

    x0 = 0.5
    f_noisy = partial(noisy_edge_sine, x0=x0)

    lp = LocalPolynomialDerivative(f_noisy, x0, config=config)
    d1, diag = lp.differentiate(order=1, diagnostics=True)

    g_true = math.cos(x0)
    err = _rel_err(float(d1), g_true)

    assert diag["ok"], f"Expected ok after trimming or stable fit; diag={diag}"
    assert err < 5e-3, f"Trimmed derivative too far from truth: err={err}, diag={diag}"


def test_center_false_matches_center_true_for_linear():
    """Tests that center=False matches center=True on a linear function."""
    x0 = 0.7

    config_center = LocalPolyConfig(
        rel_steps=(0.01, 0.02, 0.04),
        max_degree=3,
        min_samples=5,
        center=True,
    )
    config_nocenter = LocalPolyConfig(
        rel_steps=(0.01, 0.02, 0.04),
        max_degree=3,
        min_samples=5,
        center=False,
    )

    lp_c = LocalPolynomialDerivative(linear_function, x0, config=config_center)
    lp_nc = LocalPolynomialDerivative(linear_function, x0, config=config_nocenter)

    d_c, diag_c = lp_c.differentiate(order=1, diagnostics=True)
    d_nc, diag_nc = lp_nc.differentiate(order=1, diagnostics=True)

    assert _rel_err(float(d_c), 3.5) < 1e-9, f"center=True wrong: {d_c}, diag={diag_c}"
    assert _rel_err(float(d_nc), 3.5) < 1e-9, f"center=False wrong: {d_nc}, diag={diag_nc}"
    assert diag_c["ok"]
    assert diag_nc["ok"]
