"""Accuracy tests vs analytic derivatives across friendly and tricky functions."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def _rel_err(a: float, b: float) -> float:
    """Computes relative error between a and b."""
    d = max(1.0, abs(a), abs(b))
    return abs(a - b) / d


def _eval(dk: DerivativeKit, method: str, order: int, **k) -> float:
    """Calls the appropriate differentiation engine based on method name."""
    m = method.lower()
    # legacy engines available?
    if hasattr(dk, "finite") and m.startswith("fin"):
        allowed = {
            "num_points",
            "n_workers",
            "extrapolation",
            "levels",
            "stepsize",
        }
        kw = {k_: v for k_, v in k.items() if k_ in allowed}
        return float(dk.finite.differentiate(order=order, **kw))
    if hasattr(dk, "adaptive") and m.startswith("ad"):
        # Pass only kwargs that typical adaptive implementations use; omit unknowns
        allowed = {"n_workers", "tol", "rtol"}
        kw = {k_: v for k_, v in k.items() if k_ in allowed}
        # map tol->rtol if engine wants rtol only (best-effort)
        if "tol" in kw and hasattr(dk.adaptive.differentiate, "__call__"):
            params = inspect.signature(dk.adaptive.differentiate).parameters
            if "tol" not in params and "rtol" in params:
                kw["rtol"] = kw.pop("tol")
        return float(dk.adaptive.differentiate(order=order, **kw))
    # unified dispatcher
    return float(dk.differentiate(method=method, order=order, **k))


# Each entry: (name, f, df, d2f, grid, notes)
CASES = [
    # Smooth, easy
    ("sin", np.sin, np.cos, lambda x: -np.sin(x), np.linspace(-2, 2, 7)),
    ("exp", np.exp, np.exp, np.exp, np.linspace(-1.5, 2.0, 6)),
    ("log1p(x^2)",
     lambda x: np.log1p(x*x),
     lambda x: 2*x/(1+x*x),
     lambda x: 2*(1 - x*x)/(1 + x*x)**2,
     [-2.0, -0.7, 0.0, 0.8, 2.5]),
    ("exp(0.5x)*sin(x)",
     lambda x: np.exp(0.5*x) * np.sin(x),
     lambda x: np.exp(0.5*x) * (np.cos(x) + 0.5*np.sin(x)),
     lambda x: np.exp(0.5*x) * (np.cos(x) - 0.75*np.sin(x)),
     [-1.2, -0.3, 0.6, 1.4]),
    # Rational (well-behaved away from poles)
    ("rational 1/(1+x^2)",
     lambda x: 1.0/(1.0 + x*x),
     lambda x: -2*x/(1.0 + x*x)**2,
     lambda x: (6*x*x - 2)/(1.0 + x*x)**3,
     [-3.0, -1.0, -0.3, 0.3, 1.2, 2.5]),
    # Oscillatory + moderate growth
    ("cos(x^2)",
     lambda x: np.cos(x*x),
     lambda x: -2*x*np.sin(x*x),
     lambda x: -2*np.sin(x*x) - 4*x*x*np.cos(x*x),
     [-2.0, -0.5, 0.0, 0.5, 1.5]),
    # "Weird" but smooth: softplus and softsign
    ("softplus log(1+e^x)",
     lambda x: np.log1p(np.exp(x)),
     lambda x: 1.0/(1.0 + np.exp(-x)),
     lambda x: np.exp(-x)/(1.0 + np.exp(-x))**2,
     [-8.0, -2.0, 0.0, 2.0, 8.0]),
    ("softsign x/(1+|x|)",
     lambda x: x/(1.0 + abs(x)),
     # Not differentiable exactly at 0; we avoid x=0 in grid
     lambda x: 1.0/(1.0 + abs(x))**2,
     lambda x: -2.0*np.sign(x)/(1.0 + abs(x))**3,
     [-3.0, -1.0, -0.2, 0.2, 1.0, 3.0]),
    # Polynomial with large coefficients (scale)
    ("poly 7x^5 - 3x^3 + 2x - 1",
     lambda x: 7*x**5 - 3*x**3 + 2*x - 1,
     lambda x: 35*x**4 - 9*x**2 + 2,
     lambda x: 140*x**3 - 18*x,
     [-1.3, -0.6, 0.0, 0.8, 1.1]),
]

# Points to explicitly avoid (true non-differentiable locations)
EXCLUDE_POINTS = {
    "softsign x/(1+|x|)": {0.0},
}


@pytest.mark.parametrize("method,kwargs", [
    ("finite",   {"num_points": 5}),  # five-point stencil
    ("finite",   {"num_points": 5, "extrapolation": "richardson", "levels": 3}),
    ("finite",   {"num_points": 5, "extrapolation": "ridders", "levels": 3}),
    ("finite",   {"num_points": 5, "extrapolation": "gauss-richardson", "levels": 3}),
    ("adaptive", {}),  # rely on defaults across branches
])


@pytest.mark.parametrize("name,f,df,d2f,grid", CASES, ids=[c[0] for c in CASES])
def test_first_and_second_derivatives(name, f, df, d2f, grid, method, kwargs):
    """Validates first and second derivatives against analytic forms."""
    # tolerances chosen to be tight but stable across branches/step logic
    rtol1, rtol2 = 5e-6, 5e-6
    atol_small = 1e-12

    for x0 in grid:
        if name in EXCLUDE_POINTS and x0 in EXCLUDE_POINTS[name]:
            continue  # skip true kinks

        dk = DerivativeKit(f, float(x0))

        # First derivative
        g = _eval(dk, method, order=1, **kwargs)
        g_true = float(df(x0))
        assert _rel_err(g, g_true) < rtol1 or abs(g - g_true) < atol_small, \
            f"{name} @ x0={x0}: g={g}, truth={g_true} (method={method})"

        # Second derivative
        h = _eval(dk, method, order=2, **kwargs)
        h_true = float(d2f(x0))
        assert _rel_err(h, h_true) < rtol2 or abs(h - h_true) < atol_small, \
            f"{name} @ x0={x0}: h={h}, truth={h_true} (method={method})"


def test_known_edge_behavior_abs():
    """Tests derivative of abs(x), which is non-differentiable at 0."""
    f = abs

    def df_left(x: float) -> float:
        return -1.0
    def df_right(x: float) -> float:
        return +1.0

    # Away from 0 behaves fine
    for x0 in (-1.0, -0.2, 0.2, 2.0):
        dk = DerivativeKit(f, x0)
        g_fd = _eval(dk, "finite", 1, num_points=5)
        g_ad = _eval(dk, "adaptive", 1)
        truth = df_left(x0) if x0 < 0 else df_right(x0)
        assert _rel_err(g_fd, truth) < 5e-4
        assert _rel_err(g_ad, truth) < 5e-4

    # At 0, don't assert equality—just ensure it doesn't blow up / NaN
    dk0 = DerivativeKit(f, 0.0)
    g0_fd = _eval(dk0, "finite", 1, num_points=5)
    g0_ad = _eval(dk0, "adaptive", 1)
    assert math.isfinite(g0_fd)
    assert math.isfinite(g0_ad)
