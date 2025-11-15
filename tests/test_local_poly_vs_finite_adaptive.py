"""Compare local polynomial baseline vs finite and adaptive engines."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit
from derivkit.local_polynomial_derivative.local_poly_config import (
    LocalPolyConfig,
)
from derivkit.local_polynomial_derivative.local_polynomial_derivative import (
    LocalPolynomialDerivative,
)


def _rel_err(a: float, b: float) -> float:
    d = max(1.0, abs(a), abs(b))
    return abs(a - b) / d


def _eval_derivkit(dk: DerivativeKit, method: str, order: int, **k) -> float:
    """Dispatch into DerivativeKit's methods (finite/adaptive/unified)."""
    m = method.lower()

    # legacy finite interface
    if hasattr(dk, "finite") and m.startswith("fin"):
        allowed = {"num_points", "n_workers"}
        kw = {kk: vv for kk, vv in k.items() if kk in allowed}
        return float(dk.finite.differentiate(order=order, **kw))

    # legacy adaptive interface
    if hasattr(dk, "adaptive") and m.startswith("ad"):
        allowed = {"n_workers", "tol", "rtol"}
        kw = {kk: vv for kk, vv in k.items() if kk in allowed}
        # basic tol->rtol compatibility shim
        if "tol" in kw:
            params = inspect.signature(dk.adaptive.differentiate).parameters
            if "tol" not in params and "rtol" in params:
                kw["rtol"] = kw.pop("tol")
        return float(dk.adaptive.differentiate(order=order, **kw))

    # unified dispatcher
    return float(dk.differentiate(method=method, order=order, **k))


# Each entry: (name, f, df, d2f, grid)
CASES = [
    ("sin", np.sin, np.cos, lambda x: -np.sin(x), np.linspace(-2, 2, 7)),
    ("exp", np.exp, np.exp, np.exp, np.linspace(-1.5, 2.0, 6)),
    ("log1p(x^2)",
     lambda x: np.log1p(x * x),
     lambda x: 2 * x / (1 + x * x),
     lambda x: 2 * (1 - x * x) / (1 + x * x) ** 2,
     [-2.0, -0.7, 0.0, 0.8, 2.5]),
    ("exp(0.5x)*sin(x)",
     lambda x: np.exp(0.5 * x) * np.sin(x),
     lambda x: np.exp(0.5 * x) * (np.cos(x) + 0.5 * np.sin(x)),
     lambda x: np.exp(0.5 * x) * (np.cos(x) - 0.75 * np.sin(x)),
     [-1.2, -0.3, 0.6, 1.4]),
    ("rational 1/(1+x^2)",
     lambda x: 1.0 / (1.0 + x * x),
     lambda x: -2 * x / (1.0 + x * x) ** 2,
     lambda x: (6 * x * x - 2) / (1.0 + x * x) ** 3,
     [-3.0, -1.0, -0.3, 0.3, 1.2, 2.5]),
    ("cos(x^2)",
     lambda x: np.cos(x * x),
     lambda x: -2 * x * np.sin(x * x),
     lambda x: -2 * np.sin(x * x) - 4 * x * x * np.cos(x * x),
     [-2.0, -0.5, 0.0, 0.5, 1.5]),
    ("softplus log(1+e^x)",
     lambda x: np.log1p(np.exp(x)),
     lambda x: 1.0 / (1.0 + np.exp(-x)),
     lambda x: np.exp(-x) / (1.0 + np.exp(-x)) ** 2,
     [-8.0, -2.0, 0.0, 2.0, 8.0]),
    ("softsign x/(1+|x|)",
     lambda x: x / (1.0 + abs(x)),
     # away from 0 we can treat it as differentiable
     lambda x: 1.0 / (1.0 + abs(x)) ** 2,
     lambda x: -2.0 * np.sign(x) / (1.0 + abs(x)) ** 3,
     [-3.0, -1.0, -0.2, 0.2, 1.0, 3.0]),
    ("poly 7x^5 - 3x^3 + 2x - 1",
     lambda x: 7 * x**5 - 3 * x**3 + 2 * x - 1,
     lambda x: 35 * x**4 - 9 * x**2 + 2,
     lambda x: 140 * x**3 - 18 * x,
     [-1.3, -0.6, 0.0, 0.8, 1.1]),
]

EXCLUDE_POINTS = {
    "softsign x/(1+|x|)": {0.0},
}


@pytest.mark.parametrize("name,f,df,d2f,grid", CASES, ids=[c[0] for c in CASES])
def test_local_poly_vs_finite_and_adaptive(name, f, df, d2f, grid):
    """Check local polynomial baseline against analytic and compare to other engines."""
    # Tolerances: finite & adaptive should be very sharp; local poly is baseline so slightly looser.
    rtol_finite = 5e-6
    rtol_adaptive = 5e-6
    rtol_local = 5e-4
    atol_small = 1e-12

    for x0 in grid:
        if name in EXCLUDE_POINTS and x0 in EXCLUDE_POINTS[name]:
            continue

        # Ground truth
        g_true = float(df(x0))
        h_true = float(d2f(x0))

        # --- finite difference (via DerivativeKit) ---
        dk = DerivativeKit(f, float(x0))
        g_fd = _eval_derivkit(dk, "finite", order=1, num_points=5)
        h_fd = _eval_derivkit(dk, "finite", order=2, num_points=5)

        assert _rel_err(g_fd, g_true) < rtol_finite or abs(g_fd - g_true) < atol_small, \
            f"[finite] {name} @ x0={x0}: g={g_fd}, truth={g_true}"
        assert _rel_err(h_fd, h_true) < rtol_finite or abs(h_fd - h_true) < atol_small, \
            f"[finite] {name} @ x0={x0}: h={h_fd}, truth={h_true}"

        # --- adaptive (via DerivativeKit) ---
        g_ad = _eval_derivkit(dk, "adaptive", order=1)
        h_ad = _eval_derivkit(dk, "adaptive", order=2)

        assert _rel_err(g_ad, g_true) < rtol_adaptive or abs(g_ad - g_true) < atol_small, \
            f"[adaptive] {name} @ x0={x0}: g={g_ad}, truth={g_true}"
        assert _rel_err(h_ad, h_true) < rtol_adaptive or abs(h_ad - h_true) < atol_small, \
            f"[adaptive] {name} @ x0={x0}: h={h_ad}, truth={h_true}"

        # --- local polynomial baseline (direct) ---
        # Use a config with enough degree+samples to support d2 (and higher)
        lp = LocalPolynomialDerivative(
            f,
            float(x0),
            config=LocalPolyConfig(
                # you can tune these; keep simple to start
                rel_steps=(0.01, 0.02, 0.04, 0.08),
                max_degree=6,
                min_samples=8,
                center=True,
            ),
        )

        g_lp, diag1 = lp.differentiate(order=1, diagnostics=True)
        h_lp, diag2 = lp.differentiate(order=2, diagnostics=True)

        # sanity: shouldn't blow up
        assert np.isfinite(g_lp), f"[local_poly] non-finite g @ {name}, x0={x0}"
        assert np.isfinite(h_lp), f"[local_poly] non-finite h @ {name}, x0={x0}"

        # accuracy vs analytic (looser than finite/adaptive but still strong)
        assert _rel_err(g_lp, g_true) < rtol_local or abs(g_lp - g_true) < 1e-8, \
            f"[local_poly] {name} @ x0={x0}: g={g_lp}, truth={g_true}, diag={diag1}"
        assert _rel_err(h_lp, h_true) < rtol_local or abs(h_lp - h_true) < 1e-8, \
            f"[local_poly] {name} @ x0={x0}: h={h_lp}, truth={h_true}, diag={diag2}"

        # Optional: ensure when local_poly marks ok, it really is good
        if diag1["ok"]:
            assert _rel_err(g_lp, g_true) < 5e-4
        if diag2["ok"]:
            assert _rel_err(h_lp, h_true) < 5e-4
