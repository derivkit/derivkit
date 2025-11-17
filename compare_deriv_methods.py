"""Quick comparison of derivative methods in DerivativeKit.

Run with:
    python -m scripts.compare_derivative_methods

or (if placed at repo root and made executable):
    python scripts/compare_derivative_methods.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def rel_err(a: float, b: float) -> float:
    """Relative error with a safe denominator."""
    d = max(1.0, abs(a), abs(b))
    return abs(a - b) / d


rng = np.random.default_rng(12345)

def noisy_sin(x: float, noise_level: float = 1e-4) -> float:
    return np.sin(x) + noise_level * rng.normal()

def d_noisy_sin(x: float) -> float:
    return np.cos(x)


def main() -> None:
    """Main comparison routine."""
    # Functions to test: name -> (f, analytic df, order)
    cases: list[dict[str, Any]] = [
        {
            "name": "sin",
            "order": 1,
            "f": np.sin,
            "df": np.cos,
            "x0_grid": [0.1, 0.7, 1.3],
        },
        {
            "name": "exp",
            "order": 1,
            "f": np.exp,
            "df": np.exp,
            "x0_grid": [-0.5, 0.0, 1.0],
        },
        {
            "name": "poly x^4",
            "order": 3,
            "f": lambda x: x ** 4,
            "df": lambda x: 24 * x,  # 3rd derivative of x^4
            "x0_grid": [0.2, 0.7],
        },
        # evil / delicate functions
        {
            "name": "high-freq sin(30x)",
            "order": 1,
            "f": lambda x: np.sin(30.0 * x),
            "df": lambda x: 30.0 * np.cos(30.0 * x),
            # avoid points where cos(30x) ~ 0 so denom in rel_err isn't tiny
            "x0_grid": [0.03, 0.11, 0.21],
        },
        {
            "name": "gaussian * sin(10x)",
            "order": 1,
            "f": lambda x: np.exp(-x ** 2) * np.sin(10.0 * x),
            # d/dx [e^{-x^2} sin(10x)] = e^{-x^2} * (10 cos(10x) - 2x sin(10x))
            "df": lambda x: np.exp(-x ** 2) * (
                    10.0 * np.cos(10.0 * x) - 2.0 * x * np.sin(10.0 * x)
            ),
            "x0_grid": [-0.7, -0.2, 0.4],
        },
        {
            "name": "Runge 1 / (1 + 25x^2)",
            "order": 1,
            "f": lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
            # d/dx [1/(1+25x^2)] = -50x / (1+25x^2)^2
            "df": lambda x: -50.0 * x / (1.0 + 25.0 * x ** 2) ** 2,
            "x0_grid": [-0.9, -0.5, 0.5, 0.9],
        },
        {
            "name": "soft-abs sqrt(x^2 + eps)",
            "order": 1,
            "f": lambda x, eps=1e-4: np.sqrt(x ** 2 + eps),
            # d/dx sqrt(x^2 + eps) = x / sqrt(x^2 + eps)
            "df": lambda x, eps=1e-4: x / np.sqrt(x ** 2 + eps),
            "x0_grid": [-0.2, -0.02, 0.02, 0.2],
        },
        {
            "name": "noisy sin (σ=1e-4)",
            "order": 1,
            "f": noisy_sin,
            "df": d_noisy_sin,
            "x0_grid": [0.3, 0.9],
        },
    ]

    # method name + kwargs fed into DerivativeKit.differentiate
    method_configs: list[tuple[str, dict[str, Any]]] = [
        # finite, no extrapolation
        (
            "finite",
            {
                "stepsize": 0.1,
                "num_points": 5,
                "extrapolation": None,
            },
        ),
        # finite + Richardson (fixed)
        (
            "finite",
            {
                "stepsize": 0.1,
                "num_points": 5,
                "extrapolation": "richardson",
                "levels": 3,
            },
        ),
        # finite + Ridders (fixed)
        (
            "finite",
            {
                "stepsize": 0.1,
                "num_points": 5,
                "extrapolation": "ridders",
                "levels": 3,
            },
        ),
        # finite + Gauss–Richardson (fixed)
        (
            "finite",
            {
                "stepsize": 0.1,
                "num_points": 5,
                "extrapolation": "gauss-richardson",
                "levels": 3,
            },
        ),
        # local polynomial (baseline)
        (
            "local_polynomial",
            {
                "degree": 5,
            },
        ),
        # adaptive global engine (uses its defaults)
        (
            "adaptive",
            {},
        ),
    ]

    line = "-" * 80

    for case in cases:
        name = case["name"]
        order = case["order"]
        f = case["f"]
        df = case["df"]
        x0_grid = case["x0_grid"]

        print(line)
        print(f"Function: {name!r}, derivative order: {order}")
        print(line)

        for x0 in x0_grid:
            dk = DerivativeKit(f, float(x0))
            truth = float(df(x0))
            print(f"\nx0 = {x0:.6g}, analytic = {truth:.12g}")
            print("  {:>18s}  {:>18s}  {:>18s}".format("method", "estimate", "rel_err"))
            print("  " + "-" * 60)

            for method, kw in method_configs:
                est = dk.differentiate(method=method, order=order, **kw)
                est_f = float(est)
                err = rel_err(est_f, truth)
                label = method
                if method == "finite" and kw.get("extrapolation") is not None:
                    label = f"finite+{kw['extrapolation']}"
                print(
                    f"  {label:>18s}  {est_f:18.10e}  {err:18.10e}"
                )

        print()  # blank line between functions


if __name__ == "__main__":
    main()
