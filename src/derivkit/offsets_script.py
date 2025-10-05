# test_offsets_demo.py
from __future__ import annotations
import numpy as np

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative

def f(x: float) -> np.ndarray:
    # 2-component output: [x^3, sin(x)]
    return np.array([x**3, np.sin(x)], dtype=float)

def analytic_deriv(x0: float) -> np.ndarray:
    # d/dx [x^3, sin x] = [3x^2, cos x]
    return np.array([3.0 * x0**2, np.cos(x0)], dtype=float)

def small_x_demo():
    x0 = 0.004

    # ultra-tiny absolute step (forces absolute mode)
    run_case("SMALL-x ABS step", x0=x0, step_abs=1e-6)   # 1 µstep

    # ultra-tiny relative step: 0.1% of |x0| -> 4e-6
    run_case("SMALL-x REL step", x0=x0, step_rel="0.1%")

    # explicit tiny seeds around x0 (keeps your mirror/extend behavior)
    seeds = np.array([1e-7, 2e-7, 5e-7], dtype=float)
    run_case("SMALL-x EXPLICIT seeds", x0=x0, positive_seeds=seeds)

def run_case(label: str, *, x0: float, **kwargs):
    print("\n" + "="*72)
    print(f"{label} | x0={x0} | kwargs={kwargs}")
    afd = AdaptiveFitDerivative(f, x0=x0)
    est, diag = afd.differentiate(order=1, min_samples=7, include_zero=True,
                                  acceptance="balanced", diagnostics=True, **kwargs)
    # Grid + offsets
    x_all = np.asarray(diag["x_all"], float)
    offsets = x_all - x0
    print("Grid x values:\n ", np.array2string(x_all, precision=8, floatmode="fixed"))
    print("Relative offsets (x - x0):\n ", np.array2string(offsets, precision=8, floatmode="fixed"))

    # Derivative sanity
    truth = analytic_deriv(x0)
    err = est - truth
    rel = np.where(np.abs(truth) > 0, np.abs(err) / np.abs(truth), np.abs(err))
    print("Estimated derivative:", est)
    print("Analytic   derivative:", truth)
    print("Abs error:", err)
    print("Rel error:", rel)

def main():
    x0 = 10.0
    run_case("RELATIVE step", x0=x0, step_rel="1%")
    run_case("ABSOLUTE step", x0=x0, step_abs=1e-3)
    seeds = np.array([1e-4, 2e-4, 5e-4], dtype=float)
    run_case("EXPLICIT seeds", x0=x0, positive_seeds=seeds)

    # ---- NEW: tiny-offset tests at small x0 ----
    small_x_demo()


if __name__ == "__main__":
    main()
