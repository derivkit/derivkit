"""Demo for comparing GP-based derivatives to adaptive and finite difference methods.

Compare derivatives from:
  • DerivativeKit(method="adaptive")
  • DerivativeKit(method="finite")
  • DerivativeKit(method="gp")  # your GP backend

Run:  PYTHONPATH=src python demo_gp.py
"""

from __future__ import annotations

import math

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.gaussian_process.gp_diagnostics import format_gp_diagnostics

# ---- config ----
TEST_ORDER = 1  # set to 2 for second derivatives


def f_sin(x):
    """Simple sine function and derivatives."""
    return np.sin(x)


def df_sin(x):
    """First derivative of sine."""
    return np.cos(x)


def d2f_sin(x):
    """Second derivative of sine."""
    return -np.sin(x)


def f_expquad(x):
    """Exponential quadratic function and derivatives."""
    return np.exp(x**2)


def df_expquad(x):
    """First derivative of exponential quadratic."""
    return 2*x*np.exp(x**2)


def d2f_expquad(x):
    """Second derivative of exponential quadratic."""
    return (2 + 4*x**2) * np.exp(x**2)


def f_tanh(x):
    """Hyperbolic tangent function and derivatives."""
    return np.tanh(x)


def df_tanh(x):
    """First derivative of hyperbolic tangent."""
    return 1.0/np.cosh(x)**2


def d2f_tanh(x):
    """Second derivative of hyperbolic tangent."""
    return -2*np.tanh(x) / (np.cosh(x)**2)


def f_wiggly(x):
    """Wiggly function and derivatives."""
    return np.sin(5*x) / (1 + x**2)


def df_wiggly(x):
    """First derivative of wiggly function."""
    g, gp = np.sin(5*x), 5*np.cos(5*x)
    h, hp = 1 + x**2, 2*x
    return (gp*h - g*hp) / (h*h)


def d2f_wiggly(x):
    """Second derivative of wiggly function."""
    g, gp, gpp = np.sin(5*x), 5*np.cos(5*x), -25*np.sin(5*x)
    h, hp, hpp = 1 + x**2, 2*x, 2.0
    num = (gpp*h - g*hpp)*h*h - 2*(gp*h - g*hp)*h*hp
    den = (h*h)**2
    return num/den


CASES = [
    ("sin(x)", f_sin, df_sin, d2f_sin),
    ("exp(x^2)", f_expquad, df_expquad, d2f_expquad),
    ("tanh(x)", f_tanh, df_tanh, d2f_tanh),
    ("sin(5x)/(1+x^2)", f_wiggly, df_wiggly, d2f_wiggly),
]


def build_grid(x0: float, width: float, n: int) -> np.ndarray:
    """Symmetric 1D absolute grid around x0 as shape (n,1)."""
    return (np.linspace(-width, width, n) + x0)[:, None]

def grid_stats_for_gp(x: np.ndarray, func) -> tuple[float, float]:
    """Heuristic GP seeds based on grid SPAN (safer than step)."""
    x_arr = np.asarray(x)[:, 0]
    span = float(x_arr.max() - x_arr.min()) or 1.0
    ell = 0.5 * span
    y = np.array([func(float(xi)) for xi in x_arr], dtype=float)
    amp = float(np.std(y) or 1.0)
    return ell, amp


def compare_one(func, dfunc, d2func, x0: float, x: np.ndarray):
    """Compares derivative estimates at x0 on grid X for one function case."""
    truth = float((dfunc if TEST_ORDER == 1 else d2func)(x0))
    dk = DerivativeKit(func, x0)

    # GP
    length_scale, output_scale = grid_stats_for_gp(x, func)
    gp_mu, gp_var, gp_diag = dk.differentiate(
        method="gp",
        order=TEST_ORDER,
        kernel="rbf",
        kernel_params={"length_scale": length_scale, "output_scale": output_scale},
        samples=x,
        normalize=True,
        optimize=True,
        local_frac_span=0.20,
        return_variance=True,
        return_diag=True,
    )

    gp_err = abs(gp_mu - truth)
    gp_sig = math.sqrt(max(float(gp_var), 0.0))

    # Adaptive
    ad_mu = float(dk.differentiate(method="adaptive", order=TEST_ORDER))
    ad_err = abs(ad_mu - truth)

    # Finite
    fi_mu = float(dk.differentiate(method="finite", order=TEST_ORDER))
    fi_err = abs(fi_mu - truth)

    return truth, (gp_mu, gp_sig, gp_err, length_scale, output_scale, gp_diag), (ad_mu, ad_err), (fi_mu, fi_err)


def main():
    """Run the GP vs adaptive vs finite derivative comparison demo."""
    x0_list = [0.0, 0.3, 1.0]
    grids = [
        ("n=7,  width=0.4",  lambda x0: build_grid(x0, 0.4,  7)),
        ("n=15, width=0.8",  lambda x0: build_grid(x0, 0.8, 15)),
        ("n=25, width=1.0",  lambda x0: build_grid(x0, 1.0, 25)),
    ]

    hdr = "first" if TEST_ORDER == 1 else "second"
    print(f"\n=== {hdr} derivative comparison: adaptive vs finite vs gp ===\n")
    for name, f, df, d2f in CASES:
        print(f"\n--- f(x) = {name} ---")
        for x0 in x0_list:
            print(f"\n  @ x0 = {x0:.3f}")
            for gname, gfun in grids:
                x = gfun(x0)
                truth, gp, ad, fi = compare_one(f, df, d2f, x0, x)
                gp_mu, gp_sig, gp_err, ls, os_, gp_diag = gp
                ad_mu, ad_err = ad
                fi_mu, fi_err = fi

                print(".......................................................................................")
                print(f"   Adaptive: d≈ {ad_mu:+.8f}  | |Δ|≈ {ad_err:.2e}")
                print(f"     Finite: d≈ {fi_mu:+.8f}  | |Δ|≈ {fi_err:.2e}")
                print(
                    f"      GP:    d≈ {gp_mu:+.8f}  | |Δ|≈ {gp_err:.2e} | σ≈ {gp_sig:.2e}  | [length_scale≈{ls:.3g}, output_scale≈{os_:.3g}]")
                print(f"    grid: {gname:14s} | truth: {truth:+.8f}")
                print(".......................................................................................")
                print("      diag:")
                print(format_gp_diagnostics(gp_diag, decimals=4, max_rows=8))

    print("\nDone.\n")

if __name__ == "__main__":
    main()
