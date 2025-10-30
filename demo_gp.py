#!/usr/bin/env python3
"""Compare derivatives from:
  • DerivativeKit(method="adaptive")
  • DerivativeKit(method="finite")
  • DerivativeKit(method="gp")  # your GP backend

Run:  PYTHONPATH=src python demo_gp.py
"""

from __future__ import annotations

import math

import numpy as np

from derivkit.derivative_kit import DerivativeKit

# Ensure GP engine is registered (import side-effect):
from derivkit.gaussian_process.gp_engine import GPDerivative  # noqa: F401

# ---- config ----
TEST_ORDER = 1  # set to 2 for second derivatives


# ---- test functions ----
def f_sin(x):   return np.sin(x)
def df_sin(x):  return np.cos(x)
def d2f_sin(x): return -np.sin(x)

def f_expquad(x):   return np.exp(x**2)
def df_expquad(x):  return 2*x*np.exp(x**2)
def d2f_expquad(x): return (2 + 4*x**2) * np.exp(x**2)

def f_tanh(x):   return np.tanh(x)
def df_tanh(x):  return 1.0/np.cosh(x)**2
def d2f_tanh(x): return -2*np.tanh(x) / (np.cosh(x)**2)

def f_wiggly(x): return np.sin(5*x) / (1 + x**2)
def df_wiggly(x):
    g, gp = np.sin(5*x), 5*np.cos(5*x)
    h, hp = 1 + x**2, 2*x
    return (gp*h - g*hp) / (h*h)
def d2f_wiggly(x):
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


# ---- helpers ----
def build_grid(x0: float, width: float, n: int) -> np.ndarray:
    """Symmetric 1D absolute grid around x0 as shape (n,1)."""
    return (np.linspace(-width, width, n) + x0)[:, None]

def grid_stats_for_gp(X: np.ndarray, func) -> tuple[float, float]:
    """Heuristic GP seeds based on grid SPAN (safer than step)."""
    x = np.asarray(X)[:, 0]
    span = float(x.max() - x.min()) or 1.0
    ell = 0.5 * span
    y = np.array([func(float(xi)) for xi in x], dtype=float)
    amp = float(np.std(y) or 1.0)
    return ell, amp


# ---- experiment ----
def compare_one(func, dfunc, d2func, x0: float, X: np.ndarray):
    truth = float((dfunc if TEST_ORDER == 1 else d2func)(x0))
    dk = DerivativeKit(func, x0)

    # GP
    length_scale, output_scale = grid_stats_for_gp(X, func)
    gp_mu, gp_var = dk.differentiate(
        method="gp",
        order=TEST_ORDER,
        kernel="rbf",
        kernel_params={"length_scale": length_scale, "output_scale": output_scale},
        samples=X,
        normalize=True,
        optimize=True,  # <— optional but helps on wiggly
        local_frac_span=0.20,  # <— NEW: tighter local window for high curvature
        return_variance=True,
    )

    gp_err = abs(gp_mu - truth)
    gp_sig = math.sqrt(max(float(gp_var), 0.0))

    # Adaptive
    ad_mu = float(dk.differentiate(method="adaptive", order=TEST_ORDER))
    ad_err = abs(ad_mu - truth)

    # Finite
    fi_mu = float(dk.differentiate(method="finite", order=TEST_ORDER))
    fi_err = abs(fi_mu - truth)

    return truth, (gp_mu, gp_sig, gp_err, length_scale, output_scale), (ad_mu, ad_err), (fi_mu, fi_err)


def main():
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
                X = gfun(x0)
                truth, gp, ad, fi = compare_one(f, df, d2f, x0, X)
                gp_mu, gp_sig, gp_err, ls, os_ = gp
                ad_mu, ad_err = ad
                fi_mu, fi_err = fi

                print(f"    grid: {gname:14s} | truth: {truth:+.8f}")
                print(f"      GP:    d≈ {gp_mu:+.8f}  | |Δ|≈ {gp_err:.2e} | σ≈ {gp_sig:.2e}  | [length_scale≈{ls:.3g}, output_scale≈{os_:.3g}]")
                print(f"   Adaptive: d≈ {ad_mu:+.8f}  | |Δ|≈ {ad_err:.2e}")
                print(f"     Finite: d≈ {fi_mu:+.8f}  | |Δ|≈ {fi_err:.2e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
