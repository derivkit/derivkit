#!/usr/bin/env python3
"""
Compare derivatives from:
  • DerivativeKit.adaptive  (Chebyshev poly-fit, default behavior)
  • DerivativeKit.finite    (finite differences)
  • GaussianProcess         (your GP wrapper: func,x0 ctor; differentiate(order,...))

Run:  PYTHONPATH=src python demo_gp.py
"""

from __future__ import annotations
import math
import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.gaussian_process.gaussian_process import GaussianProcess


# set which derivative order to test (1 or 2)
TEST_ORDER = 1  # change to 2 to test second derivatives


# ---------------------------
# test functions + derivatives
# ---------------------------
def f_sin(x):           return np.sin(x)
def df_sin(x):          return np.cos(x)
def d2f_sin(x):         return -np.sin(x)

def f_expquad(x):       return np.exp(x**2)
def df_expquad(x):      return 2*x*np.exp(x**2)
def d2f_expquad(x):     return (2 + 4*x**2) * np.exp(x**2)

def f_tanh(x):          return np.tanh(x)
def df_tanh(x):         return 1.0/np.cosh(x)**2
def d2f_tanh(x):        return -2*np.tanh(x) / (np.cosh(x)**2)

def f_wiggly(x):        return np.sin(5*x) / (1 + x**2)
def df_wiggly(x):
    g  = np.sin(5*x);   gp = 5*np.cos(5*x)
    h  = 1 + x**2;      hp = 2*x
    return (gp*h - g*hp) / (h*h)
def d2f_wiggly(x):
    # derivative of df_wiggly (closed form)
    g  = np.sin(5*x);   gp = 5*np.cos(5*x);   gpp = -25*np.sin(5*x)
    h  = 1 + x**2;      hp = 2*x;             hpp = 2.0
    num = (gpp*h - g*hpp)*h*h - 2*(gp*h - g*hp)*h*hp
    den = (h*h)**2
    return num/den

CASES = [
    ("sin(x)",             f_sin,     df_sin,     d2f_sin),
    ("exp(x^2)",           f_expquad, df_expquad, d2f_expquad),
    ("tanh(x)",            f_tanh,    df_tanh,    d2f_tanh),
    ("sin(5x)/(1+x^2)",    f_wiggly,  df_wiggly,  d2f_wiggly),
]


# ---------------------------
# helpers
# ---------------------------
def build_grid(x0, width, n):
    """symmetric 1D absolute grid around x0 as shape (n,1)."""
    return (np.linspace(-width, width, n) + x0)[:, None]

def grid_stats_for_gp(X, func):
    """heuristic initial GP hyperparams from the grid + values."""
    x = X[:, 0]
    diffs = np.diff(np.unique(x))
    ell = float(np.median(diffs)) if diffs.size else 1.0
    y = np.array([func(float(xi)) for xi in x])
    amp = float(np.std(y) or 1.0)
    return ell, amp


# ---------------------------
# experiment
# ---------------------------
def compare_one(func, dfunc, d2func, x0, X):
    # ground truth
    truth = float((dfunc if TEST_ORDER == 1 else d2func)(x0))

    # --- GaussianProcess (your new API)
    ell, amp = grid_stats_for_gp(X, func)
    gp = GaussianProcess(func, x0, ell=ell, amp=amp, noise_var=1e-6, normalize=True, optimize=True)
    gp_out = gp.differentiate(order=TEST_ORDER, samples=X, return_variance=True)
    gp_dmu, gp_dvar = gp_out
    gp_err = abs(gp_dmu - truth)
    gp_sig = math.sqrt(max(float(gp_dvar), 0.0))

    # --- DerivativeKit backends
    dk = DerivativeKit(func, x0)

    # adaptive: DEFAULT behavior (Chebyshev + spacing auto, domain-aware)
    adaptive_val = dk.adaptive.differentiate(order=TEST_ORDER, diagnostics=False)
    adaptive_dmu = float(np.squeeze(adaptive_val if not isinstance(adaptive_val, tuple) else adaptive_val[0]))
    adaptive_err = abs(adaptive_dmu - truth)

    # finite: defaults
    finite_val = dk.finite.differentiate(order=TEST_ORDER)
    finite_dmu = float(np.squeeze(finite_val if not isinstance(finite_val, tuple) else finite_val[0]))
    finite_err = abs(finite_dmu - truth)

    return {
        "truth": truth,
        "gp": (gp_dmu, gp_sig, gp_err, ell, amp),
        "adaptive": (adaptive_dmu, adaptive_err),
        "finite": (finite_dmu, finite_err),
    }


def main():
    x0_list = [0.0, 0.3, 1.0]
    grids = [
        ("n=7,  width=0.4",  lambda x0: build_grid(x0, 0.4,  7)),
        ("n=15, width=0.8",  lambda x0: build_grid(x0, 0.8, 15)),
        ("n=25, width=1.0",  lambda x0: build_grid(x0, 1.0, 25)),
    ]

    hdr = "first" if TEST_ORDER == 1 else "second"
    print(f"\n=== {hdr} derivative comparison: adaptive vs finite vs gaussian_process ===\n")
    for name, f, df, d2f in CASES:
        print(f"\n--- f(x) = {name} ---")
        for x0 in x0_list:
            print(f"\n  @ x0 = {x0:.3f}")
            for gname, gfun in grids:
                X = gfun(x0)
                out = compare_one(f, df, d2f, x0, X)
                truth = out["truth"]
                gp_mu, gp_sig, gp_err, ell, amp = out["gp"]
                ad_mu, ad_err = out["adaptive"]
                fi_mu, fi_err = out["finite"]

                print(f"    grid: {gname:14s} | truth: {truth:+.8f}")
                print(f"      GP:    d≈ {gp_mu:+.8f}  | σ≈ {gp_sig:.2e}  | |Δ|≈ {gp_err:.2e}  [ell≈{ell:.3g}, amp≈{amp:.3g}]")
                print(f"   Adaptive: d≈ {ad_mu:+.8f}  | |Δ|≈ {ad_err:.2e}")
                print(f"     Finite: d≈ {fi_mu:+.8f}  | |Δ|≈ {fi_err:.2e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
