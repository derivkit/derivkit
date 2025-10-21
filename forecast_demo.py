#!/usr/bin/env python3
"""
1D exact vs Fisher vs doublet-DALI on a deliberately nonlinear toy model.

Model:
    observable o(x) = 100 * exp(x^2)

Exact log-like:
    logL_exact(x) = -0.5 * [(o(x) - o(x0)) / sigma_o]^2

Approx log-like (your conventions):
    If tensors = [F]                    -> logL ≈ -0.5 * F * dx^2
    If tensors = [F, G, H] (doublet)   -> logL ≈ -0.5 * F * dx^2
                                          -0.5 * G * dx^3
                                          -0.125 * H * dx^4
    (i.e. your G and H already include any combinatorics you prefer)

We compute tensors via ForecastKit and plot the three curves
over a wide grid to clearly see the differences.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from derivkit.forecast_kit import ForecastKit  # adjust import if needed


# --------------------------
# Your model & likelihoods
# --------------------------

def testModel1d(paramList):
    """One observable with a quadratic dependence on one parameter."""
    x = float(paramList[0])
    obs = 1e2 * np.exp(0.2*x**2)
    return np.array([obs], dtype=float)


def logLike1dExact(sigma_o: float, fiducial_x: float, x: float) -> float:
    """Exact Gaussian log-likelihood for one observable with variance sigma_o^2."""
    delta_o = testModel1d([x]) - testModel1d([fiducial_x])
    return float(-0.5 * (delta_o[0] / sigma_o) ** 2)


def logLike1dApprox(tensors: list[np.ndarray], fiducial_x: float, x: float) -> float:
    """Approximate log-likelihood using your Fisher + doublet-DALI terms.

    Conventions (matching your snippet):
      logL ≈ -0.5 * F * dx^2
             -0.5 * G * dx^3
             -0.125 * H * dx^4
    where F is (1,1), G is (1,1,1), H is (1,1,1,1) in 1D.
    """
    dx = x - fiducial_x
    logLike = 0.0
    if len(tensors) >= 1:
        F = tensors[0]
        logLike += -0.5 * float(F[0][0]) * dx**2
    if len(tensors) >= 3:
        G = tensors[1]
        H = tensors[2]
        logLike += -0.5   * float(G[0][0][0])     * dx**3
        logLike += -0.125 * float(H[0][0][0][0])  * dx**4
    return float(logLike)


def main():
    # --------------------------
    # Experiment knobs
    # --------------------------
    fiducial_values = np.array([0.1])     # x0 (keep off 0 so cubic ≠ 0)
    fiducial_x = float(fiducial_values[0])
    sigma_o = 0.05                        # 1σ of the single observable
    covmat = np.array([[sigma_o**2]])     # (1,1)

    # Wide grid to “see” differences well
    xgrid = np.linspace(-1.0, 1.0, 1000)

    # --------------------------
    # ForecastKit: Fisher & DALI
    # --------------------------
    forecaster = ForecastKit(function=lambda t: testModel1d([t[0]]),
                             theta0=fiducial_values,
                             cov=covmat)

    fisher_matrix = forecaster.fisher()     # shape (1,1)
    DALI_G, DALI_H = forecaster.dali()      # shapes (1,1,1), (1,1,1,1)

    # Quick diagnostics
    print("[tensors]")
    print("  F[0,0]     =", float(np.asarray(fisher_matrix)[0,0]))
    print("  G[0,0,0]   =", float(np.asarray(DALI_G)[0,0,0]))
    print("  H[0,0,0,0] =", float(np.asarray(DALI_H)[0,0,0,0]))

    tensors_doublet = [fisher_matrix, DALI_G, DALI_H]

    # --------------------------
    # Evaluate likelihoods
    # --------------------------
    exactLike       = np.array([logLike1dExact(sigma_o, fiducial_x, x) for x in xgrid])
    fisherLike      = np.array([logLike1dApprox([fisher_matrix], fiducial_x, x) for x in xgrid])
    doubletDALILike = np.array([logLike1dApprox(tensors_doublet, fiducial_x, x) for x in xgrid])

    # --------------------------
    # Plot
    # --------------------------
    plt.figure(figsize=(9.5, 5.5))
    plt.plot(xgrid, exactLike,       label="Exact Likelihood",        color="C0", linewidth=2.5)
    plt.plot(xgrid, fisherLike,      label="Fisher (quadratic)",      color="C2", linewidth=2.5, linestyle="-")
    plt.plot(xgrid, doubletDALILike, label="Doublet DALI (F,G,H)",    color="C1", linewidth=2.5, linestyle="--")

    plt.title(r"$o(x)=100\,e^{x^2}$,  $x_0=0.1$,  $\sigma_o=0.05$", fontsize=13)
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$\log \mathcal{L}$  (up to const.)", fontsize=12)

    # Nice view window (wide enough to show differences)
    # You can tweak limits if needed.
    y_min = 1.1 * min(exactLike.min(), fisherLike.min(), doubletDALILike.min())
    plt.xlim(xgrid.min(), xgrid.max())
    plt.ylim(y_min, 0.05)

    plt.legend(fontsize=11, frameon=True)
    plt.grid(alpha=0.25, ls=":")
    outdir = Path("plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "dali_1d_wide_grid.png"
    #plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")

    # Sanity check: all curves should meet at x0
    print("\nSanity at fiducial x0:")
    print("  exact   :", logLike1dExact(sigma_o, fiducial_x, fiducial_x))
    print("  Fisher  :", logLike1dApprox([fisher_matrix], fiducial_x, fiducial_x))
    print("  DALI    :", logLike1dApprox(tensors_doublet, fiducial_x, fiducial_x))

    # Comment on expected behaviour
    print("\nWhat to look for:")
    print("  • Fisher (quadratic) is symmetric around x0 (parabola in x - x0).")
    print("  • DALI includes cubic and quartic; with x0=0.1 the cubic term is nonzero.")
    print("    → DALI should skew away from Fisher and track the exact curve better")
    print("      as you move away from x0, especially where nonlinearity kicks in.")
    print("  • If you move x0 to 0.0, the model is symmetric → G≈0 and DALI≈Fisher.")

if __name__ == "__main__":
    main()
