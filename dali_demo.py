#!/usr/bin/env python3
"""
Piecewise DALI demo (1D):
- Model:      o(x) = 100 * exp(x^2)
- Exact logL: -0.5 * [(o(x)-o(x0))/sigma_o]^2
- Approximations:
    * Single-center Fisher (quadratic)
    * Single-center DALI (quad+cubic+quartic)
    * Piecewise DALI stitched from multiple centers (nearest-center rule)

Why piecewise?
DALI is a local Taylor expansion; accuracy fades far from the expansion point.
Stitching multiple local expansions (e.g., at x0-δ, x0, x0+δ) extends the region
where the approximation tracks the exact log-likelihood well.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from derivkit.forecast_kit import ForecastKit  # adjust if needed


# --------------------------
# Model & likelihoods
# --------------------------

def testModel1d(paramList):
    """One observable with a quadratic dependence on one parameter."""
    x = float(paramList[0])
    obs = 1e2 * np.exp(x**2)
    return np.array([obs], dtype=float)


def logLike1dExact(sigma_o: float, fiducial_x: float, x: float) -> float:
    """Exact Gaussian log-likelihood for one observable with variance sigma_o^2."""
    delta_o = testModel1d([x]) - testModel1d([fiducial_x])
    return float(-0.5 * (delta_o[0] / sigma_o) ** 2)


def logLike1dApprox(tensors: list[np.ndarray], fiducial_x: float, x: float) -> float:
    """Approx log-likelihood using your Fisher + doublet-DALI terms.
       Conventions (matching your snippet):
         logL ≈ -0.5 * F * dx^2
                -0.5 * G * dx^3
                -0.125 * H * dx^4
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


# --------------------------
# Piecewise wrapper
# --------------------------

class LocalExpansion:
    """Container for one local DALI expansion around center x_c."""
    def __init__(self, x_c: float, F: np.ndarray, G: np.ndarray, H: np.ndarray):
        self.x_c = float(x_c)
        self.F, self.G, self.H = F, G, H

    def logL(self, x: float) -> float:
        return logLike1dApprox([self.F, self.G, self.H], self.x_c, x)


def build_local_expansion(x_c: float, sigma_o: float) -> LocalExpansion:
    """Build ForecastKit tensors at center x_c and return a LocalExpansion."""
    cov = np.array([[sigma_o**2]], dtype=float)
    fk = ForecastKit(function=lambda t: testModel1d([t[0]]),
                     theta0=np.array([x_c]),
                     cov=cov)
    F = np.asarray(fk.fisher(), float).reshape(1, 1)
    G, H = fk.dali()
    G = np.asarray(G, float).reshape(1, 1, 1)
    H = np.asarray(H, float).reshape(1, 1, 1, 1)
    return LocalExpansion(x_c, F, G, H)


def piecewise_dali(x: np.ndarray, locals: list[LocalExpansion]) -> np.ndarray:
    """Nearest-center piecewise DALI: pick the local expansion with smallest |x - x_c|."""
    centers = np.array([le.x_c for le in locals])
    idx = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    out = np.empty_like(x, dtype=float)
    for i, le in enumerate(locals):
        mask = (idx == i)
        if np.any(mask):
            out[mask] = np.array([le.logL(xj) for xj in x[mask]])
    return out


# --------------------------
# Demo
# --------------------------

def main():
    # --- knobs ---
    x0       = 0.10     # central fiducial
    sigma_o  = 0.05     # observable stdev
    delta    = 0.25     # spacing between piecewise centers
    n_side   = 1        # how many centers on each side of x0 (1 → three centers total)
    # (Try n_side=2 to use five centers: x0-2δ, x0-δ, x0, x0+δ, x0+2δ)

    # wide plotting grid
    xgrid = np.linspace(-1.0, 1.0, 1201)

    # exact log-likelihood is defined with "fiducial_x" as the reference point.
    # We'll compare all approximations to the exact likelihood referenced to x0.
    exact_like = np.array([logLike1dExact(sigma_o, x0, x) for x in xgrid])

    # Single-center (at x0) Fisher & DALI for comparison
    cov = np.array([[sigma_o**2]], dtype=float)
    fk0 = ForecastKit(function=lambda t: testModel1d([t[0]]),
                      theta0=np.array([x0]),
                      cov=cov)
    F0 = np.asarray(fk0.fisher(), float).reshape(1, 1)
    G0, H0 = fk0.dali()
    G0 = np.asarray(G0, float).reshape(1, 1, 1)
    H0 = np.asarray(H0, float).reshape(1, 1, 1, 1)

    fisher_single = np.array([logLike1dApprox([F0], x0, x) for x in xgrid])
    dali_single   = np.array([logLike1dApprox([F0, G0, H0], x0, x) for x in xgrid])

    # Build piecewise centers and local expansions
    centers = [x0 + k * delta for k in range(-n_side, n_side + 1)]
    locals_ = [build_local_expansion(xc, sigma_o) for xc in centers]

    # Piecewise DALI stitched by nearest-center rule
    dali_piece = piecewise_dali(xgrid, locals_)

    # ---- Plot ----
    plt.figure(figsize=(10.5, 6.0))
    plt.plot(xgrid, exact_like,    label="Exact",               color="C0", lw=2.3)
    plt.plot(xgrid, fisher_single, label="Fisher @ x0",         color="C2", lw=2.0, ls="-")
    plt.plot(xgrid, dali_single,   label="DALI @ x0",           color="C1", lw=2.0, ls="--")
    plt.plot(xgrid, dali_piece,    label="Piecewise DALI (nearest center)", color="C3", lw=2.4)

    # mark the local expansion centers
    for xc in centers:
        plt.axvline(xc, color="0.8", lw=0.8, ls=":")
        plt.text(xc, 0.02, f"x_c={xc:+.2f}", ha="center", va="bottom", fontsize=9, rotation=90)

    plt.title(r"Piecewise DALI for $o(x)=100\,e^{x^2}$  (x0=0.10, $\sigma_o=0.05$)", fontsize=13)
    plt.xlabel("x"); plt.ylabel(r"$\log \mathcal{L}$  (up to const.)")
    # choose y-lims to reveal curvature differences
    y_min = 1.1 * min(exact_like.min(), fisher_single.min(), dali_single.min(), dali_piece.min())
    plt.ylim(y_min, 0.05)
    plt.xlim(xgrid.min(), xgrid.max())
    plt.grid(alpha=0.25, ls=":")
    plt.legend(frameon=True, fontsize=10, loc="lower left")

    outdir = Path("plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "dali_1d_piecewise.png"
    #plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")

    # Diagnostics
    print("\n[Tensors at centers]")
    for le in locals_:
        print(f"  center {le.x_c:+.3f} : F={float(le.F[0,0]):.6g}, G={float(le.G[0,0,0]):.6g}, H={float(le.H[0,0,0,0]):.6g}")

    print("\nWhat you should see:")
    print("  • 'DALI @ x0' improves Fisher near x0 but drifts far away (local expansion).")
    print("  • 'Piecewise DALI' stitches multiple local expansions and tracks")
    print("     the exact curve much better across the whole range.")
    print("  • Increase n_side (more centers) or adjust delta to extend/Refine coverage.")
    print("  • If transitions feel sharp, replace the nearest-center rule with a smooth")
    print("     blend (e.g. cosine or Gaussian weights over |x - x_c|).")

if __name__ == "__main__":
    main()
