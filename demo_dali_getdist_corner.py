"""Demo: DALI (3 params) -> GetDist corner plot.

- Builds Fisher + DALI tensors using ForecastKit
- Converts tensors into GetDist MCSamples (importance sampling)
- Makes a GetDist triangle plot and saves it

Usage:
  python demos/demo_dali_getdist_corner_3d.py

Notes:
- Requires getdist installed.
- Uses `convention="matplotlib_loglike"` to match Niko's contour prefactors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from derivkit import ForecastKit
from derivkit.forecasting.integrations import dali_to_mcsamples_importance


def model_3d(theta: np.ndarray) -> np.ndarray:
    """3-parameter nonlinear toy model.

    theta = [x, eps, a]

    observables (3):
      o1 = (1 + eps) * 100 * exp(x^2) + 20*a
      o2 = (1 + 0.3*eps) *  40 * exp(0.5*x) + 10*a*exp(-x)
      o3 = (1 + 0.2*eps) *  30 * exp(-0.7*x) + 50*a**2
    """
    x, eps, a = float(theta[0]), float(theta[1]), float(theta[2])

    o1 = (1.0 + eps) * (1e2 * np.exp(x**2)) + 2e1 * a
    o2 = (1.0 + 0.3 * eps) * (4e1 * np.exp(0.5 * x)) + 1e1 * a * np.exp(-x)
    o3 = (1.0 + 0.2 * eps) * (3e1 * np.exp(-0.7 * x)) + 5e1 * (a**2)

    return np.array([o1, o2, o3])


def resolve_outdir(file: str, default_rel: str = "plots") -> Path:
    outdir = Path(file).resolve().parent / default_rel
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main() -> None:
    theta0 = np.array([0.10, 0.0, 0.02])  # [x, eps, a]

    # 3 observables, simple correlated covariance (SPD)
    cov = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ],
        dtype=float,
    )

    fk = ForecastKit(function=model_3d, theta0=theta0, cov=cov)
    F = fk.fisher()
    G, H = fk.dali()

    gd = dali_to_mcsamples_importance(
        theta0, F, G, H,
        names=["x", "eps", "a"],
        labels=[r"x", r"\epsilon", r"a"],
        nsamp=600_000,
        proposal_scale=1.7,
        convention="matplotlib_loglike",
        seed=0,
        label="DALI (importance)",
    )

    from getdist import plots

    g = plots.get_subplot_plotter(width_inch=7)
    g.triangle_plot([gd], filled=True)

    outdir = resolve_outdir(__file__, default_rel="plots")
    outfile = outdir / "demo_dali_getdist_corner_3d.png"
    g.export(str(outfile))
    print("saved:", outfile)


if __name__ == "__main__":
    main()
