"""Slow CCL benchmark for cached vs uncached forecasting pipelines.

This benchmark exercises a realistic weak-lensing forecasting workflow using
PyCCL, including:
- model evaluation
- Fisher matrix construction
- second-order DALI tensors
- Fisher Gaussian sampling
- DALI emcee sampling
- GetDist triangle plotting

The goal is not only to time the cached vs uncached paths, but also to confirm
that enabling input caching does not materially alter the scientific outputs.

Run:
    pytest -q tests/benchmarks/test_cache_forecasting_ccl_pipeline.py -m slow -s
"""

from __future__ import annotations

# IMPORTANT:
# These thread limits must be set before importing numpy / pyccl / scipy-backed code.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import pytest
from getdist import plots as getdist_plots

try:
    import cmasher as cmr
except ImportError:  # pragma: no cover
    cmr = None

from derivkit import ForecastKit

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

BENCHMARK_DIR = Path(__file__).resolve().parent
PLOT_DIR = BENCHMARK_DIR / "artifacts" / "cache_forecasting_ccl"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def smail_source_bins(
    z: np.ndarray,
    n_source: int,
    *,
    z0: float = 0.13,
    alpha: float = 0.78,
    beta: float = 2.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a parent Smail n(z) and equipopulated source bins."""
    z = np.asarray(z, dtype=float)

    nz = (z / z0) ** beta * np.exp(-((z / z0) ** alpha))
    nz /= np.trapezoid(nz, z)

    cdf = np.concatenate(
        [
            [0.0],
            np.cumsum(0.5 * (nz[1:] + nz[:-1]) * (z[1:] - z[:-1])),
        ]
    )

    def slice_bins(edges: np.ndarray) -> list[np.ndarray]:
        bins: list[np.ndarray] = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (z >= lo) & (z < hi if hi < edges[-1] else z <= hi)
            nz_i = np.where(mask, nz, 0.0)
            area = np.trapezoid(nz_i, z)
            if area <= 0.0:
                raise ValueError("Encountered empty tomographic bin.")
            nz_i /= area
            bins.append(nz_i)
        return bins

    edges = np.interp(np.linspace(0.0, cdf[-1], n_source + 1), cdf, z)
    edges[0] = z[0]
    edges[-1] = z[-1]

    return nz, slice_bins(edges)


def shear_power_spectra(
    theta: np.ndarray,
    *,
    z: np.ndarray,
    ell: np.ndarray,
    source_bins: list[np.ndarray],
) -> np.ndarray:
    """Return a cosmic-shear data vector with IA and baryonic effects.

    Parameters
    ----------
    theta
        Model parameters:
        [Omega_m, sigma8, A_IA, eta_IA, f_bar]
    z
        Redshift grid.
    ell
        Multipole array.
    source_bins
        List of source-bin redshift distributions.
    """
    om_m, sig8, ia_amp, ia_eta, fbar = map(float, np.asarray(theta, dtype=float))

    cosmo = ccl.Cosmology(
        Omega_c=om_m - 0.045,
        Omega_b=0.045,
        h=0.67,
        sigma8=sig8,
        n_s=0.96,
        transfer_function="boltzmann_camb",
    )

    z_p = 0.62
    ia_signal = ia_amp * ((1.0 + z) / (1.0 + z_p)) ** ia_eta

    vd = ccl.baryons.BaryonsvanDaalen19(fbar=fbar, mass_def="500c")
    pk_nl = cosmo.get_nonlin_power()
    pk_bar = vd.include_baryonic_effects(cosmo, pk_nl)

    tracers = [
        ccl.WeakLensingTracer(cosmo, dndz=(z, nz_i), ia_bias=(z, ia_signal))
        for nz_i in source_bins
    ]

    n_tr = len(tracers)
    n_cls = n_tr * (n_tr + 1) // 2
    out = np.empty(n_cls * ell.size, dtype=float)

    k = 0
    for a in range(n_tr):
        for b in range(a, n_tr):
            out[k : k + ell.size] = ccl.angular_cl(
                cosmo,
                tracers[a],
                tracers[b],
                ell,
                p_of_k_a=pk_bar,
            )
            k += ell.size

    return out


def make_ccl_problem() -> dict[str, Any]:
    """Build and return a deterministic CCL forecasting setup."""
    ell = np.geomspace(20.0, 2000.0, 20)
    z = np.linspace(0.0, 3.0, 300)

    n_source = 5
    _, source_bins = smail_source_bins(z, n_source=n_source)

    theta0 = np.array([0.315, 0.80, 0.50, 2.2, 0.70], dtype=float)

    def model(theta: np.ndarray) -> np.ndarray:
        return shear_power_spectra(theta, z=z, ell=ell, source_bins=source_bins)

    y0 = model(theta0)

    floor = 1e-12 * np.max(np.abs(y0))
    sigma_i = 0.05 * np.maximum(np.abs(y0), floor)
    cov = np.diag(sigma_i**2)

    names = ["om_m", "sig8", "ia_amp", "ia_eta", "f_bar"]
    labels = [
        r"\Omega_m",
        r"\sigma_8",
        r"A_{\rm IA}",
        r"\eta_{\rm IA}",
        r"f_{\rm bar}",
    ]

    sigma_prior = 0.10 * np.abs(theta0)
    fisher_prior = np.diag(1.0 / sigma_prior**2)
    prior_cov = np.diag(sigma_prior**2)

    return {
        "theta0": theta0,
        "cov": cov,
        "model": model,
        "names": names,
        "labels": labels,
        "fisher_prior": fisher_prior,
        "prior_cov": prior_cov,
    }


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return the relative Frobenius difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return float(np.linalg.norm(a - b) / denom)


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return the max absolute difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


def dali_flat_norm(dali: dict[int, tuple[np.ndarray, ...]]) -> float:
    """Return a single Frobenius norm for all DALI tensors."""
    flat_parts: list[np.ndarray] = []
    for order in sorted(dali):
        for tensor in dali[order]:
            flat_parts.append(np.asarray(tensor, dtype=float).ravel())
    if not flat_parts:
        return 0.0
    flat = np.concatenate(flat_parts)
    return float(np.linalg.norm(flat))


def plot_triangle(
    samples: list[Any],
    names: list[str],
    outfile: Path,
    *,
    title: str,
) -> None:
    """Save a GetDist triangle plot."""
    n = len(samples)

    if cmr is not None:
        colors = cmr.take_cmap_colors(
            "cmr.prinsenvlag",
            n,
            cmap_range=(0.2, 0.8),
            return_fmt="hex",
        )
    else:
        colors = None

    plotter = getdist_plots.get_subplot_plotter(width_inch=6)
    plotter.triangle_plot(
        samples,
        params=names,
        filled=False,
        contour_colors=colors,
        contour_lws=[2] * n,
        contour_ls=["-"] * n,
    )
    plotter.fig.suptitle(title, fontsize=11)
    plotter.export(str(outfile))
    plt.close("all")


def run_pipeline(
    *,
    use_input_cache: bool,
    emcee_seed: int,
    emcee_nwalkers: int = 32,
    emcee_nsteps: int = 180,
    emcee_burnin: int = 60,
) -> dict[str, Any]:
    """Run the full CCL forecasting pipeline and return all outputs."""
    cfg = make_ccl_problem()

    fk = ForecastKit(
        function=cfg["model"],
        theta0=cfg["theta0"],
        cov=cfg["cov"],
        use_input_cache=use_input_cache,
    )

    timings: dict[str, float] = {}

    t0 = perf_counter()
    fisher = np.asarray(
        fk.fisher(
            method="finite",
            extrapolation="ridders",
            n_workers=1,
        ),
        dtype=float,
    )
    timings["fisher"] = perf_counter() - t0

    t0 = perf_counter()
    dali = fk.dali(
        forecast_order=2,
        method="finite",
        extrapolation="ridders",
        n_workers=1,
    )
    timings["dali"] = perf_counter() - t0

    fisher_post = fisher + cfg["fisher_prior"]

    t0 = perf_counter()
    fisher_samples = fk.getdist_fisher_gaussian(
        fisher=fisher_post,
        names=cfg["names"],
        labels=cfg["labels"],
        label=f"Fisher {'cached' if use_input_cache else 'uncached'}",
    )
    timings["fisher_sampling"] = perf_counter() - t0

    prior_terms = [
        ("gaussian", {"mean": cfg["theta0"], "cov": cfg["prior_cov"]}),
    ]

    t0 = perf_counter()
    dali_samples = fk.getdist_dali_emcee(
        dali=dali,
        names=cfg["names"],
        labels=cfg["labels"],
        label=f"DALI {'cached' if use_input_cache else 'uncached'}",
        prior_terms=prior_terms,
        n_walkers=emcee_nwalkers,
        n_steps=emcee_nsteps,
        burn=emcee_burnin,
        seed=emcee_seed,
    )
    timings["dali_sampling"] = perf_counter() - t0

    fisher_plot = PLOT_DIR / f"triangle_fisher_cache_{int(use_input_cache)}.pdf"
    dali_plot = PLOT_DIR / f"triangle_dali_cache_{int(use_input_cache)}.pdf"

    t0 = perf_counter()
    plot_triangle(
        [fisher_samples],
        cfg["names"],
        fisher_plot,
        title=f"Fisher only | cache={use_input_cache}",
    )
    timings["fisher_plot"] = perf_counter() - t0

    t0 = perf_counter()
    plot_triangle(
        [fisher_samples, dali_samples],
        cfg["names"],
        dali_plot,
        title=f"Fisher + DALI | cache={use_input_cache}",
    )
    timings["dali_plot"] = perf_counter() - t0

    n_params = len(cfg["theta0"])

    fisher_means = np.asarray(fisher_samples.means[0, :n_params], dtype=float)
    fisher_cov = np.asarray(fisher_samples.covs[0][:n_params, :n_params], dtype=float)

    dali_means = np.asarray(dali_samples.getMeans()[:n_params], dtype=float)
    dali_cov = np.asarray(dali_samples.cov()[:n_params, :n_params], dtype=float)

    return {
        "fisher": fisher,
        "dali": dali,
        "fisher_samples": fisher_samples,
        "dali_samples": dali_samples,
        "fisher_means": fisher_means,
        "fisher_cov": fisher_cov,
        "dali_means": dali_means,
        "dali_cov": dali_cov,
        "fisher_plot": fisher_plot,
        "dali_plot": dali_plot,
        "timings": timings,
    }


def print_matrix_comparison(label: str, a: np.ndarray, b: np.ndarray) -> None:
    """Print a compact matrix comparison."""
    print("=" * 88)
    print(label)
    print("=" * 88)
    print(f"relative diff : {relative_difference(a, b):.3e}")
    print(f"max abs diff  : {max_abs_difference(a, b):.3e}")
    print()


def print_pipeline_summary(
    uncached: dict[str, Any],
    cached: dict[str, Any],
) -> None:
    """Print a compact cache comparison summary."""
    print("=" * 88)
    print("CCL forecasting pipeline cache comparison")
    print("=" * 88)

    for key in [
        "fisher",
        "dali",
        "fisher_sampling",
        "dali_sampling",
        "fisher_plot",
        "dali_plot",
    ]:
        t0 = uncached["timings"][key]
        t1 = cached["timings"][key]
        speedup = t0 / t1 if t1 > 0.0 else np.inf
        print(
            f"{key:16s} | uncached={t0:9.3f} s | cached={t1:9.3f} s | speedup={speedup:7.3f}x"
        )

    print()
    print(
        "DALI total norm | "
        f"uncached={dali_flat_norm(uncached['dali']):.6e} | "
        f"cached={dali_flat_norm(cached['dali']):.6e}"
    )
    print(
        "DALI mean diff  | "
        f"rel={relative_difference(uncached['dali_means'], cached['dali_means']):.3e} | "
        f"max_abs={max_abs_difference(uncached['dali_means'], cached['dali_means']):.3e}"
    )
    print(
        "DALI cov diff   | "
        f"rel={relative_difference(uncached['dali_cov'], cached['dali_cov']):.3e} | "
        f"max_abs={max_abs_difference(uncached['dali_cov'], cached['dali_cov']):.3e}"
    )
    print()


def assert_dali_dict_allclose(
    cached: dict[int, tuple[np.ndarray, ...]],
    uncached: dict[int, tuple[np.ndarray, ...]],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert that two DALI tensor dictionaries match."""
    assert set(cached) == set(uncached)

    for order in sorted(cached):
        tensors_a = cached[order]
        tensors_b = uncached[order]
        assert len(tensors_a) == len(tensors_b)

        for arr_a, arr_b in zip(tensors_a, tensors_b, strict=True):
            np.testing.assert_allclose(
                np.asarray(arr_a, dtype=float),
                np.asarray(arr_b, dtype=float),
                rtol=rtol,
                atol=atol,
            )


def test_ccl_cache_pipeline_numerical_health_and_sampling() -> None:
    """Check that caching preserves real CCL forecasting outputs.

    This is intentionally a slow, end-to-end scientific benchmark.
    """
    uncached = run_pipeline(
        use_input_cache=False,
        emcee_seed=12345,
    )
    cached = run_pipeline(
        use_input_cache=True,
        emcee_seed=12345,
    )

    print_pipeline_summary(uncached, cached)

    print_matrix_comparison("Fisher matrix", uncached["fisher"], cached["fisher"])
    print_matrix_comparison(
        "Fisher posterior sample covariance",
        uncached["fisher_cov"],
        cached["fisher_cov"],
    )
    print_matrix_comparison(
        "DALI posterior sample covariance",
        uncached["dali_cov"],
        cached["dali_cov"],
    )

    # Fisher matrix must agree very tightly.
    np.testing.assert_allclose(
        cached["fisher"],
        uncached["fisher"],
        rtol=1e-9,
        atol=1e-12,
    )

    # DALI tensors should also agree very tightly.
    assert_dali_dict_allclose(
        cached["dali"],
        uncached["dali"],
        rtol=1e-8,
        atol=1e-11,
    )

    # Fisher Gaussian samples are deterministic once Fisher is fixed.
    np.testing.assert_allclose(
        cached["fisher_means"],
        uncached["fisher_means"],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cached["fisher_cov"],
        uncached["fisher_cov"],
        rtol=1e-10,
        atol=1e-12,
    )

    # DALI emcee sampling is stochastic, so compare summary statistics.
    np.testing.assert_allclose(
        cached["dali_means"],
        uncached["dali_means"],
        rtol=5e-2,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        cached["dali_cov"],
        uncached["dali_cov"],
        rtol=1e-1,
        atol=1e-4,
    )

    # Plot files should exist and be non-empty.
    for key in ["fisher_plot", "dali_plot"]:
        path_uncached = uncached[key]
        path_cached = cached[key]

        assert path_uncached.exists()
        assert path_cached.exists()

        assert path_uncached.stat().st_size > 0
        assert path_cached.stat().st_size > 0

    # Sanity: everything finite.
    assert np.all(np.isfinite(uncached["fisher"]))
    assert np.all(np.isfinite(cached["fisher"]))
    assert np.all(np.isfinite(uncached["fisher_means"]))
    assert np.all(np.isfinite(cached["fisher_means"]))
    assert np.all(np.isfinite(uncached["dali_means"]))
    assert np.all(np.isfinite(cached["dali_means"]))


def test_ccl_cache_pipeline_speed_smoke() -> None:
    """Smoke benchmark for timing cached vs uncached full CCL runs.

    This does not require caching to beat uncached in every environment, but it
    prints the timings so you can monitor whether cache is helping in realistic
    science code.
    """
    uncached = run_pipeline(
        use_input_cache=False,
        emcee_seed=24680,
        emcee_nwalkers=24,
        emcee_nsteps=120,
        emcee_burnin=40,
    )
    cached = run_pipeline(
        use_input_cache=True,
        emcee_seed=24680,
        emcee_nwalkers=24,
        emcee_nsteps=120,
        emcee_burnin=40,
    )

    print_pipeline_summary(uncached, cached)

    total_uncached = sum(uncached["timings"].values())
    total_cached = sum(cached["timings"].values())

    print("=" * 88)
    print("Total pipeline wall time")
    print("=" * 88)
    print(f"uncached total : {total_uncached:.3f} s")
    print(f"cached total   : {total_cached:.3f} s")
    print(
        f"speedup        : {total_uncached / total_cached:.3f}x"
        if total_cached > 0.0
        else "speedup        : inf"
    )
    print()

    # This is a smoke check, not a hard performance contract.
    assert total_uncached > 0.0
    assert total_cached > 0.0
