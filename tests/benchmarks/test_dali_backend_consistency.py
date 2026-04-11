"""Small fast backend check for Fisher + DALI(2) on a reduced CCL problem.

Purpose
-------
Quickly test whether the new higher-order backend tuning helps polyfit/adaptive
preserve more DALI structure, without waiting for the full slow benchmark.

Run
---
python tests/benchmarks/test_dali_backend_consistency.py
"""

from __future__ import annotations

import os
import warnings
from time import perf_counter
from typing import Any

import pytest

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from derivkit import ForecastKit
ccl = pytest.importorskip("pyccl")

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


def smail_source_bins(
    z: np.ndarray,
    n_source: int,
    *,
    z0: float = 0.13,
    alpha: float = 0.78,
    beta: float = 2.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns Smail-type tomographic bins."""
    z = np.asarray(z, dtype=float)

    nz = (z / z0) ** beta * np.exp(-((z / z0) ** alpha))
    nz /= np.trapezoid(nz, z)

    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (nz[1:] + nz[:-1]) * (z[1:] - z[:-1]))]
    )

    edges = np.interp(np.linspace(0.0, cdf[-1], n_source + 1), cdf, z)
    edges[0] = z[0]
    edges[-1] = z[-1]

    bins: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (z >= lo) & (z < hi if hi < edges[-1] else z <= hi)
        nz_i = np.where(mask, nz, 0.0)
        area = np.trapezoid(nz_i, z)
        nz_i /= area
        bins.append(nz_i)

    return nz, bins


def shear_power_spectra(
    theta: np.ndarray,
    *,
    z: np.ndarray,
    ell: np.ndarray,
    source_bins: list[np.ndarray],
) -> np.ndarray:
    """Computes the shear power spectrum at each bin."""
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
            out[k:k + ell.size] = ccl.angular_cl(
                cosmo,
                tracers[a],
                tracers[b],
                ell,
                p_of_k_a=pk_bar,
            )
            k += ell.size

    return out


def make_small_problem() -> dict[str, Any]:
    """Config for a simple cosmic shear problem."""
    print("=" * 80)
    print("Building SMALL CCL problem")
    print("=" * 80)

    ell = np.geomspace(30.0, 1000.0, 8)
    z = np.linspace(0.0, 2.0, 120)
    _, source_bins = smail_source_bins(z, n_source=3)

    theta0 = np.array([0.315, 0.80, 0.50, 2.2, 0.70], dtype=float)

    def model(theta: np.ndarray) -> np.ndarray:
        return shear_power_spectra(theta, z=z, ell=ell, source_bins=source_bins)

    y0 = model(theta0)
    floor = 1e-12 * np.max(np.abs(y0))
    sigma_i = 0.05 * np.maximum(np.abs(y0), floor)
    cov = np.diag(sigma_i**2)

    print(f"theta0 shape: {theta0.shape}")
    print(f"n_source bins: {len(source_bins)}")
    print(f"data vector length: {y0.size}")
    print(f"covariance shape: {cov.shape}")
    print()

    return {
        "theta0": theta0,
        "cov": cov,
        "model": model,
    }


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative difference between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return float(np.linalg.norm(a - b) / denom)


def run_backend(
    *,
    cfg: dict[str, Any],
    name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Runs the derivative backend for fisher and dali computations."""
    print("=" * 80)
    print(f"Running backend: {name}")
    print("=" * 80)
    print("kwargs:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print()

    fk = ForecastKit(
        function=cfg["model"],
        theta0=cfg["theta0"],
        cov=cfg["cov"],
        use_input_cache=True,
    )

    fisher_kwargs = dict(kwargs)
    dali_kwargs = dict(kwargs)

    print("-> Fisher")
    t0 = perf_counter()
    fisher = np.asarray(fk.fisher(n_workers=1, **fisher_kwargs), dtype=float)
    tf = perf_counter() - t0
    print(f"   done in {tf:.3f} s")
    print(f"   ||F||  = {np.linalg.norm(fisher):.6e}")
    print()

    print("-> DALI order 2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = perf_counter()
        dali = fk.dali(forecast_order=2, n_workers=1, **dali_kwargs)
        td = perf_counter() - t0

    print(f"   done in {td:.3f} s")
    print(f"   warnings caught: {len(caught)}")
    for w in caught:
        print(f"   warning: {w.message}")

    fisher_from_dali = np.asarray(dali[1][0], dtype=float)
    d1, d2 = map(lambda x: np.asarray(x, dtype=float), dali[2])

    print(f"   ||F_from_dali|| = {np.linalg.norm(fisher_from_dali):.6e}")
    print(f"   ||d1||          = {np.linalg.norm(d1):.6e}")
    print(f"   ||d2||          = {np.linalg.norm(d2):.6e}")
    print(f"   Fisher internal rel diff = {rel_diff(fisher, fisher_from_dali):.3e}")
    print()

    return {
        "name": name,
        "kwargs": kwargs,
        "fisher": fisher,
        "fisher_from_dali": fisher_from_dali,
        "d1": d1,
        "d2": d2,
        "fisher_time": tf,
        "dali_time": td,
    }


def compare_to_reference(
    *,
    results: dict[str, dict[str, Any]],
    ref_name: str,
) -> None:
    """Compares the forecast results to the reference results."""
    ref = results[ref_name]

    print("=" * 80)
    print(f"Comparison summary | reference = {ref_name}")
    print("=" * 80)

    for name, res in results.items():
        fisher_rel = rel_diff(res["fisher"], ref["fisher"])
        d1_rel = rel_diff(res["d1"], ref["d1"])
        d2_rel = rel_diff(res["d2"], ref["d2"])

        d1_ratio = np.linalg.norm(res["d1"]) / max(np.linalg.norm(ref["d1"]), 1e-30)
        d2_ratio = np.linalg.norm(res["d2"]) / max(np.linalg.norm(ref["d2"]), 1e-30)

        print(name)
        print(f"  fisher_time : {res['fisher_time']:.3f} s")
        print(f"  dali_time   : {res['dali_time']:.3f} s")
        print(f"  fisher_rel  : {fisher_rel:.3e}")
        print(f"  d1_rel      : {d1_rel:.3e}")
        print(f"  d2_rel      : {d2_rel:.3e}")
        print(f"  d1_ratio    : {d1_ratio:.3e}")
        print(f"  d2_ratio    : {d2_ratio:.3e}")
        print()


def main() -> None:
    """Runs the script."""
    cfg = make_small_problem()

    backends = [
        (
            "finite_ridders",
            {
                "method": "finite",
                "extrapolation": "ridders",
            },
        ),
        (
            "polyfit",
            {
                "method": "polyfit",
            },
        ),
        (
            "adaptive",
            {
                "method": "adaptive",
            },
        ),
    ]

    results: dict[str, dict[str, Any]] = {}
    for name, kwargs in backends:
        results[name] = run_backend(cfg=cfg, name=name, kwargs=kwargs)

    compare_to_reference(results=results, ref_name="finite_ridders")


if __name__ == "__main__":
    main()
