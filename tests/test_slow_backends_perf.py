"""Slow backend perf + sanity test (CAMB + PyCCL) for DerivKit.

What it does (for each backend, if installed):
- builds y0 once (for toy diagonal covariance)
- runs Fisher at two fiducials
- runs DALI(2) at two fiducials
- prints clean diagnostics: wall time, CPU time, RSS, calls/exec counts,
  average expensive-call time, and basic shapes

It is SKIPPED by default in normal pytest runs.

Run it explicitly:
  RUN_SLOW=1 pytest -q -k slow_backends_perf -s


Notes:
- Keep n_workers=1 (CCL/CAMB stability).
- This file intentionally does *not* add any callable-level caching.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

try:
    import camb  # type: ignore
except Exception:
    camb = None  # type: ignore

try:
    import pyccl as ccl  # type: ignore
except Exception:
    ccl = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

from derivkit.forecast_kit import ForecastKit

_CURRENT_PHASE = "init"


def _slow_enabled() -> bool:
    """Enables the slow backend perf test."""
    return os.environ.get("RUN_SLOW", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _should_skip() -> bool:
    """Decides whether to skip the slow backend perf test."""
    if _slow_enabled():
        return False
    # If not explicitly enabled, keep it skipped by default.
    return True


pytestmark = pytest.mark.skipif(
    _should_skip(),
    reason="slow backend perf test (set RUN_SLOW=1 to enable)",
)


def _force_single_thread_math() -> None:
    """Force single-threaded math for stability."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _pid() -> int:
    """Return the current process ID, or -1 if unavailable."""
    try:
        return os.getpid()
    except Exception:
        return -1


def _cpu_time_s() -> float:
    """Get the current process CPU time in seconds."""
    t = os.times()
    return float(t.user + t.system)


def _rss_mb() -> float:
    """Get the current process resident set size in MB."""
    if psutil is None:
        return float("nan")
    try:
        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / (1024.0**2)
    except Exception:
        return float("nan")


def _stamp(msg: str) -> None:
    """Print a timestamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Counters:
    """Counters for the slow backend perf test."""
    calls: int = 0
    expensive: int = 0
    model_time_s: float = 0.0


class PhaseTimer:
    """Context manager for timing a phase of the test."""
    def __init__(self, label: str):
        """Initialize the timer."""
        self.label = label
        self.t_wall0 = 0.0
        self.t_cpu0 = 0.0
        self.rss0 = 0.0

    def __enter__(self) -> "PhaseTimer":
        """Start the timer."""
        self.t_wall0 = time.time()
        self.t_cpu0 = _cpu_time_s()
        self.rss0 = _rss_mb()
        _stamp(f"--- {self.label} (pid={_pid()}) ---")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the timer."""
        wall = time.time() - self.t_wall0
        cpu = _cpu_time_s() - self.t_cpu0
        rss = _rss_mb()
        drss = rss - self.rss0 if np.isfinite(rss) and np.isfinite(self.rss0) else float("nan")
        _stamp(
            f"{self.label} done | wall={wall:.2f}s | cpu={cpu:.2f}s | "
            f"rss={rss:.1f}MB (Δ{drss:+.1f}MB)"
        )


class StopEarly(RuntimeError):
    """Raised to stop early."""
    pass


def _maybe_stop_early(
    *,
    backend: str,
    phase: str,
    counters: Counters,
    max_expensive: int,
) -> None:
    """Raise StopEarly if the counters indicate we should stop early."""
    if max_expensive <= 0:
        return
    if counters.expensive >= max_expensive:
        raise StopEarly(
            f"{backend} {phase}: stopping early after {counters.expensive} expensive evals"
        )


def _report_counters(tag: str, c: Counters) -> None:
    """Print a summary of the counters."""
    avg = (c.model_time_s / c.expensive) if c.expensive else float("nan")
    rate = (c.expensive / c.model_time_s) if c.model_time_s > 0 else float("nan")
    _stamp(
        f"{tag}: CALLS={c.calls} | EXPENSIVE={c.expensive} | "
        f"backend_time={c.model_time_s:.2f}s | avg_expensive={avg:.3f}s | "
        f"rate={rate:.2f} exp/s"
    )


def _toy_cov(y0: np.ndarray, frac: float) -> np.ndarray:
    """Build a toy diagonal covariance matrix."""
    y0 = np.asarray(y0, float).reshape(-1)
    return np.diag((frac * np.abs(y0) + 1e-30) ** 2)


def _make_camb_model(
    *,
    lmax: int,
    ells: np.ndarray
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Make a CAMB model function and metadata."""
    if camb is None:
        pytest.skip("CAMB not available (pip install camb)")

    ells = np.asarray(ells, int).reshape(-1)
    if np.any(ells < 2) or np.any(ells > lmax):
        raise ValueError("Bad ells for CAMB.")

    meta = {
        "backend": "camb",
        "version": getattr(camb, "__version__", "unknown"),
        "lmax": int(lmax),
        "ells": ells.copy(),
    }

    counters = Counters()

    def f(theta: np.ndarray) -> np.ndarray:
        """CAMB model function."""
        counters.calls += 1
        counters.expensive += 1
        t0 = time.time()

        max_exp = int(os.environ.get("DK_SLOW_MAX_EXPENSIVE", "0"))
        _maybe_stop_early(
            backend=meta["backend"],
            phase=_CURRENT_PHASE,
            counters=counters,
            max_expensive=max_exp,
        )

        every = int(os.environ.get("DK_SLOW_PROGRESS_EVERY", "10"))
        if every > 0 and (counters.expensive % every == 0):
            _report_counters(f"{meta['backend']} {_CURRENT_PHASE} progress", counters)

        H0, ombh2 = map(float, np.asarray(theta, float).reshape(-1))

        pars = camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=0.122,
            mnu=0.06,
            omk=0.0,
            tau=0.06,
            As=2.0e-9,
            ns=0.965,
            halofit_version="mead",
            lmax=lmax,
        )
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        tt_full = powers["total"][:, 0]  # ell=0..lmax
        y = tt_full[ells]

        counters.model_time_s += time.time() - t0
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"Non-finite CAMB output for theta={theta!r}")
        return np.asarray(y, float)

    meta["counters"] = counters
    return f, meta


def _make_ccl_model(
    *,
    n_bins: int,
    n_ell: int,
    z_n: int
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Make a PyCCL model function and metadata."""
    if ccl is None:
        pytest.skip("PyCCL not available (pip install pyccl)")

    # grids
    ell = np.unique(np.round(np.geomspace(20, 2000, n_ell)).astype(int))
    z = np.linspace(0.0, 3.5, z_n)

    def smail_nz(
        zv: np.ndarray,
        *,
        z0: float,
        alpha: float,
        beta: float
    ) -> np.ndarray:
        """Compute Smail-type redshift distribution."""
        zz = np.maximum(np.asarray(zv, float), 0.0)
        return (zz / z0) ** beta * np.exp(-((zz / z0) ** alpha))

    def norm_over_z(zv: np.ndarray, nz: np.ndarray) -> np.ndarray:
        """Normalize a redshift distribution."""
        norm = np.trapezoid(nz, x=zv)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Bad normalization.")
        return nz / norm

    alpha, beta, z0 = 0.78, 2.0, 0.13
    nz_parent = norm_over_z(z, smail_nz(z, z0=z0, alpha=alpha, beta=beta))

    # equipopulated bin edges + hard bins
    dz = z[1:] - z[:-1]
    nz_mid = 0.5 * (nz_parent[1:] + nz_parent[:-1])
    cdf_inner = np.cumulative_sum(nz_mid * dz)

    cdf = np.concatenate(([0.0], cdf_inner))
    targets = (np.arange(1, n_bins) / n_bins) * cdf[-1]
    inner = np.interp(targets, cdf, z)
    edges = np.concatenate([[z[0]], inner, [z[-1]]])

    dndz_list: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi if hi < edges[-1] else z <= hi)
        ni = np.where(m, nz_parent, 0.0)
        if np.trapezoid(ni, x=z) <= 0:
            raise ValueError("Empty bin.")
        dndz_list.append(norm_over_z(z, ni))

    n_pairs = n_bins * (n_bins + 1) // 2
    n_data = n_pairs * ell.size

    counters = Counters()

    meta = {
        "backend": "ccl",
        "version": getattr(ccl, "__version__", "unknown"),
        "n_bins": int(n_bins),
        "n_pairs": int(n_pairs),
        "n_ell": int(ell.size),
        "n_data": int(n_data),
        "edges": edges.copy(),
        "counters": counters,
    }

    def f(theta: np.ndarray) -> np.ndarray:
        """PyCCL model function."""
        counters.calls += 1
        counters.expensive += 1
        t0 = time.time()

        max_exp = int(os.environ.get("DK_SLOW_MAX_EXPENSIVE", "0"))
        _maybe_stop_early(
            backend=meta["backend"],
            phase=_CURRENT_PHASE,
            counters=counters,
            max_expensive=max_exp,
        )

        every = int(os.environ.get("DK_SLOW_PROGRESS_EVERY", "10"))
        if every > 0 and (counters.expensive % every == 0):
            _report_counters(f"{meta['backend']} {_CURRENT_PHASE} progress", counters)

        om_m, sig8, ia_amp = map(float, np.asarray(theta, float).reshape(-1))

        cosmo = ccl.Cosmology(
            Omega_c=om_m - 0.045,
            Omega_b=0.045,
            h=0.67,
            sigma8=sig8,
            n_s=0.96,
        )

        ia_signal = ia_amp * (z / 0.62) ** 2.2

        tracers = [
            ccl.WeakLensingTracer(cosmo, dndz=(z, dndz_list[i]), ia_bias=(z, ia_signal))
            for i in range(n_bins)
        ]

        out = np.empty(n_data, dtype=float)
        k = 0
        for a in range(n_bins):
            ta = tracers[a]
            for b in range(a, n_bins):
                out[k : k + ell.size] = ccl.angular_cl(cosmo, ta, tracers[b], ell)
                k += ell.size

        counters.model_time_s += time.time() - t0
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"Non-finite CCL output for theta={theta!r}")
        return out

    return f, meta


def _run_one_backend(
    *,
    backend_name: str,
    function: Callable[[np.ndarray], np.ndarray],
    counters: Counters,
    theta0_1: np.ndarray,
    theta0_2: np.ndarray,
    cov_frac: float,
    method: str,
    stepsize: float,
) -> None:
    """Run a single backend end-to-end: y0, Fisher(F1,F2), DALI(D1,D2).

    StopEarly exits the current backend and continues to the next backend.
    """
    global _CURRENT_PHASE

    _stamp(
        f"backend={backend_name} | method={method} | stepsize={stepsize:g} | cov_frac={cov_frac:g}"
    )
    _stamp(f"theta0_1={np.array2string(theta0_1, precision=6)}")
    _stamp(f"theta0_2={np.array2string(theta0_2, precision=6)}")

    # -------------------------
    # y0 (for covariance)
    # -------------------------
    counters.calls = counters.expensive = 0
    counters.model_time_s = 0.0
    try:
        _CURRENT_PHASE = f"{backend_name}: y0"
        with PhaseTimer(_CURRENT_PHASE):
            y0 = function(theta0_1)
    except StopEarly as e:
        _report_counters(str(e), counters)
        _stamp(f"{backend_name}: early-stop during y0 -> "
               f"continuing to next backend")
        return

    _stamp(
        f"y0: n_data={y0.size} | preview={np.array2string(np.asarray(y0)[:5],
                                                          precision=3)} ..."
    )
    _report_counters(f"{backend_name}: y0 counters", counters)

    cov = _toy_cov(y0, cov_frac)

    # Fisher (two fiducials)
    for tag, th in (("F1", theta0_1), ("F2", theta0_2)):
        fk = ForecastKit(function=function, theta0=th, cov=cov)

        counters.calls = counters.expensive = 0
        counters.model_time_s = 0.0
        try:
            _CURRENT_PHASE = f"{backend_name}: Fisher {tag}"
            with PhaseTimer(_CURRENT_PHASE):
                fisher = fk.fisher(method=method, n_workers=1, stepsize=stepsize)
        except StopEarly as e:
            _report_counters(str(e), counters)
            _stamp(f"{backend_name}: early-stop during Fisher {tag} -> "
                   f"continuing to next backend")
            return

        fisher = np.asarray(fisher, float)
        _stamp(
            f"{backend_name}: Fisher {tag} shape={fisher.shape} "
            f"diag={np.array2string(np.diag(fisher), precision=3)}"
        )
        _report_counters(f"{backend_name}: Fisher {tag} counters", counters)

    # DALI(2) (two fiducials)
    for tag, th in (("D1", theta0_1), ("D2", theta0_2)):
        fk = ForecastKit(function=function, theta0=th, cov=cov)

        counters.calls = counters.expensive = 0
        counters.model_time_s = 0.0
        try:
            _CURRENT_PHASE = f"{backend_name}: DALI(2) {tag}"
            with PhaseTimer(_CURRENT_PHASE):
                dali = fk.dali(
                    forecast_order=2,
                    n_workers=1,
                    method=method,
                    stepsize=stepsize,
                )
        except StopEarly as e:
            _report_counters(str(e), counters)
            _stamp(f"{backend_name}: early-stop during DALI {tag} ->"
                   f" continuing to next backend")
            return

        assert dali is not None
        _report_counters(f"{backend_name}: DALI(2) {tag} counters",
                         counters)

    _stamp(f"{backend_name}: completed all phases")


def test_slow_backends_perf() -> None:
    """Run the slow backend perf test."""
    _force_single_thread_math()

    # Defaults for this slow test (can be overridden in the shell)
    os.environ.setdefault("DK_SLOW_MAX_EXPENSIVE",
                          "20")  # stop each backend/phase early
    os.environ.setdefault("DK_SLOW_PROGRESS_EVERY", "5")  # frequent feedback
    os.environ.setdefault("DK_SLOW_METHOD", "finite")
    os.environ.setdefault("DK_SLOW_STEPSIZE", "1e-2")
    os.environ.setdefault("DK_SLOW_COV_FRAC", "0.10")

    _stamp("=== DerivKit slow backend perf test ===")
    _stamp(f"platform={platform.platform()}")
    _stamp(f"python={platform.python_version()} | pid={_pid()} | rss={_rss_mb():.1f}MB")

    # knobs (env-overridable)
    method = os.environ.get("DK_SLOW_METHOD", "finite").strip()
    stepsize = float(os.environ.get("DK_SLOW_STEPSIZE", "1e-2"))
    cov_frac = float(os.environ.get("DK_SLOW_COV_FRAC", "0.10"))

    # 1) CAMB (small vector)
    camb_f, camb_meta = _make_camb_model(lmax=50, ells=np.arange(2, 51, 4))
    _stamp(f"CAMB version={camb_meta['version']} | lmax={camb_meta['lmax']} | n_data={camb_meta['ells'].size}")
    _run_one_backend(
        backend_name="CAMB",
        function=camb_f,
        counters=camb_meta["counters"],
        theta0_1=np.array([67.5, 0.0220], float),
        theta0_2=np.array([67.5, 0.0222], float),
        cov_frac=cov_frac,
        method=method,
        stepsize=stepsize,
    )

    # 2) PyCCL (bigger vector)
    ccl_f, ccl_meta = _make_ccl_model(n_bins=5, n_ell=60, z_n=300)
    _stamp(
        f"PyCCL version={ccl_meta['version']} | n_bins={ccl_meta['n_bins']} | "
        f"n_ell={ccl_meta['n_ell']} | n_data={ccl_meta['n_data']} | "
        f"edges={np.array2string(ccl_meta['edges'], precision=3)}"
    )
    _run_one_backend(
        backend_name="PyCCL",
        function=ccl_f,
        counters=ccl_meta["counters"],
        theta0_1=np.array([0.315, 0.80, 0.40], float),
        theta0_2=np.array([0.315, 0.80, 0.50], float),
        cov_frac=cov_frac,
        method=method,
        stepsize=stepsize,
    )

    _stamp("=== done ===")
