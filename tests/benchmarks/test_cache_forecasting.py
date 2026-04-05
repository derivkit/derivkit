"""Benchmark input caching for forecasting-level builders.

Run:
    pytest -q tests/benchmarks/test_cache_forecasting.py -m slow -s
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

COMPACT_RESULTS: list[dict[str, Any]] = []


def benchmark(
    func: Callable[[], np.ndarray],
    repeats: int = 5,
    warmups: int = 1,
) -> dict[str, Any]:
    """Return timing summary for repeated executions of ``func``.

    Args:
        func: Zero-argument callable to benchmark.
        repeats: Number of timed repetitions.
        warmups: Number of untimed warmup executions before timing begins.

    Returns:
        Dictionary containing timing statistics and the last computed result.
    """
    for _ in range(warmups):
        func()

    times: list[float] = []
    last_result: np.ndarray | None = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_result = np.asarray(result, dtype=float)

    assert last_result is not None

    return {
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "result": last_result,
    }


def benchmark_object(
    func: Callable[[], dict[int, tuple[np.ndarray, ...]]],
    repeats: int = 3,
    warmups: int = 1,
) -> dict[str, Any]:
    """Return timing summary for repeated executions of an object-valued callable.

    This is used for DALI, which returns a nested dictionary of tensors rather than
    a single ndarray.

    Args:
        func: Zero-argument callable to benchmark.
        repeats: Number of timed repetitions.
        warmups: Number of untimed warmup executions before timing begins.

    Returns:
        Dictionary containing timing statistics and the last computed result.
    """
    for _ in range(warmups):
        func()

    times: list[float] = []
    last_result: dict[int, tuple[np.ndarray, ...]] | None = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_result = result

    assert last_result is not None

    return {
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "result": last_result,
    }


def print_summary(label: str, summary: dict[str, Any]) -> None:
    """Print a compact benchmark summary.

    Args:
        label: Human-readable label for the benchmark variant.
        summary: Timing summary returned by :func:`benchmark` or
            :func:`benchmark_object`.
    """
    print(label)
    print(f"  mean   : {summary['mean']:.6f} s")
    print(f"  median : {summary['median']:.6f} s")
    print(f"  min    : {summary['min']:.6f} s")
    print(f"  max    : {summary['max']:.6f} s")
    print()


def report_speedup(
    case_label: str,
    uncached_summary: dict[str, Any],
    cached_summary: dict[str, Any],
    calls_no_cache: dict[str, int],
    calls_with_cache: dict[str, int],
) -> None:
    """Print timing and call-count comparison for cached vs uncached runs.

    Args:
        case_label: Short label identifying the benchmark case.
        uncached_summary: Summary for the uncached benchmark run.
        cached_summary: Summary for the cached benchmark run.
        calls_no_cache: Model call counter for the uncached case.
        calls_with_cache: Model call counter for the cached case.
    """
    print_summary("No cache", uncached_summary)
    print_summary("With cache", cached_summary)

    speedup = uncached_summary["mean"] / cached_summary["mean"]
    reduction = 100.0 * (
        uncached_summary["mean"] - cached_summary["mean"]
    ) / uncached_summary["mean"]

    print(f"Model calls without cache : {calls_no_cache['count']}")
    print(f"Model calls with cache    : {calls_with_cache['count']}")
    print(f"Speedup factor            : {speedup:.3f}x")
    print(f"Wall-time reduction       : {reduction:.2f}%")
    print()

    COMPACT_RESULTS.append(
        {
            "case": case_label,
            "calls_no_cache": calls_no_cache["count"],
            "calls_with_cache": calls_with_cache["count"],
            "speedup": speedup,
            "reduction": reduction,
        }
    )

    assert calls_with_cache["count"] <= calls_no_cache["count"]


def make_theta0(p: int, *, scale: float = 1.0) -> np.ndarray:
    """Return a deterministic test point of length ``p``.

    Args:
        p: Number of parameters.
        scale: Overall multiplicative scale applied to the base point.

    Returns:
        One-dimensional parameter vector with alternating signs.
    """
    base = np.linspace(0.25, 0.85, p, dtype=float)
    signs = np.where(np.arange(p) % 2 == 0, 1.0, -1.0)
    return scale * base * signs


def expensive_forecast_model_factory(
    p: int,
    n_obs: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, int]]:
    """Return an expensive vector-valued forecasting model and call counter.

    Args:
        p: Number of parameters.
        n_obs: Number of observables returned by the model.

    Returns:
        Tuple ``(model, calls)`` where ``calls["count"]`` tracks evaluations.
    """
    calls = {"count": 0}
    weights = np.linspace(0.7, 1.5, n_obs, dtype=float)

    def model(theta: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != p:
            raise ValueError(f"Expected theta with size {p}, got {theta.size}.")

        grid = np.linspace(0.0, 1.0, 5000)
        w = np.exp(-0.5 * grid)

        total_theta = np.sum(theta)
        out = np.empty(n_obs, dtype=float)

        for k in range(n_obs):
            acc = 0.0
            for i in range(p):
                phase = theta[i] + 0.08 * (i + 1) * (k + 1)
                acc += np.sum(np.sin(phase * grid) * w)
                acc += 0.2 * weights[k] * theta[i] ** 2

            for i in range(p - 1):
                acc += 0.15 * (k + 1) * theta[i] * theta[i + 1]
                acc += np.sum(
                    np.cos((theta[i] - 0.25 * theta[i + 1] + 0.05 * k) * grid) * w
                )

            acc += np.exp(0.03 * (k + 1) * total_theta)
            out[k] = acc

        return out

    return model, calls


def expensive_xy_model_factory(
    p: int,
    n_x: int,
    n_y: int,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], dict[str, int]]:
    """Return an expensive X-Y model and call counter.

    Args:
        p: Number of parameters.
        n_x: Number of noisy input variables.
        n_y: Number of output observables.

    Returns:
        Tuple ``(mu_xy, calls)`` where ``calls["count"]`` tracks evaluations.
    """
    calls = {"count": 0}

    def mu_xy(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        x = np.asarray(x, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)

        if theta.size != p:
            raise ValueError(f"Expected theta with size {p}, got {theta.size}.")
        if x.size != n_x:
            raise ValueError(f"Expected x with size {n_x}, got {x.size}.")

        grid = np.linspace(0.0, 1.0, 4000)
        w = np.exp(-grid)

        out = np.empty(n_y, dtype=float)
        sx = np.sum(x)
        st = np.sum(theta)

        for j in range(n_y):
            acc = 0.0
            for i in range(p):
                coeff = theta[i] + 0.1 * (j + 1)
                acc += np.sum(np.sin((coeff + 0.2 * sx) * grid) * w)

            for i in range(n_x):
                acc += 0.3 * x[i] ** 2
                acc += np.sum(np.cos((x[i] + 0.15 * st + 0.05 * j) * grid) * w)

            acc += 0.2 * np.dot(theta, theta)
            acc += 0.1 * np.dot(x, x)
            acc += 0.05 * sx * st
            out[j] = acc

        return out

    return mu_xy, calls


def make_spd_matrix(n: int, diag_boost: float = 1.0) -> np.ndarray:
    """Return a deterministic symmetric positive definite matrix.

    Args:
        n: Matrix size.
        diag_boost: Value added to the diagonal for stability.

    Returns:
        SPD matrix with shape ``(n, n)``.
    """
    a = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    mat = a @ a.T
    mat /= np.max(np.abs(mat))
    mat += diag_boost * np.eye(n, dtype=float)
    return mat


def flatten_dali(dali: dict[int, tuple[np.ndarray, ...]]) -> np.ndarray:
    """Flatten a DALI tensor dictionary into a 1D array for comparison.

    Args:
        dali: DALI tensor dictionary keyed by forecast order.

    Returns:
        One-dimensional array containing all tensor entries in order.
    """
    flat: list[np.ndarray] = []

    for order in sorted(dali):
        for tensor in dali[order]:
            flat.append(np.asarray(tensor, dtype=float).ravel())

    if not flat:
        return np.array([], dtype=float)

    return np.concatenate(flat)


METHOD_CASES = [
    pytest.param("adaptive", {}, id="adaptive"),
    pytest.param("polyfit", {}, id="polyfit"),
    pytest.param("finite", {}, id="finite"),
    pytest.param("finite", {"extrapolation": "ridders"}, id="finite-ridders"),
]

FORECAST_SIZE_CASES = [
    pytest.param(2, 4, id="p2-n4"),
    pytest.param(3, 6, id="p3-n6"),
]

DALI_SIZE_CASES = [
    pytest.param(2, 4, id="p2-n4"),
    pytest.param(3, 5, id="p3-n5"),
]

XY_SIZE_CASES = [
    pytest.param(2, 2, 3, id="p2-nx2-ny3"),
    pytest.param(3, 2, 4, id="p3-nx2-ny4"),
]


@pytest.mark.parametrize(("p", "n_obs"), FORECAST_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_fisher_cache_benchmark(
    p: int,
    n_obs: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for ForecastKit Fisher matrix construction.

    Args:
        p: Number of parameters.
        n_obs: Number of observables in the model output.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"ForecastKit fisher | p={p} | n_obs={n_obs} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.8)
    cov = make_spd_matrix(n_obs, diag_boost=2.0)

    model_no_cache, calls_no_cache = expensive_forecast_model_factory(p, n_obs)
    model_with_cache, calls_with_cache = expensive_forecast_model_factory(p, n_obs)

    fk_no_cache = ForecastKit(
        function=model_no_cache,
        theta0=theta0,
        cov=cov,
        use_input_cache=False,
    )
    fk_with_cache = ForecastKit(
        function=model_with_cache,
        theta0=theta0,
        cov=cov,
        use_input_cache=True,
    )

    def uncached() -> np.ndarray:
        out = fk_no_cache.fisher(
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = fk_with_cache.fisher(
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached)
    cached_summary = benchmark(cached)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-10,
        atol=1e-12,
    )

    report_speedup(
        case_label=f"ForecastKit fisher | p={p} | n_obs={n_obs} | method={method}",
        uncached_summary=uncached_summary,
        cached_summary=cached_summary,
        calls_no_cache=calls_no_cache,
        calls_with_cache=calls_with_cache,
    )

@pytest.mark.parametrize(("p", "n_obs"), DALI_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_dali_cache_benchmark(
    p: int,
    n_obs: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for ForecastKit DALI tensor construction.

    Args:
        p: Number of parameters.
        n_obs: Number of observables in the model output.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"ForecastKit dali | p={p} | n_obs={n_obs} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.6)
    cov = make_spd_matrix(n_obs, diag_boost=2.5)

    model_no_cache, calls_no_cache = expensive_forecast_model_factory(p, n_obs)
    model_with_cache, calls_with_cache = expensive_forecast_model_factory(p, n_obs)

    fk_no_cache = ForecastKit(
        function=model_no_cache,
        theta0=theta0,
        cov=cov,
        use_input_cache=False,
    )
    fk_with_cache = ForecastKit(
        function=model_with_cache,
        theta0=theta0,
        cov=cov,
        use_input_cache=True,
    )

    def uncached() -> dict[int, tuple[np.ndarray, ...]]:
        return fk_no_cache.dali(
            method=method,
            forecast_order=2,
            n_workers=1,
            **method_kwargs,
        )

    def cached() -> dict[int, tuple[np.ndarray, ...]]:
        return fk_with_cache.dali(
            method=method,
            forecast_order=2,
            n_workers=1,
            **method_kwargs,
        )

    uncached_summary = benchmark_object(uncached, repeats=3)
    cached_summary = benchmark_object(cached, repeats=3)

    uncached_flat = flatten_dali(uncached_summary["result"])
    cached_flat = flatten_dali(cached_summary["result"])

    np.testing.assert_allclose(
        cached_flat,
        uncached_flat,
        rtol=1e-9,
        atol=1e-11,
    )

    report_speedup(
        case_label=f"ForecastKit dali | p={p} | n_obs={n_obs} | method={method}",
        uncached_summary=uncached_summary,
        cached_summary=cached_summary,
        calls_no_cache=calls_no_cache,
        calls_with_cache=calls_with_cache,
    )


@pytest.mark.parametrize(("p", "n_x", "n_y"), XY_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_xy_fisher_cache_benchmark(
    p: int,
    n_x: int,
    n_y: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for ForecastKit X-Y Fisher construction.

    Args:
        p: Number of parameters.
        n_x: Number of noisy inputs.
        n_y: Number of noisy outputs.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(
        f"ForecastKit xy_fisher | p={p} | n_x={n_x} | n_y={n_y} | method={method}"
    )
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.9)
    x0 = np.linspace(-0.4, 0.6, n_x, dtype=float)

    mu_xy_no_cache, calls_no_cache = expensive_xy_model_factory(p, n_x, n_y)
    mu_xy_with_cache, calls_with_cache = expensive_xy_model_factory(p, n_x, n_y)

    cov_xy = make_spd_matrix(n_x + n_y, diag_boost=3.0)
    cov_yy = cov_xy[n_x:, n_x:]

    fk_no_cache = ForecastKit(
        function=None,
        theta0=theta0,
        cov=cov_yy,
        use_input_cache=False,
    )
    fk_with_cache = ForecastKit(
        function=None,
        theta0=theta0,
        cov=cov_yy,
        use_input_cache=False,
    )

    def uncached() -> np.ndarray:
        out = fk_no_cache.xy_fisher(
            x0=x0,
            mu_xy=mu_xy_no_cache,
            cov_xy=cov_xy,
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = fk_with_cache.xy_fisher(
            x0=x0,
            mu_xy=mu_xy_with_cache,
            cov_xy=cov_xy,
            method=method,
            n_workers=1,
            dk_init_kwargs={"use_input_cache": True},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached, repeats=4)
    cached_summary = benchmark(cached, repeats=4)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-10,
        atol=1e-12,
    )

    report_speedup(
        case_label=(
            f"ForecastKit xy_fisher | p={p} | n_x={n_x} | "
            f"n_y={n_y} | method={method}"
        ),
        uncached_summary=uncached_summary,
        cached_summary=cached_summary,
        calls_no_cache=calls_no_cache,
        calls_with_cache=calls_with_cache,
    )


def print_compact_summary() -> None:
    """Print a compact one-line summary for all benchmark cases."""
    print("=" * 80)
    print("Compact benchmark summary")
    print("=" * 80)

    for row in COMPACT_RESULTS:
        print(
            f"{row['case']} | "
            f"Model calls without cache : {row['calls_no_cache']} | "
            f"Model calls with cache    : {row['calls_with_cache']} | "
            f"Speedup factor            : {row['speedup']:.3f}x | "
            f"Wall-time reduction       : {row['reduction']:.2f}%"
        )

    print()


def test_print_compact_summary() -> None:
    """Print a compact summary after all benchmark cases finish."""
    print_compact_summary()
