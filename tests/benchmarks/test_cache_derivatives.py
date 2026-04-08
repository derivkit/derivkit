"""Benchmark input caching for low-level derivative evaluations.

Run:
    pytest -q tests/benchmarks/test_cache_derivatives.py -m slow -s
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


def benchmark(
    func: Callable[[], np.ndarray],
    repeats: int = 7,
    warmups: int = 1,
) -> dict:
    """Returns timing summary for repeated executions of ``func``."""
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


def print_summary(label: str, summary: dict) -> None:
    """Prints a compact benchmark summary."""
    print(label)
    print(f"  mean   : {summary['mean']:.6f} s")
    print(f"  median : {summary['median']:.6f} s")
    print(f"  min    : {summary['min']:.6f} s")
    print(f"  max    : {summary['max']:.6f} s")
    print()


def expensive_scalar_model_factory() -> tuple[Callable[[float], float], dict]:
    """Returns an expensive scalar-valued function and its call counter."""
    calls = {"count": 0}

    def model(x: float) -> float:
        calls["count"] += 1
        grid = np.linspace(0.0, 1.0, 8000)
        val = np.sin(x * grid) + np.cos(2.3 * x * grid)
        weight = np.exp(-grid**2)
        return float(np.sum(val * weight) + x**2 + np.exp(0.1 * x))

    return model, calls


def expensive_vector_model_factory() -> tuple[Callable[[float], np.ndarray], dict]:
    """Returns an expensive vector-valued function and its call counter."""
    calls = {"count": 0}

    def model(x: float) -> np.ndarray:
        calls["count"] += 1
        grid = np.linspace(0.0, 1.0, 6000)
        a = np.sin(x * grid)
        b = np.cos(1.7 * x * grid)
        c = np.exp(-0.5 * grid)

        return np.array([
            np.sum(a * c) + x**2,
            np.sum(b * c) + x**3,
            np.sum((a + b) * c) + np.sin(x),
        ], dtype=float)

    return model, calls


def run_scalar_derivative_case(
    *,
    method: str,
    method_kwargs: dict | None = None,
    repeats: int = 7,
) -> None:
    """Benchmarks a scalar first-derivative case."""
    print("=" * 80)
    print(f"DerivativeKit scalar first derivative | method={method}")
    print("=" * 80)

    x0 = 0.7
    method_kwargs = dict(method_kwargs or {})

    model_no_cache, calls_no_cache = expensive_scalar_model_factory()
    model_with_cache, calls_with_cache = expensive_scalar_model_factory()

    def uncached() -> np.ndarray:
        kit = DerivativeKit(
            model_no_cache,
            x0,
            use_input_cache=False,
        )
        out = kit.differentiate(
            order=1,
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        kit = DerivativeKit(
            model_with_cache,
            x0,
            use_input_cache=True,
        )
        out = kit.differentiate(
            order=1,
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached, repeats=repeats)
    cached_summary = benchmark(cached, repeats=repeats)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-10,
        atol=1e-12,
    )

    print_summary("No cache", uncached_summary)
    print_summary("With cache", cached_summary)

    print(f"Model calls without cache : {calls_no_cache['count']}")
    print(f"Model calls with cache    : {calls_with_cache['count']}")

    speedup = uncached_summary["mean"] / cached_summary["mean"]
    reduction = 100.0 * (
        uncached_summary["mean"] - cached_summary["mean"]
    ) / uncached_summary["mean"]

    print(f"Speedup factor            : {speedup:.3f}x")
    print(f"Wall-time reduction       : {reduction:.2f}%")
    print()

    assert calls_with_cache["count"] <= calls_no_cache["count"]


def run_vector_derivative_case(
    *,
    method: str,
    method_kwargs: dict | None = None,
    repeats: int = 7,
) -> None:
    """Benchmarks a vector first-derivative case."""
    print("=" * 80)
    print(f"DerivativeKit vector first derivative | method={method}")
    print("=" * 80)

    x0 = 0.35
    method_kwargs = dict(method_kwargs or {})

    model_no_cache, calls_no_cache = expensive_vector_model_factory()
    model_with_cache, calls_with_cache = expensive_vector_model_factory()

    def uncached() -> np.ndarray:
        kit = DerivativeKit(
            model_no_cache,
            x0,
            use_input_cache=False,
        )
        out = kit.differentiate(
            order=1,
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        kit = DerivativeKit(
            model_with_cache,
            x0,
            use_input_cache=True,
        )
        out = kit.differentiate(
            order=1,
            method=method,
            n_workers=1,
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached, repeats=repeats)
    cached_summary = benchmark(cached, repeats=repeats)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-10,
        atol=1e-12,
    )

    print_summary("No cache", uncached_summary)
    print_summary("With cache", cached_summary)

    print(f"Model calls without cache : {calls_no_cache['count']}")
    print(f"Model calls with cache    : {calls_with_cache['count']}")

    speedup = uncached_summary["mean"] / cached_summary["mean"]
    reduction = 100.0 * (
        uncached_summary["mean"] - cached_summary["mean"]
    ) / uncached_summary["mean"]

    print(f"Speedup factor            : {speedup:.3f}x")
    print(f"Wall-time reduction       : {reduction:.2f}%")
    print()

    assert calls_with_cache["count"] <= calls_no_cache["count"]


METHOD_CASES = [
    pytest.param("adaptive", {}, id="adaptive"),
    pytest.param("polyfit", {}, id="polyfit"),
    pytest.param("finite", {}, id="finite"),
    pytest.param(
        "finite",
        {"extrapolation": "ridders"},
        id="finite-ridders",
    ),
]


@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_scalar_derivative_cache_benchmark(
    method: str,
    method_kwargs: dict,
) -> None:
    """Benchmarks scalar derivative caching across derivative methods."""
    run_scalar_derivative_case(
        method=method,
        method_kwargs=method_kwargs,
        repeats=7,
    )


@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_vector_derivative_cache_benchmark(
    method: str,
    method_kwargs: dict,
) -> None:
    """Benchmarks vector derivative caching across derivative methods."""
    run_vector_derivative_case(
        method=method,
        method_kwargs=method_kwargs,
        repeats=7,
    )
