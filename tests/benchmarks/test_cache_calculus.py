"""Benchmark input caching for calculus-level derivative builders.

Run:
    pytest -q tests/benchmarks/test_cache_calculus.py -m slow -s
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from derivkit.calculus.gradient import build_gradient
from derivkit.calculus.hessian import build_hessian
from derivkit.calculus.hyper_hessian import build_hyper_hessian
from derivkit.calculus.jacobian import build_jacobian

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


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


def print_summary(label: str, summary: dict[str, Any]) -> None:
    """Print a compact benchmark summary.

    Args:
        label: Human-readable label for the benchmark variant.
        summary: Timing summary returned by :func:`benchmark`.
    """
    print(label)
    print(f"  mean   : {summary['mean']:.6f} s")
    print(f"  median : {summary['median']:.6f} s")
    print(f"  min    : {summary['min']:.6f} s")
    print(f"  max    : {summary['max']:.6f} s")
    print()


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


def expensive_scalar_model_factory(
    p: int,
) -> tuple[Callable[[np.ndarray], float], dict[str, int]]:
    """Return an expensive scalar-valued model and its call counter.

    Args:
        p: Number of model parameters.

    Returns:
        Tuple of ``(model, calls)``, where ``calls["count"]`` tracks the
        number of model evaluations.
    """
    calls = {"count": 0}

    def model(theta: np.ndarray) -> float:
        calls["count"] += 1
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != p:
            raise ValueError(f"Expected theta with size {p}, got {theta.size}.")

        grid = np.linspace(0.0, 1.0, 8000)
        w = np.exp(-grid**2)

        total = 0.0
        for i in range(p):
            total += np.sum(np.sin((theta[i] + 0.2 * (i + 1)) * grid) * w)
            total += 0.3 * theta[i] ** 2

        for i in range(p - 1):
            total += 0.5 * theta[i] * theta[i + 1]
            total += np.sum(
                np.cos((theta[i] - 0.3 * theta[i + 1]) * grid) * w
            )

        total += np.exp(0.1 * np.sum(theta))
        return float(total)

    return model, calls


def expensive_vector_model_factory(
    p: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, int]]:
    """Return an expensive vector-valued model and its call counter.

    Args:
        p: Number of model parameters and output components.

    Returns:
        Tuple of ``(model, calls)``, where ``calls["count"]`` tracks the
        number of model evaluations.
    """
    calls = {"count": 0}

    def model(theta: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != p:
            raise ValueError(f"Expected theta with size {p}, got {theta.size}.")

        grid = np.linspace(0.0, 1.0, 6000)
        w = np.exp(-0.5 * grid)

        out = np.empty(p, dtype=float)
        total_theta = np.sum(theta)

        for i in range(p):
            a = np.sin((theta[i] + 0.1 * (i + 1)) * grid)
            b = np.cos((theta[i] + 0.2 * total_theta) * grid)
            val = np.sum(a * w) + np.sum(b * w) + theta[i] ** 2

            if i > 0:
                val += theta[i] * theta[i - 1]
            if i < p - 1:
                val -= 0.5 * theta[i] * theta[i + 1]

            out[i] = val

        return out

    return model, calls


def report_speedup(
    uncached_summary: dict[str, Any],
    cached_summary: dict[str, Any],
    calls_no_cache: dict[str, int],
    calls_with_cache: dict[str, int],
) -> None:
    """Print timing and call-count comparison for cached vs uncached runs.

    Args:
        uncached_summary: Summary for the uncached benchmark run.
        cached_summary: Summary for the cached benchmark run.
        calls_no_cache: Model call counter for the uncached case.
        calls_with_cache: Model call counter for the cached case.
    """
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

SIZE_CASES = [
    pytest.param(2, id="p2"),
    pytest.param(3, id="p3"),
    pytest.param(5, id="p5"),
]

HYPER_SIZE_CASES = [
    pytest.param(2, id="p2"),
    pytest.param(3, id="p3"),
]


@pytest.mark.parametrize("p", SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_gradient_cache_benchmark(
    p: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for calculus-level gradient construction.

    Args:
        p: Number of parameters in the scalar-valued model.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"Calculus gradient | p={p} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=1.0)

    model_no_cache, calls_no_cache = expensive_scalar_model_factory(p)
    model_with_cache, calls_with_cache = expensive_scalar_model_factory(p)

    def uncached() -> np.ndarray:
        out = build_gradient(
            function=model_no_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            dk_init_kwargs={"use_input_cache": False},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = build_gradient(
            function=model_with_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            dk_init_kwargs={"use_input_cache": True},
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
        uncached_summary,
        cached_summary,
        calls_no_cache,
        calls_with_cache,
    )


@pytest.mark.parametrize("p", SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_jacobian_cache_benchmark(
    p: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for calculus-level Jacobian construction.

    Args:
        p: Number of parameters and outputs in the vector-valued model.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"Calculus jacobian | p={p} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.8)

    model_no_cache, calls_no_cache = expensive_vector_model_factory(p)
    model_with_cache, calls_with_cache = expensive_vector_model_factory(p)

    def uncached() -> np.ndarray:
        out = build_jacobian(
            function=model_no_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            dk_init_kwargs={"use_input_cache": False},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = build_jacobian(
            function=model_with_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            dk_init_kwargs={"use_input_cache": True},
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
        uncached_summary,
        cached_summary,
        calls_no_cache,
        calls_with_cache,
    )


@pytest.mark.parametrize("p", SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_hessian_cache_benchmark(
    p: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for calculus-level Hessian construction.

    Args:
        p: Number of parameters in the scalar-valued model.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"Calculus hessian | p={p} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.7)

    model_no_cache, calls_no_cache = expensive_scalar_model_factory(p)
    model_with_cache, calls_with_cache = expensive_scalar_model_factory(p)

    def uncached() -> np.ndarray:
        out = build_hessian(
            function=model_no_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            inner_workers=1,
            dk_init_kwargs={"use_input_cache": False},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = build_hessian(
            function=model_with_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            inner_workers=1,
            dk_init_kwargs={"use_input_cache": True},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached, repeats=3)
    cached_summary = benchmark(cached, repeats=3)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-9,
        atol=1e-11,
    )

    report_speedup(
        uncached_summary,
        cached_summary,
        calls_no_cache,
        calls_with_cache,
    )


@pytest.mark.parametrize("p", HYPER_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_hyper_hessian_cache_benchmark(
    p: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Benchmark caching for calculus-level hyper-Hessian construction.

    Args:
        p: Number of parameters in the scalar-valued model.
        method: Differentiation method name.
        method_kwargs: Additional differentiation keyword arguments.
    """
    print("=" * 80)
    print(f"Calculus hyper-hessian | p={p} | method={method}")
    print("=" * 80)

    theta0 = make_theta0(p, scale=0.5)

    model_no_cache, calls_no_cache = expensive_scalar_model_factory(p)
    model_with_cache, calls_with_cache = expensive_scalar_model_factory(p)

    def uncached() -> np.ndarray:
        out = build_hyper_hessian(
            function=model_no_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            inner_workers=1,
            dk_init_kwargs={"use_input_cache": False},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    def cached() -> np.ndarray:
        out = build_hyper_hessian(
            function=model_with_cache,
            theta0=theta0,
            method=method,
            n_workers=1,
            inner_workers=1,
            dk_init_kwargs={"use_input_cache": True},
            **method_kwargs,
        )
        return np.asarray(out, dtype=float)

    uncached_summary = benchmark(uncached, repeats=2)
    cached_summary = benchmark(cached, repeats=2)

    np.testing.assert_allclose(
        cached_summary["result"],
        uncached_summary["result"],
        rtol=1e-8,
        atol=1e-10,
    )

    report_speedup(
        uncached_summary,
        cached_summary,
        calls_no_cache,
        calls_with_cache,
    )
