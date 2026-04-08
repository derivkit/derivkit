"""Tests for DerivativeKit input caching."""

from __future__ import annotations

import time

import numpy as np
import pytest

from derivkit import DerivativeKit


def test_cache_info_hits_increase_on_repeated_finite_call() -> None:
    """Tests that repeated identical finite-difference calls should produce cache hits."""
    def f(x: float) -> float:
        return x**2 + 3.0 * x - 1.0

    dk = DerivativeKit(f, x0=1.0, use_input_cache=True)

    info0 = dk.cache_info()
    assert info0 is not None
    assert info0.hits == 0

    dk.differentiate(method="finite", order=1)
    info1 = dk.cache_info()
    assert info1 is not None
    misses_after_first = info1.misses

    dk.differentiate(method="finite", order=1)
    info2 = dk.cache_info()
    assert info2 is not None

    assert info2.hits > info1.hits
    assert info2.misses == misses_after_first


def test_cache_reduces_raw_function_evaluations_for_repeated_finite_call() -> None:
    """Tests that repeated identical finite-difference calls should reuse cached evaluations."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return x**2

    dk = DerivativeKit(f, x0=1.0, use_input_cache=True)

    dk.differentiate(method="finite", order=1)
    first_total = counter["n"]
    assert first_total > 0

    dk.differentiate(method="finite", order=1)
    second_total = counter["n"]

    assert second_total == first_total

    info = dk.cache_info()
    assert info is not None
    assert info.hits > 0


def test_cache_wrapper_direct_calls_hit_cache() -> None:
    """Tests that direct repeated calls to the cached wrapper should only evaluate once."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return x**2 + 1.0

    dk = DerivativeKit(f, x0=0.5, use_input_cache=True)

    g = dk._function_cached

    y1 = g(1.25)
    y2 = g(1.25)
    y3 = g(1.25)

    assert y1 == pytest.approx(y2)
    assert y2 == pytest.approx(y3)
    assert counter["n"] == 1

    info = dk.cache_info()
    assert info is not None
    assert info.hits >= 2


def test_cache_clear_resets_cache_contents() -> None:
    """Tests that clearing the cache should force recomputation on the next identical input."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return 2.0 * x

    dk = DerivativeKit(f, x0=1.0, use_input_cache=True)
    g = dk._function_cached

    _ = g(0.75)
    _ = g(0.75)
    assert counter["n"] == 1

    dk.cache_clear()

    _ = g(0.75)
    assert counter["n"] == 2


def test_cache_can_be_disabled() -> None:
    """Tests that disabling caching should cause repeated identical calls to re-evaluate."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return x**2

    dk = DerivativeKit(f, x0=1.0, use_input_cache=False)

    dk.differentiate(method="finite", order=1)
    first_total = counter["n"]
    assert first_total > 0

    dk.differentiate(method="finite", order=1)
    second_total = counter["n"]

    assert second_total > first_total
    assert dk.cache_info() is None


def test_cache_works_for_adaptive_repeated_call() -> None:
    """Tests that repeated identical adaptive calls should also produce cache hits."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return np.sin(x)

    dk = DerivativeKit(f, x0=0.3, use_input_cache=True)

    dk.differentiate(method="adaptive", order=1)
    first_total = counter["n"]
    assert first_total > 0

    dk.differentiate(method="adaptive", order=1)
    second_total = counter["n"]

    assert second_total == first_total

    info = dk.cache_info()
    assert info is not None
    assert info.hits > 0


@pytest.mark.slow
def test_repeated_identical_call_is_faster_with_cache() -> None:
    """The second identical call is usually faster thanks to cache reuse.

    This test uses a deliberately expensive scalar function. Timing-based
    assertions can be noisy on shared CI, so we only require the second
    call to be not slower, with a small tolerance.
    """
    def expensive(x: float) -> float:
        arr = np.linspace(0.0, 1.0, 20000)
        return float(np.sum(np.sin(arr * x) + np.cos(arr * x**2)))

    dk = DerivativeKit(expensive, x0=0.7, use_input_cache=True)

    t0 = time.perf_counter()
    dk.differentiate(method="finite", order=1)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    dk.differentiate(method="finite", order=1)
    t3 = time.perf_counter()

    first = t1 - t0
    second = t3 - t2

    info = dk.cache_info()
    assert info is not None
    assert info.hits > 0

    # Allow slight noise, but second call should not be meaningfully slower.
    assert second <= first * 1.10


def test_cache_rounding_reuses_nearby_float_inputs_when_enabled() -> None:
    """Tests that rounded cache keys should merge nearby float inputs."""
    counter = {"n": 0}

    def f(x: float) -> float:
        counter["n"] += 1
        return x**2

    dk = DerivativeKit(
        f,
        x0=1.0,
        use_input_cache=True,
        cache_number_decimal_places=6,
    )

    g = dk._function_cached

    y1 = g(1.0000001)
    y2 = g(1.0000002)

    assert y1 == pytest.approx(y2)
    assert counter["n"] == 1

    info = dk.cache_info()
    assert info is not None
    assert info.hits >= 1
