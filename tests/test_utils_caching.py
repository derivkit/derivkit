"""Tests for derivkit.utils.caching."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose

from derivkit.utils.caching import wrap_theta_cache_builtin


def test_wrap_theta_cache_builtin_has_cache_api():
    """Tests that wrap_theta_cache_builtin should preserve lru_cache attributes."""
    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that adds 1 to its input."""
        return np.asarray([theta[0] + 1.0], dtype=float)

    wrapped = wrap_theta_cache_builtin(f, maxsize=128)

    assert callable(wrapped)

    assert hasattr(wrapped, "cache_info")
    assert hasattr(wrapped, "cache_clear")

    cache_info = getattr(wrapped, "cache_info")
    cache_clear = getattr(wrapped, "cache_clear")

    assert callable(cache_info)
    assert callable(cache_clear)

    info = cache_info()

    assert hasattr(info, "hits")
    assert hasattr(info, "misses")
    assert hasattr(info, "maxsize")
    assert hasattr(info, "currsize")

    assert getattr(info, "maxsize") == 128


def test_wrap_theta_cache_builtin_caches_function_values():
    """Tests that wrap_theta_cache_builtin should cache results for identical inputs."""
    calls = {"n": 0}

    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that squares its input and adds 2.0."""
        calls["n"] += 1
        return np.asarray([theta[0] ** 2, theta[0] + 2.0], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, maxsize=256))

    theta = np.asarray([2.0], dtype=float)

    y1 = wrapped(theta)
    y2 = wrapped(theta)

    assert_allclose(y1, [4.0, 4.0], rtol=0.0, atol=0.0)
    assert_allclose(y2, [4.0, 4.0], rtol=0.0, atol=0.0)

    assert calls["n"] == 1

    info = wrapped.cache_info()
    assert info.misses == 1
    assert info.hits == 1
    assert info.currsize == 1


def test_wrap_theta_cache_builtin_cache_clear_resets_counts():
    """Tests that cache_clear should reset hits/misses and empty the cache."""
    calls = {"n": 0}

    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that returns its input."""
        calls["n"] += 1
        return np.asarray([theta[0]], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, maxsize=16))

    wrapped(np.asarray([1.0], dtype=float))
    info1 = wrapped.cache_info()
    assert calls["n"] == 1
    assert info1.misses == 1
    assert info1.hits == 0
    assert info1.currsize == 1

    wrapped.cache_clear()

    info2 = wrapped.cache_info()
    assert info2.misses == 0
    assert info2.hits == 0
    assert info2.currsize == 0

    wrapped(np.asarray([1.0], dtype=float))
    info3 = wrapped.cache_info()
    assert calls["n"] == 2
    assert info3.misses == 1
    assert info3.hits == 0
    assert info3.currsize == 1


def test_wrap_theta_cache_builtin_copy_true_returns_distinct_arrays():
    """Tests that copy=True should return a fresh array each call."""
    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that adds 1 to its input."""
        return np.asarray([theta[0], theta[0] + 1.0], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, copy=True))

    a1 = wrapped(np.asarray([3.0], dtype=float))
    a2 = wrapped(np.asarray([3.0], dtype=float))

    assert_allclose(a1, a2, rtol=0.0, atol=0.0)
    assert a1 is not a2
    assert not np.shares_memory(a1, a2)


def test_wrap_theta_cache_builtin_copy_false_reuses_cached_object():
    """Tests that copy=False should return the cached array object."""
    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that adds 1 to its input."""
        return np.asarray([theta[0], theta[0] + 1.0], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, copy=False))

    a1 = wrapped(np.asarray([3.0], dtype=float))
    a2 = wrapped(np.asarray([3.0], dtype=float))

    assert a1 is a2


def test_wrap_theta_cache_builtin_rounds_output_to_decimal_places():
    """Tests that wrap_theta_cache_builtin should round outputs to the requested precision."""
    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that returns pi and 1.234567890123456."""
        _ = theta
        return np.asarray([np.pi, 1.234567890123456], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, number_decimal_places=3))

    y = wrapped(np.asarray([0.0], dtype=float))

    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.shape == (2,)

    assert_allclose(y[0], np.round(np.pi, 3), rtol=0.0, atol=0.0)
    assert_allclose(y[1], np.round(1.234567890123456, 3), rtol=0.0, atol=0.0)


def test_wrap_theta_cache_builtin_flattens_outputs_to_1d():
    """Tests that wrap_theta_cache_builtin should flatten function outputs to 1D."""
    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that returns a 3x1 array."""
        _ = theta
        return np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f))

    y = wrapped(np.asarray([0.0], dtype=float))

    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.shape == (3,)
    assert_allclose(y, [1.0, 2.0, 3.0], rtol=0.0, atol=0.0)


def test_wrap_theta_cache_builtin_does_not_truncate_cache_key_current_behavior():
    """Tests that cache keys are based on raw float tuples (no input truncation in key)."""
    calls = {"n": 0}

    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that returns 0.0."""
        _ = theta
        calls["n"] += 1
        return np.asarray([0.0], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, number_decimal_places=14))

    wrapped(np.asarray([1.0], dtype=float))
    wrapped(np.asarray([1.0 + 1e-15], dtype=float))

    assert calls["n"] == 2
    info = wrapped.cache_info()
    assert info.misses == 2
    assert info.hits == 0


def test_wrap_theta_cache_builtin_lru_eviction_respects_maxsize():
    """Tests that wrap_theta_cache_builtin should respect maxsize via LRU eviction."""
    calls = {"n": 0}

    def f(theta: np.ndarray) -> np.ndarray:
        """A dummy function that returns its input."""
        calls["n"] += 1
        return np.asarray([theta[0]], dtype=float)

    wrapped = cast(Any, wrap_theta_cache_builtin(f, maxsize=2))

    wrapped(np.asarray([1.0], dtype=float))  # miss
    wrapped(np.asarray([2.0], dtype=float))  # miss
    wrapped(np.asarray([3.0], dtype=float))  # miss -> evict 1.0
    wrapped(np.asarray([2.0], dtype=float))  # hit
    wrapped(np.asarray([1.0], dtype=float))  # miss again

    info = wrapped.cache_info()
    assert calls["n"] == 4
    assert info.misses == 4
    assert info.hits == 1
    assert info.maxsize == 2
