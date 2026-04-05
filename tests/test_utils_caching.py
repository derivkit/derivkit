"""Unit tests for ``utils_caching``."""

import numpy as np

from derivkit.utils.caching import (
    _normalize_cache_input,
    wrap_input_cache,
)


def test_normalize_cache_input_scalar_exact():
    """Tests that scalar inputs are normalized exactly without rounding."""
    key = _normalize_cache_input(1.2345, number_decimal_places=None)
    assert key == ("scalar", 1.2345)


def test_normalize_cache_input_scalar_rounded():
    """Tests that scalar inputs are normalized with rounding when requested."""
    key = _normalize_cache_input(1.2345, number_decimal_places=2)
    assert key == ("scalar", round(1.2345, 2))


def test_normalize_cache_input_array_preserves_shape_and_values():
    """Tests that array inputs preserve shape and flattened values in the cache key."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    key = _normalize_cache_input(x, number_decimal_places=None)

    assert key[0] == "array"
    assert key[1] == (2, 2)
    assert key[2] == (1.0, 2.0, 3.0, 4.0)


def test_normalize_cache_input_array_rounding_collapses_nearby_inputs():
    """Tests that rounding makes nearby array inputs share the same cache key."""
    x1 = np.array([1.2344, 2.3454])
    x2 = np.array([1.23449, 2.34549])

    key1 = _normalize_cache_input(x1, number_decimal_places=3)
    key2 = _normalize_cache_input(x2, number_decimal_places=3)

    assert key1 == key2


def test_wrap_input_cache_hits_for_repeated_scalar_calls():
    """Tests that repeated scalar calls hit the cache."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x**2

    cached_f = wrap_input_cache(f)

    assert cached_f(3.0) == 9.0
    assert cached_f(3.0) == 9.0
    assert calls["n"] == 1

    info = cached_f.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_wrap_input_cache_hits_for_equivalent_array_calls():
    """Tests that equivalent array inputs hit the cache."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.asarray(x) * 2.0

    cached_f = wrap_input_cache(f)

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([1.0, 2.0, 3.0])

    y1 = cached_f(x1)
    y2 = cached_f(x2)

    np.testing.assert_allclose(y1, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(y2, [2.0, 4.0, 6.0])
    assert calls["n"] == 1

    info = cached_f.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_wrap_input_cache_rounding_can_merge_nearby_scalar_inputs():
    """Tests that rounding can merge nearby scalar inputs into one cache entry."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x

    cached_f = wrap_input_cache(f, number_decimal_places=3)

    y1 = cached_f(1.2344)
    y2 = cached_f(1.23449)

    assert y1 == round(1.2344, 3)
    assert y2 == y1
    assert calls["n"] == 1


def test_wrap_input_cache_exact_mode_does_not_merge_nearby_scalar_inputs():
    """Tests that exact mode keeps nearby scalar inputs distinct."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x

    cached_f = wrap_input_cache(f, number_decimal_places=None)

    y1 = cached_f(1.2344)
    y2 = cached_f(1.23449)

    assert y1 == 1.2344
    assert y2 == 1.23449
    assert calls["n"] == 2


def test_wrap_input_cache_rounding_can_merge_nearby_array_inputs():
    """Tests that rounding can merge nearby array inputs into one cache entry."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.sum(x)

    cached_f = wrap_input_cache(f, number_decimal_places=3)

    x1 = np.array([1.2344, 2.3454])
    x2 = np.array([1.23449, 2.34549])

    y1 = cached_f(x1)
    y2 = cached_f(x2)

    assert y1 == y2
    assert calls["n"] == 1


def test_cached_array_result_is_protected_from_user_mutation_when_copy_true():
    """Tests that cached array outputs are protected from mutation when copy is enabled."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.asarray(x, dtype=float) * 10.0

    cached_f = wrap_input_cache(f, copy=True)

    x = np.array([1.0, 2.0])
    y1 = cached_f(x)
    y1[0] = -999.0

    y2 = cached_f(x)

    np.testing.assert_allclose(y2, [10.0, 20.0])
    assert calls["n"] == 1


def test_copy_false_returns_same_cached_object_for_array_outputs():
    """Tests that copy=False returns the same cached array object."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.asarray(x, dtype=float) + 1.0

    cached_f = wrap_input_cache(f, copy=False)

    x = np.array([1.0, 2.0])
    y1 = cached_f(x)
    y2 = cached_f(x)

    assert y1 is y2
    assert calls["n"] == 1


def test_copy_false_allows_mutation_of_returned_cached_array():
    """Tests that copy=False allows mutation of the cached array output."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.asarray(x, dtype=float) + 1.0

    cached_f = wrap_input_cache(f, copy=False)

    x = np.array([1.0, 2.0])
    y1 = cached_f(x)
    y1[0] = -5.0

    y2 = cached_f(x)

    assert y2[0] == -5.0
    assert calls["n"] == 1


def test_scalar_outputs_are_not_copied_and_still_cache_correctly():
    """Tests that scalar outputs are cached correctly without copying."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x)))

    cached_f = wrap_input_cache(f, copy=True)

    y1 = cached_f([1.0, 2.0, 3.0])
    y2 = cached_f([1.0, 2.0, 3.0])

    assert y1 == 6.0
    assert y2 == 6.0
    assert calls["n"] == 1


def test_cache_clear_resets_cache_statistics_and_forces_recompute():
    """Tests that cache_clear resets cache statistics and forces recomputation."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x

    cached_f = wrap_input_cache(f)

    cached_f(2.0)
    cached_f(2.0)
    assert calls["n"] == 1
    assert cached_f.cache_info().hits == 1

    cached_f.cache_clear()

    info = cached_f.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0

    cached_f(2.0)
    assert calls["n"] == 2


def test_maxsize_one_evicts_previous_entry():
    """Tests that maxsize=1 evicts the previous cache entry."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x

    cached_f = wrap_input_cache(f, maxsize=1)

    cached_f(1.0)  # miss
    cached_f(2.0)  # miss, evicts 1.0
    cached_f(1.0)  # miss again

    assert calls["n"] == 3
    assert cached_f.cache_info().misses == 3


def test_wraps_preserves_function_metadata():
    """Tests that the cache wrapper preserves function metadata."""
    def my_function(x):
        """Test docstring."""
        return x

    cached_f = wrap_input_cache(my_function)

    assert cached_f.__name__ == "my_function"
    assert cached_f.__doc__ == "Test docstring."
