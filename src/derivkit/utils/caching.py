"""Provides :func:`wrap_input_cache`."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any

import numpy as np


def _normalize_cache_input(
    x: Any,
    *,
    number_decimal_places: int | None,
) -> tuple[Any, ...]:
    """Convert a numeric scalar or array-like input into a hashable cache key.

    Args:
        x: Function input. Can be a scalar, sequence, or NumPy array.
        number_decimal_places: Decimal places used to round floating-point
            inputs before key construction. If ``None``, exact values are used.

    Returns:
        Hashable tuple representing the normalized input.
    """
    if np.isscalar(x):
        scalar = np.asarray(x, dtype=float).item()
        value = float(scalar)
        if number_decimal_places is not None:
            value = round(value, number_decimal_places)
        return "scalar", value

    arr = np.asarray(x, dtype=float)
    if number_decimal_places is not None:
        arr = np.round(arr, number_decimal_places)

    flat = arr.ravel().astype(float, copy=False)
    return (
        "array",
        tuple(arr.shape),
        tuple(flat.tolist()),
    )


def _copy_if_needed(value: Any, *, copy: bool) -> Any:
    """Return the input value, copying NumPy arrays when requested.

    Args:
        value: Input value.
        copy: If ``True``, return a copy when ``value`` is a NumPy array.

    Returns:
        The original value, or a copied NumPy array if copying is enabled.
    """
    if not copy:
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def wrap_input_cache(
    function: Callable[[Any], Any],
    *,
    number_decimal_places: int | None = None,
    maxsize: int | None = 4096,
    copy: bool = True,
) -> Callable[[Any], Any]:
    """Wrap a callable with an input-based LRU cache.

    This wrapper supports both scalar and array-like numeric inputs. Cache keys
    are built from normalized inputs, with optional rounding applied before
    hashing. Cached NumPy arrays are always stored as copies to protect the
    cache from accidental mutation.

    Args:
        function: Function to cache.
        number_decimal_places: Decimal places used when constructing the cache
            key. If ``None``, exact input values are used.
        maxsize: Maximum size of the LRU cache.
        copy: If ``True``, return copies of cached NumPy array outputs.

    Returns:
        Wrapped callable with ``cache_info`` and ``cache_clear`` attached.
    """

    @lru_cache(maxsize=maxsize)
    def cached_wrapper(cache_key: tuple[Any, ...]) -> Any:
        """Evaluate ``function`` for a normalized cache key and store the result.

        Args:
            cache_key: Hashable normalized representation of the function input,
                as produced by :func:`_normalize_cache_input`.

        Returns:
            Function output for the reconstructed input. NumPy array outputs are
            stored as copies to avoid mutation of cached values.

        Raises:
            ValueError: If the cache key kind is not recognized.
        """
        kind = cache_key[0]

        if kind == "scalar":
            x = float(cache_key[1])
        elif kind == "array":
            shape = cache_key[1]
            flat_values = cache_key[2]
            x = np.asarray(flat_values, dtype=float).reshape(shape)
        else:
            raise ValueError(f"Unsupported cache key kind: {kind}")

        value = function(x)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    @wraps(function)
    def wrapped(x: Any) -> Any:
        """Call the cached wrapper on a raw input value.

        Args:
            x: Scalar or array-like input passed to ``function``.

        Returns:
            Cached or newly computed function output. NumPy array outputs are
            copied on return when ``copy=True``.
        """
        key = _normalize_cache_input(
            x,
            number_decimal_places=number_decimal_places,
        )
        value = cached_wrapper(key)
        return _copy_if_needed(value, copy=copy)

    wrapped.cache_info = cached_wrapper.cache_info
    wrapped.cache_clear = cached_wrapper.cache_clear

    return wrapped
