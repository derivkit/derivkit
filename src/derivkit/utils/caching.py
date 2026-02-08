"""Provides :func:`wrap_theta_cache_builtin`."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any

import numpy as np


def wrap_theta_cache_builtin(
    function: Callable[[np.ndarray], np.ndarray],
    *,
    number_decimal_places: int = 14,
    maxsize: int | None = 4096,
    copy: bool = True,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Creates a cache for function values.

    As part of the caching the input is truncated to a pre-set number
    of decimal places.

    Args:
        function: The function to be cached.
        number_decimal_places: The number of decimal places to include before
            truncating.
        maxsize: The size of the cache.
        copy: A flag that, when set to ``True``, causes the function
            to create a copy of the function input.

    Returns:
        A caching wrapper of :func:`function` if it is a ``Callable``.
        If :func:`function: is not a ``Callable`` it will be returned instead.
    """
    @lru_cache(maxsize=maxsize)
    def cached_wrapper(
        cachable_array: tuple[float, ...]
    ) -> np.ndarray[float, ...]:
        """Creates a function value cache for the given function."""
        theta = np.asarray(cachable_array, dtype=float)
        y = np.round(
            np.asarray(
                function(theta),
                dtype=float
            ).reshape(-1),
            number_decimal_places
        )
        return y

    @wraps(function)
    def wrapped(theta: np.ndarray) -> np.ndarray:
        "Wrapper that connects like functions to the same cache."""
        arr = cached_wrapper(tuple(theta))
        return arr.copy() if copy else arr

    # Ensure that the lru_cache atrtibutes are preserved.
    # Hijacked from https://stackoverflow.com/a/52332109
    wrapped.cache_info = cached_wrapper.cache_info
    wrapped.cache_clear = cached_wrapper.cache_clear

    return wrapped if isinstance(function, Callable) else function
