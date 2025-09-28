"""Finite-difference fallback helper for adaptive derivative estimation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from derivkit.finite.finite_difference import FiniteDifferenceDerivative

__all__ = ["fallback_fd"]


def fallback_fd(
    function: Callable[[float], Any],
    x0: float,
    order: int,
    n_workers: int = 1,
) -> np.ndarray:
    """Compute a finite-difference derivative as a fallback path.

    Emits a one-time ``RuntimeWarning`` and returns a 1D array for uniform
    handling with vector outputs.

    Args:
        function: Callable mapping a scalar ``x`` to a scalar or 1D array-like.
        x0: Expansion point at which to evaluate the derivative.
        order: Derivative order.
        n_workers: Number of workers used by the finite-difference backend.

    Returns:
        A NumPy array with shape ``(n_components,)`` (scalar becomes length-1).
    """
    warnings.warn("Falling back to finite difference derivative.", RuntimeWarning)
    fd = FiniteDifferenceDerivative(function=function, x0=x0)
    val = fd.differentiate(order=order, n_workers=n_workers)
    return np.atleast_1d(val)
