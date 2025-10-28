"""Validation utilities for DerivativeKit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .sandbox import get_partial_function


def is_finite_and_differentiable(
    function: Callable[[float], Any],
    x: float,
    delta: float = 1e-5,
) -> bool:
    """Check that ``function`` is finite at ``x`` and ``x + delta``.

    Evaluates without exceptions and returns finite values at both points.

    Args:
      function: Callable ``f(x)`` returning a scalar or array-like.
      x: Probe point.
      delta: Small forward step.

    Returns:
      True if finite at both points; otherwise False.
    """
    f0 = np.asarray(function(x))
    f1 = np.asarray(function(x + delta))
    return np.isfinite(f0).all() and np.isfinite(f1).all()


def check_scalar_valued(function, theta0: np.ndarray, i: int, n_workers: int):
    """Helper used by ``build_gradient`` and ``build_hessian``.

    Args:
        function (callable): The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return a scalar observable value.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)

    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "build_gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )
