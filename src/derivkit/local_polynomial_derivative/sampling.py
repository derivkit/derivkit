"""Sampling utilities for local polynomial derivative estimation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from derivkit.local_polynomial_derivative.local_poly_config import LocalPolyConfig

__all__ = ["build_samples"]


def build_samples(
    func: Callable[[float], Any],
    x0: float,
    config: LocalPolyConfig,
    n_workers: int = 1,
):
    """Builds sample points and evaluates the function there.

    Args:
        func:
            Function to evaluate. Takes a float and returns a scalar or np.ndarray.
        x0:
            Point around which to sample.
        config:
            LocalPolyConfig instance with sampling settings.
        n_workers:
            Number of parallel workers for function evaluation.

    Returns:
        xs:
            an array of sample points (shape (n_samples,)).
        ys:
            an array of function evaluations (shape (n_samples, n_components)).

    Raises:
        ValueError:
            if rel_steps in config is not a 1D non-empty sequence of floats.

    """
    rel_steps = np.asarray(config.rel_steps, float)
    if rel_steps.ndim != 1 or rel_steps.size == 0:
        raise ValueError("rel_steps must be a 1D non-empty sequence of floats.")

    if np.isscalar(rel_steps):
        rel_steps = np.array([rel_steps], dtype=float)

    if x0 == 0.0:
        xs = np.concatenate([-rel_steps, rel_steps])
    else:
        xs = x0 * (1.0 + np.concatenate([-rel_steps, rel_steps]))

    xs = np.unique(xs)
    xs.sort()

    if n_workers == 1:
        ys_list = [np.atleast_1d(func(float(x))) for x in xs]
    else:
        def _eval_one(x):
            return np.atleast_1d(func(float(x)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            ys_list = list(ex.map(_eval_one, xs))

    ys = np.stack(ys_list, axis=0)
    if ys.ndim != 2:
        ys = ys.reshape(ys.shape[0], -1)

    return xs, ys
