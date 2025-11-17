"""Finite difference extrapolation utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.extrapolation import (
    richardson_extrapolate,
    ridders_extrapolate,
)

__all__ = [
    "fixed_richardson_fd",
    "fixed_ridders_fd",
    "adaptive_richardson_fd",
    "adaptive_ridders_fd",
]


def fixed_richardson_fd(
    single_finite: Callable[[int, float, int, int], NDArray | float],
    *,
    order: int,
    stepsize: float,
    num_points: int,
    n_workers: int,
    levels: int,
    p: int = 2,
    r: float = 2.0,
    return_error: bool = False,
) -> NDArray | float | tuple[NDArray | float, NDArray | float]:
    """Returns fixed-level Richardson extrapolation for finite differences.

    Fixed level means we compute m base estimates with step sizes h, h/r, h/r^2, ..., h/r^(m-1)
    and then apply Richardson extrapolation to get the final estimate.

    Args:
        single_finite: Function that computes a single finite difference estimate.
        order: The order of the derivative to compute.
        stepsize: The initial step size h.
        num_points: Number of points in the finite difference stencil.
        n_workers: Number of parallel workers to use.
        levels: Number of levels (m) for Richardson extrapolation.
        p: The order of the leading error term in the finite difference approximation. Default is 2.
        r: The step-size reduction factor between successive levels (default is 2.0).
        return_error: Whether to return an error estimate along with the value (default is False).

    Returns:
        The Richardson-extrapolated finite difference estimate. If `return_error` is True,
        also returns an error estimate.
    """
    if levels < 2:
        raise ValueError("fixed_richardson_fd requires levels >= 2 for Richardson extrapolation.")

    base_values: list[NDArray | float] = []
    h = float(stepsize)

    for _ in range(levels):
        base_values.append(single_finite(order, h, num_points, n_workers))
        h /= r

    est = richardson_extrapolate(base_values, p=p, r=r)

    if not return_error:
        return est

    est_arr = np.asarray(est, dtype=float)

    # With only two levels, we can’t form a previous Richardson estimate; use zeros.
    if levels == 2:
        err = np.zeros_like(est_arr)
        return est, err

    prev_est = richardson_extrapolate(base_values[:-1], p=p, r=r)
    prev_arr = np.asarray(prev_est, dtype=float)
    err = np.abs(est_arr - prev_arr)
    return est, err


def fixed_ridders_fd(
    single_finite: Callable[[int, float, int, int], NDArray | float],
    *,
    order: int,
    stepsize: float,
    num_points: int,
    n_workers: int,
    levels: int,
    p: int = 2,
    r: float = 2.0,
    return_error: bool = False,
) -> NDArray | float | tuple[NDArray | float, NDArray | float]:
    """Returns a fixed-level Ridders extrapolation for finite differences.

    Fixed level means we compute m base estimates with step sizes h, h/r, h/r^2, ..., h/r^(m-1)
    and then apply Ridders extrapolation to get the final estimate.

    Args:
        single_finite:
            Function that computes a single finite difference estimate.
        order:
            The order of the derivative to compute.
        stepsize:
            The initial step size h.
        num_points:
            Number of points in the finite difference stencil.
        n_workers:
            Number of parallel workers to use.
        levels:
            Number of levels (m) for Ridders extrapolation.
        p:
            The order of the leading error term in the finite difference approximation (default is 2).
        r:
            The step-size reduction factor between successive levels (default is 2.0).
        return_error:
            Whether to return an error estimate along with the value (default is False).

    Returns:
        The Ridders-extrapolated finite difference estimate. If `return_error` is True,
        also returns an error estimate.
    """
    base_values: list[NDArray | float] = []
    h = float(stepsize)

    for _ in range(levels):
        base_values.append(single_finite(order, h, num_points, n_workers))
        h /= r

    value, err = ridders_extrapolate(base_values, r=r, p=p)
    return (value, err) if return_error else value


def adaptive_richardson_fd(
    single_finite: Callable[[int, float, int, int], NDArray | float],
    *,
    order: int,
    stepsize: float,
    num_points: int,
    n_workers: int,
    p: int,
    max_levels: int = 6,
    min_levels: int = 2,
    r: float = 2.0,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    return_error: bool = False,
) -> NDArray | float | tuple[NDArray | float, NDArray | float]:
    """Returns an adaptive Richardson extrapolation for finite differences.

    This function computes finite difference estimates at decreasing step sizes
    and applies Richardson extrapolation iteratively until convergence is achieved
    based on specified tolerances.

    Args:
        single_finite:
            Function that computes a single finite difference estimate.
        order:
            The order of the derivative to compute.
        stepsize:
            The initial step size h.
        num_points:
            Number of points in the finite difference stencil.
        n_workers:
            Number of parallel workers to use.
        p:
            The order of the leading error term in the finite difference approximation.
        max_levels:
            Maximum number of levels of extrapolation to perform (default is 6).
        min_levels:
            Minimum number of levels of extrapolation before checking for convergence.
            Default is 2.
        r:
            The step-size reduction factor between successive levels (default is 2.0).
        rtol:
            Relative tolerance for convergence (default is 1e-8).
        atol:
            Absolute tolerance for convergence (default is 1e-12).
        return_error:
            Whether to return an error estimate along with the value (default is False).

    Returns:
        The Richardson-extrapolated finite difference estimate.
    """
    base_values: list[NDArray | float] = []
    h = float(stepsize)

    best = None
    best_est = None
    last_err = None

    for level in range(max_levels):
        val = single_finite(order, h, num_points, n_workers)
        base_values.append(val)
        h /= r

        if level + 1 < min_levels:
            continue

        est = richardson_extrapolate(base_values, p=p, r=r)

        if best_est is None:
            best_est = est
            best = est
            continue

        est_arr = np.asarray(est, dtype=float)
        best_arr = np.asarray(best_est, dtype=float)

        diff = np.max(np.abs(est_arr - best_arr))
        scale = np.max([1.0, np.max(np.abs(est_arr)), np.max(np.abs(best_arr))])
        err_arr = np.full_like(est_arr, diff)

        if diff <= atol + rtol * scale:
            if not return_error:
                return est
            err = np.zeros_like(est_arr) if last_err is None else last_err
            return est, err

        if diff > 10.0 * (atol + rtol * scale):
            if not return_error:
                return best
            return best, err_arr

        last_err = err_arr
        best_est = est
        best = est

    if best_est is None:
        best_est = base_values[-1]
        last_err = np.zeros_like(np.asarray(best_est, dtype=float))

    if not return_error:
        return best_est
    return best_est, last_err


def adaptive_ridders_fd(
        single_finite: Callable[[int, float, int, int], NDArray | float],
        *,
        order: int,
        stepsize: float,
        num_points: int,
        n_workers: int,
        p: int,
        max_levels: int = 6,
        min_levels: int = 2,
        r: float = 2.0,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        return_error: bool = False,
) -> NDArray | float | tuple[NDArray | float, NDArray | float]:
    """Returns an adaptive Ridders extrapolation for finite differences.

    This function computes finite difference estimates at decreasing step sizes,
    building up a sequence of base values and repeatedly applying Ridders
    extrapolation until convergence is achieved based on specified tolerances.

    Args:
        single_finite:
            Function that computes a single finite difference estimate for a given
            derivative order and step size:
            ``single_finite(order, h, num_points, n_workers)``.
        order:
            The order of the derivative to compute.
        stepsize:
            The initial step size ``h``.
        num_points:
            Number of points in the finite difference stencil.
        n_workers:
            Number of parallel workers to use.
        p:
            The order of the leading error term in the finite difference
            approximation.
        max_levels:
            Maximum number of levels of extrapolation to perform (default is 6).
        min_levels:
            Minimum number of levels of extrapolation before checking for
            convergence (default is 2).
        r:
            Step-size reduction factor between successive levels (default is 2.0).
        rtol:
            Relative tolerance for convergence (default is 1e-8).
        atol:
            Absolute tolerance for convergence (default is 1e-12).
        return_error:
            Whether to return an error estimate along with the value
            (default is False).

    Returns:
        The Ridders-extrapolated finite difference estimate.
    """
    base_values: list[NDArray | float] = []
    h = float(stepsize)

    best_est: NDArray | float | None = None
    best_err: NDArray | float | None = None

    for level in range(max_levels):
        val = single_finite(order, h, num_points, n_workers)
        base_values.append(val)
        h /= r

        if level + 1 < min_levels:
            continue

        est, err = ridders_extrapolate(base_values, p=p, r=r)

        if best_est is None:
            best_est = est
            best_err = err
            continue

        est_arr = np.asarray(est, dtype=float)
        best_arr = np.asarray(best_est, dtype=float)

        diff = np.max(np.abs(est_arr - best_arr))
        scale = np.max([1.0, np.max(np.abs(est_arr)), np.max(np.abs(best_arr))])
        thresh = atol + rtol * scale
        diff_arr = np.full_like(est_arr, diff)

        # Converged: we accept latest estimate
        if diff <= thresh:
            if not return_error:
                return est
            out_err = err if err is not None else diff_arr
            return est, out_err

        # Diverged badly: we fall back to previous best
        if diff > 10.0 * thresh:
            if not return_error:
                return best_est
            out_err = best_err if best_err is not None else diff_arr
            return best_est, out_err

        best_est = est
        best_err = err

    if best_est is None:
        best_est = base_values[-1]
        best_err = np.zeros_like(np.asarray(best_est, dtype=float))

    if not return_error:
        return best_est
    return best_est, best_err
