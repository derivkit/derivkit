"""Gaussian sampling-kernel helpers derived from a Fisher matrix.

This module implements a Fisher-based Gaussian sampling distribution
``q`` centered at ``theta0`` with covariance::

    (sampling_scale**2) * pinv(F)

where ``F`` is the Fisher information matrix evaluated at ``theta0``.
In this approximation, ``F`` is the Hessian of ``-log L`` at ``theta0`` and
acts as the local inverse covariance.

This kernel is used to generate candidate points and, for importance sampling,
to evaluate ``log(q)`` when sampling posteriors approximated with Fisher or DALI.

It provides:

- construction of Fisher-based sampling covariances,
- stabilized Cholesky factorization for near-singular kernels,
- sampling and log-density evaluation of the kernel,
- fast hard-bounds rejection masks,
- MCMC walker initialization under hard bounds.

These utilities do not depend on a specific sampler implementation.
They provide kernel draws, log-densities, and hard-bounds filtering that are
called by the GetDist importance-sampling wrappers and by the emcee walker
initialization routine.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.linalg import solve_or_pinv
from derivkit.utils.validate import (
    validate_fisher_shapes,
    validate_square_matrix,
)

__all__ = [
    "kernel_cov_from_fisher",
    "stabilized_cholesky",
    "kernel_samples_from_fisher",
    "apply_parameter_bounds",
    "log_gaussian_kernel",
    "init_walkers_from_fisher",
    "fisher_to_cov",
]


def kernel_cov_from_fisher(
    fisher: NDArray[np.floating],
    *,
    kernel_scale: float,
) -> NDArray[np.float64]:
    """Returns the covariance of the Fisher-based Gaussian sampling kernel.

    This is a thin wrapper around
    :func:`derivkit.forecasting.integrations.sampling_utils.fisher_to_cov` that converts
    ``fisher`` to a covariance via pseudoinverse and applies the scaling ``kernel_scale^2``.

    The Fisher matrix is treated as the Hessian of ``-log L`` at ``theta0``.
    In this local approximation it acts as an inverse covariance, and
    ``kernel_scale`` controls the overall kernel width.
    A pseudoinverse is used so the covariance is defined even if ``fisher`` is singular.

    Args:
        fisher: Fisher information matrix with shape ``(p, p)``.
        kernel_scale: Multiplicative scale factor applied to the covariance.
            Increasing values widen the kernel; decreasing values narrow it.

    Returns:
        Kernel covariance matrix with shape ``(p, p)``, equal to
        ``(kernel_scale^2) * fisher_to_cov(fisher)``.

    Raises:
        ValueError: If ``fisher`` is not a square 2D array.
    """
    kernel_cov = fisher_to_cov(fisher)
    return (float(kernel_scale) ** 2) * kernel_cov


def stabilized_cholesky(cov: NDArray[np.floating]) -> NDArray[np.float64]:
    """Returns a Cholesky factor of a covariance matrix.

    This function computes a lower-triangular matrix ``L`` such that ``cov`` is
    approximately ``L @ L.T``, even when ``cov`` is nearly singular or only
    positive semi-definite.

    Args:
        cov: Covariance matrix with shape ``(p, p)`` with ``p`` parameters.

    Returns:
        A lower-triangular Cholesky factor ``L`` of the regularized covariance.

    Raises:
        ValueError: If ``cov`` is not a square 2D array.
    """
    cov_matrix = np.asarray(cov, dtype=float)

    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(f"cov must be square 2D, got {cov_matrix.shape}")

    n_params = cov_matrix.shape[0]
    trace_cov = float(np.trace(cov_matrix))
    regularization_scale = max(trace_cov, 1.0)
    jitter = 1e-12 * regularization_scale / max(n_params, 1)

    regularized_cov = cov_matrix + jitter * np.eye(n_params)
    cholesky_factor = np.linalg.cholesky(regularized_cov)
    return cholesky_factor


def kernel_samples_from_fisher(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    n_samples: int,
    kernel_scale: float,
    seed: int | None,
) -> NDArray[np.float64]:
    """Draws samples from a Fisher-based Gaussian sampling kernel.

    Samples are drawn from the Gaussian kernel density ``q(theta)`` with mean
    ``theta0`` and covariance ``(kernel_scale^2) * pinv(F)``::

        q(theta) = Normal(mean=theta0, cov=(kernel_scale^2) * pinv(F))

    ``F`` is the Fisher information matrix evaluated at ``theta0``.
    In this approximation, ``F`` acts as an inverse covariance and
    Normal represents the multivariate normal distribution.
    A pseudoinverse is used so the covariance is defined even if
    ``F`` is singular or ill-conditioned.

    Args:
        theta0: Kernel mean with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher information matrix with shape ``(p, p)``.
        n_samples: Number of samples to draw.
        kernel_scale: Multiplicative scale applied to the covariance.
            Increasing values widen the kernel; decreasing values narrow it.
        seed: Optional random seed for reproducible draws.

    Returns:
        Array of samples with shape ``(n_samples, p)``.
    """
    rng = np.random.default_rng(seed)

    theta0_vec = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    validate_fisher_shapes(theta0_vec, fisher_matrix)

    kernel_cov = kernel_cov_from_fisher(fisher_matrix, kernel_scale=float(kernel_scale))
    chol_lower = stabilized_cholesky(kernel_cov)

    n_params = theta0_vec.size
    standard_normals = rng.standard_normal((int(n_samples), n_params))

    kernel_mean = theta0_vec[None, :]
    noise = (chol_lower @ standard_normals.T).T
    return kernel_mean + noise


def apply_parameter_bounds(
    samples: NDArray[np.floating],
    parameter_bounds: Sequence[tuple[float | None, float | None]] | None,
) -> NDArray[np.float64]:
    """Applies per-parameter hard bounds to a set of samples.

    This function performs a fast, axis-aligned rejection step: any sample with
    at least one parameter value outside the specified bounds is discarded.
    It does not evaluate priors or compute weights.

    Args:
        samples: Sample array with shape ``(n_samples, p)`` with ``p`` parameters.
        parameter_bounds: Optional sequence of ``(lower, upper)`` bounds, one
            per parameter. Use ``None`` to indicate an unbounded side
            (e.g. ``(0.0, None)``). If ``None``, no filtering is applied.

    Returns:
        Samples satisfying all bounds, with shape ``(n_kept, p)``.
        ``n_kept`` may be zero.

    Raises:
        ValueError: If ``parameter_bounds`` is provided but does not have
            length ``p``.
    """
    sample_array = np.asarray(samples, dtype=np.float64)

    if parameter_bounds is None:
        return sample_array

    n_params = sample_array.shape[1]
    if len(parameter_bounds) != n_params:
        raise ValueError(
            f"parameter_bounds must have length {n_params}; got {len(parameter_bounds)}"
        )

    keep_mask = np.ones(sample_array.shape[0], dtype=bool)
    for param_index, (lower, upper) in enumerate(parameter_bounds):
        if lower is not None:
            keep_mask &= sample_array[:, param_index] >= lower
        if upper is not None:
            keep_mask &= sample_array[:, param_index] <= upper

    return sample_array[keep_mask]


def log_gaussian_kernel(
    samples: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    kernel_scale: float,
) -> NDArray[np.float64]:
    """Log-density of a Fisher-based Gaussian sampling kernel.

    Defines a Gaussian kernel ``q`` with mean ``theta0`` and covariance::

            (kernel_scale^2) @ pinv(F)

    where ``F`` is the Fisher information matrix. The covariance is formed with a
    pseudoinverse (allowing singular or ill-conditioned ``F``). A small diagonal
    jitter is added before evaluating the log-density so the calculation remains
    well-defined for near-singular covariances.

    Args:
        samples: Sample locations with shape ``(n_samples, p)``.
        theta0: Kernel mean with shape ``(p,)``.
        fisher: Fisher information matrix with shape ``(p, p)``.
        kernel_scale: Multiplicative scale applied to the covariance.

    Returns:
        Log of the kernel probability density ``log(q(theta))``, evaluated at each
        sample point, with shape ``(n_samples,)``.

    Raises:
        ValueError: If input shapes are incompatible.
        RuntimeError: If the jittered covariance is not positive-definite.
    """
    theta0_vec = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    validate_fisher_shapes(theta0_vec, fisher_matrix)
    sample_array = np.asarray(samples, dtype=np.float64)

    if sample_array.ndim != 2:
        raise ValueError(
            "samples must be 2D with shape (n_samples, p); "
            f"got shape {sample_array.shape}."
        )

    if sample_array.shape[1] != theta0_vec.size:
        raise ValueError(
            f"samples must have p={theta0_vec.size} columns; got {sample_array.shape[1]}."
        )

    cov_matrix = kernel_cov_from_fisher(fisher_matrix, kernel_scale=float(kernel_scale))

    n_params = theta0_vec.size
    trace_cov = float(np.trace(cov_matrix))
    trace_scale = max(trace_cov, 1.0)
    jitter = 1e-12 * trace_scale / max(n_params, 1)
    cov_matrix = cov_matrix + jitter * np.eye(n_params)

    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError("Kernel covariance is not positive-definite after adding jitter.")

    centered = sample_array - theta0_vec[None, :]
    solved = solve_or_pinv(
        cov_matrix,
        centered.T,
        rcond=1e-12,
        assume_symmetric=True,
        warn_context="log_gaussian_kernel",
    )

    quad_form = np.einsum("ij,ij->j", centered.T, solved)
    norm_const = n_params * np.log(2.0 * np.pi) + logdet
    log_gauss_density = -0.5 * (quad_form + norm_const)
    return log_gauss_density


def init_walkers_from_fisher(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    n_walkers: int,
    init_scale: float,
    seed: int | None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None,
) -> NDArray[np.float64]:
    """Returns initial MCMC walker positions from a Fisher-based Gaussian sampling kernel.

    Returns an array of walker positions centered at ``theta0`` with scatter set by
    a Fisher-derived covariance ``(init_scale^2) @ pinv(F)``. If ``hard_bounds``
    are provided, positions outside the bounds are rejected and additional candidates
    are generated until ``n_walkers`` positions are collected or a retry limit is reached.

    Args:
        theta0: Kernel mean with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher information matrix with shape ``(p, p)``.
        n_walkers: Number of walker positions to return.
        init_scale: Multiplicative scale applied to the kernel covariance.
        seed: Optional random seed for reproducible initialization.
        hard_bounds: Optional per-parameter ``(lower, upper)`` bounds. Use ``None``
            for an unbounded side.

    Returns:
        Array of initial positions with shape ``(n_walkers, p)``.

    Raises:
        ValueError: If ``theta0`` and ``fisher`` have incompatible shapes.
        ValueError: If ``hard_bounds`` is provided and does not have length ``p``.
        RuntimeError: If sufficient in-bounds positions cannot be generated within
            the retry limit.
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    if hard_bounds is not None and len(hard_bounds) != theta0.size:
        raise ValueError(
            f"hard_bounds must have length {theta0.size}; got {len(hard_bounds)}."
        )

    if hard_bounds is None:
        return kernel_samples_from_fisher(
            theta0, fisher, n_samples=int(n_walkers), kernel_scale=float(init_scale), seed=seed
        )

    out: list[np.ndarray] = []
    need = int(n_walkers)
    tries = 0
    while need > 0:
        tries += 1
        if tries > 50:
            raise RuntimeError(
                "Failed to initialize emcee walkers within hard_bounds. "
                "Try increasing init_scale or relaxing hard_bounds."
            )

        draw = kernel_samples_from_fisher(
            theta0, fisher, n_samples=max(need, int(n_walkers)), kernel_scale=float(init_scale), seed=None if seed is None else seed + tries
        )
        draw = apply_parameter_bounds(draw, hard_bounds)
        # apply_parameter_bounds returns an array with shape (n_kept, p).
        # If n_kept == 0, all proposed walkers fell outside the hard bounds,
        # so we retry with a fresh draw.
        if draw.shape[0] == 0:
            continue

        take = min(need, draw.shape[0])
        out.append(draw[:take])
        need -= take

    return np.vstack(out)


def fisher_to_cov(
        fisher: NDArray[np.floating],
        *,
        rcond: float | None = None
) -> NDArray[np.float64]:
    """Converts a Fisher matrix to a covariance matrix using pseudoinverse.

    Args:
        fisher: Fisher information matrix with shape ``(p, p)``.
        rcond: Cutoff ratio for small singular values in pseudoinverse.
            If ``None``, the default from ``np.linalg.pinv`` is used.

    Returns:
        Covariance matrix with shape ``(p, p)`` given by ``pinv(fisher)``.

    Raises:
        ValueError: If ``fisher`` is not a square 2D array.
    """
    fisher = validate_square_matrix(fisher, name="fisher")
    if rcond is None:
        return np.linalg.pinv(fisher, hermitian=True)
    return np.linalg.pinv(fisher, rcond=rcond, hermitian=True)
