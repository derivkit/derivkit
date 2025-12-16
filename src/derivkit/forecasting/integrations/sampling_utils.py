"""Gaussian proposal helpers for Fisher-based sampling.

This module implements a Fisher-Gaussian proposal centered at ``theta0`` with
covariance ``(proposal_scale**2) * pinv(F)``. It includes:

- stabilized Cholesky factorization for near-singular covariances,
- sampling and log-density evaluation for the proposal,
- fast hard-bounds rejection masks,
- robust emcee walker initialization under hard bounds.

These utilities are sampler-agnostic and are used by GetDist integration code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.validate import (
    validate_fisher_shapes,
    validate_square_matrix,
)

__all__ = [
    "proposal_cov_from_fisher",
    "stabilized_cholesky",
    "proposal_samples_from_fisher",
    "apply_hard_bounds_mask",
    "log_gaussian_proposal",
    "init_walkers_from_fisher",
    "fisher_to_cov",
]


def proposal_cov_from_fisher(
    fisher: NDArray[np.floating],
    *,
    proposal_scale: float,
) -> NDArray[np.floating]:
    """Builds a Gaussian proposal covariance from a Fisher matrix.

    The Fisher matrix is interpreted as an inverse covariance (curvature) at the
    expansion point. This function converts it to a covariance using a
    pseudoinverse (so it still works if the Fisher matrix is singular or
    ill-conditioned) and then applies an overall scaling factor to widen or
    tighten the proposal distribution.

    Args:
        fisher: Fisher information matrix with shape ``(p, p)`` for ``p``
            parameters.
        proposal_scale: Multiplicative scale factor applied to the covariance.
            Values > 1 widen the proposal; values < 1 narrow it.

    Returns:
        Proposal covariance matrix with shape ``(p, p)`` given by
        ``(proposal_scale**2) * pinv(fisher)``.

    Raises:
        ValueError: If ``fisher`` is not a square 2D array.
    """
    cov = fisher_to_cov(fisher)
    return (float(proposal_scale) ** 2) * cov


def stabilized_cholesky(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    """Computes a numerically stable Cholesky factor of a covariance matrix.

    This function returns a lower-triangular matrix ``L`` such that
    ``cov ≈ L @ L.T``. If the input covariance is nearly singular or only
    positive semi-definite due to numerical noise, a tiny diagonal
    regularization (jitter) is applied so the factorization succeeds.

    Args:
        cov: Covariance matrix (2D square array).

    Returns:
        Lower-triangular Cholesky factor of the (regularized) covariance.

    Raises:
        ValueError: If the input is not a square 2D array.
    """
    cov = np.asarray(cov, float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2D, got {cov.shape}")
    p = cov.shape[0]
    tr = float(np.trace(cov))
    scale = max(tr, 1.0)
    jitter = 1e-12 * scale / max(p, 1)
    return np.linalg.cholesky(cov + jitter * np.eye(p))


def proposal_samples_from_fisher(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    nsamp: int,
    proposal_scale: float,
    seed: int | None,
) -> NDArray[np.floating]:
    """Draws a Gaussian proposal samples using a Fisher matrix.

    This method generates samples from a multivariate normal distribution centered
    at ``theta0`` with covariance given by a (scaled) pseudoinverse of the Fisher
    matrix. The Fisher matrix is treated as local curvature (inverse covariance)
    at the expansion point, and ``proposal_scale`` controls the overall width of
    the proposal distribution.

    Args:
        theta0: Expansion point (mean of the proposal), shape ``(p,)`` for ``p`` parameters.
        fisher: Fisher information matrix with shape ``(p, p)``.
        nsamp: Number of samples to draw.
        proposal_scale: Multiplicative scale factor applied to the covariance.
            Values > 1 widen the proposal; values < 1 narrow it.
        seed: Optional RNG seed for reproducible draws.

    Returns:
        Array of proposal samples with shape ``(nsamp, p)``.

    Raises:
        ValueError: If shapes are inconsistent (e.g., ``fisher`` not square or
            incompatible with ``theta0``), or if the covariance cannot be
            factorized for sampling.
    """
    rng = np.random.default_rng(seed)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    cov = proposal_cov_from_fisher(fisher, proposal_scale=proposal_scale)
    lower_triangle = stabilized_cholesky(cov)

    p = theta0.size
    return theta0[None, :] + rng.standard_normal((int(nsamp), p)) @ lower_triangle.T


def apply_hard_bounds_mask(
    samples: NDArray[np.floating],
    hard_bounds: Sequence[tuple[float | None, float | None]] | None,
) -> NDArray[np.floating]:
    """Filters samples using simple per-parameter hard bounds.

    This is a lightweight, fast rejection step that removes any sample with at
    least one parameter outside the provided bounds. It is intentionally separate
    from the unified prior system: it does not evaluate prior terms or log-priors,
    and it does not attempt to compute weights. It only applies axis-aligned
    min/max cuts to the sample array.

    Args:
        samples: Array of samples with shape ``(n_samples, p)`` for ``p``
            parameters.
        hard_bounds: Optional sequence of ``(lower, upper)`` bounds, one per
            parameter. Use ``None`` for an unbounded side, e.g. ``(0.0, None)``.
            If ``None``, no filtering is applied and ``samples`` is returned.

    Returns:
        Subset of ``samples`` that satisfy all bounds, shape ``(n_kept, p)``.
        Note: ``n_kept`` may be zero.

    Raises:
        ValueError: If ``hard_bounds`` is provided but does not have length ``p``.
    """
    if hard_bounds is None:
        return samples

    p = samples.shape[1]
    if len(hard_bounds) != p:
        raise ValueError(f"hard_bounds must have length p={p}; got {len(hard_bounds)}")

    mask = np.ones(samples.shape[0], dtype=bool)
    for j, (lo, hi) in enumerate(hard_bounds):
        if lo is not None:
            mask &= samples[:, j] >= lo
        if hi is not None:
            mask &= samples[:, j] <= hi

    out = samples[mask]

    return out


def log_gaussian_proposal(
    samples: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    proposal_scale: float,
) -> NDArray[np.floating]:
    """Evaluates the log-density of a Fisher-based Gaussian proposal.

    This computes ``log q(theta)`` for a multivariate normal proposal
    distribution centered at ``theta0`` with covariance proportional to the
    (pseudo-)inverse Fisher matrix:

        q(theta) = N(theta0, (proposal_scale^2) * pinv(F)).

    The result is typically used for importance sampling, where you need the
    proposal density to form weights proportional to ``p(theta) / q(theta)``
    where ``p(theta)`` is the target density and ``q(theta)`` is the proposal.

    Args:
        samples: Array of sample locations with shape ``(n_samples, p)``
        for ``p`` parameters.
        theta0: Mean of the proposal distribution, shape ``(p,)``.
        fisher: Fisher information matrix, shape ``(p, p)``.
        proposal_scale: Multiplicative scale applied to the proposal covariance.

    Returns:
        Array of log proposal densities ``log q(samples)`` with shape
        ``(n_samples,)``.

    Raises:
        RuntimeError: If the proposal covariance cannot be treated as
            positive-definite for evaluating the Gaussian density.
        ValueError: If input shapes are incompatible.
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    samples = np.asarray(samples, float)

    cov = proposal_cov_from_fisher(fisher, proposal_scale=proposal_scale)

    # Match sampling jitter convention to keep logq consistent with draws
    p = theta0.size
    tr = float(np.trace(cov))
    scale = max(tr, 1.0)
    jitter = 1e-12 * scale / max(p, 1)
    cov = cov + jitter * np.eye(p)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError("Proposal covariance is not positive-definite (even after jitter).")

    inv_cov = np.linalg.inv(cov)
    diff_vec = samples - theta0[None, :]
    quad = np.einsum("...i,ij,...j->...", diff_vec, inv_cov, diff_vec)

    return -0.5 * (quad + p * np.log(2.0 * np.pi) + logdet)


def init_walkers_from_fisher(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    nwalkers: int,
    init_scale: float,
    seed: int | None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None,
) -> NDArray[np.floating]:
    """Initializes MCMC walker positions from a Fisher-based Gaussian proposal.

    This draws an initial ensemble of walker positions centered at ``theta0``,
    using a Gaussian with covariance proportional to the (pseudo-)inverse Fisher
    matrix. If ``hard_bounds`` are provided, the function enforces them by
    rejecting out-of-bounds draws and resampling until exactly ``nwalkers``
    valid positions are obtained (or until a retry limit is hit).

    This is intended for samplers like emcee, where you want a reasonable
    starting cloud near the Fisher approximation while still respecting simple
    parameter bounds.

    Args:
        theta0: Expansion point / proposal mean, shape ``(p,)`` for ``p``
            parameters.
        fisher: Fisher information matrix, shape ``(p, p)``.
        nwalkers: Number of walker positions to return.
        init_scale: Scale factor controlling the spread of the initial cloud.
        seed: Optional random seed for reproducibility.
        hard_bounds: Optional per-parameter ``(lower, upper)`` bounds used to
            reject invalid initial positions. Use ``None`` for an unbounded side.

    Returns:
        Array of initial walker positions with shape ``(nwalkers, p)``.

    Raises:
        RuntimeError: If valid walker positions cannot be generated within the
            retry limit (commonly due to overly tight bounds or too small
            ``init_scale``).
        ValueError: If input shapes are incompatible.
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    if hard_bounds is None:
        return proposal_samples_from_fisher(
            theta0, fisher, nsamp=int(nwalkers), proposal_scale=float(init_scale), seed=seed
        )

    # Rejection sampling with cap
    # because we don't know how many draws will be needed
    out: list[np.ndarray] = []
    need = int(nwalkers)
    tries = 0
    while need > 0:
        tries += 1
        if tries > 50:
            raise RuntimeError(
                "Failed to initialize emcee walkers within hard_bounds. "
                "Try increasing init_scale or relaxing hard_bounds."
            )

        draw = proposal_samples_from_fisher(
            theta0, fisher, nsamp=max(need, int(nwalkers)), proposal_scale=float(init_scale), seed=None if seed is None else seed + tries
        )
        draw = apply_hard_bounds_mask(draw, hard_bounds)
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
) -> NDArray[np.floating]:
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
    return np.linalg.pinv(fisher) if rcond is None else np.linalg.pinv(fisher, rcond=rcond)
