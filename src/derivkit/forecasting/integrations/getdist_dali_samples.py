"""Conversion utilities for DALI posteriors and GetDist ``MCSamples``.

This module provides helpers to turn a DALI-approximated posterior into
GetDist-compatible samples. Two sampling strategies are supported:

- Importance sampling from a Fisher-Gaussian proposal.
- MCMC sampling using ``emcee``.

The target posterior is evaluated via :func:`derivkit.forecasting.expansions.logposterior_dali`,
optionally including user-specified priors and hard parameter bounds.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Sequence

import emcee
import numpy as np
from getdist import MCSamples
from numpy.typing import NDArray

from derivkit.forecasting.expansions import logposterior_dali
from derivkit.forecasting.integrations.sampling_utils import (
    apply_parameter_bounds,
    init_walkers_from_fisher,
    kernel_samples_from_fisher,
    log_gaussian_kernel,
)
from derivkit.forecasting.priors.core import build_prior
from derivkit.utils.validate import validate_dali_shapes

__all__ = [
    "dali_to_getdist_importance",
    "dali_to_getdist_emcee",
]


def dali_to_getdist_importance(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 50_000,
    proposal_scale: float = 1.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (importance)",
):
    """Draws GetDist ``MCSamples`` via importance sampling from a Fisher-Gaussian proposal.

    Builds a weighted sample set for the DALI posterior using a multivariate normal
    proposal centered at ``theta0`` with covariance proportional to the (pseudo-)inverse
    Fisher matrix.

    The workflow is:

        1. Draw proposal samples ``theta ~ q(theta)`` where
           ``q = N(theta0, proposal_scale^2 * pinv(fisher))``.
        2. Optionally enforce hard bounds by rejecting samples outside ``hard_bounds``
           (or ``prior_bounds`` if ``hard_bounds`` is ``None``).
        3. Evaluate the target log-posterior ``log p(theta)`` using the DALI expansion and
           any configured priors.
        4. Compute importance weights ``w ~ exp(log p(theta) - log q(theta))``.
        5. Return a GetDist ``MCSamples`` instance with associated importance ``weights`` and
           a 1D ``loglikes`` array (GetDist convention: ``-log(posterior)`` up to an additive
           constant).

    Args:
        theta0: Fiducial parameter vector (shape ``(p,)``) for ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` (shape ``(p, p)``).
        g_tensor: DALI third-order tensor ``G`` (shape ``(p, p, p)``).
        h_tensor: Optional DALI fourth-order tensor ``H`` (shape ``(p, p, p, p)``).
        names: GetDist parameter names (length ``p``).
        labels: GetDist parameter labels (length ``p``).
        nsamp: Number of proposal samples to draw.
        proposal_scale: Scale factor applied to the proposal covariance.
        convention: DerivKit DALI convention passed through to :func:`logposterior_dali`.
        seed: Random seed for proposal sampling.
        prior_terms: Prior specification used to build a single ``logprior`` via
            :func:`derivkit.forecasting.priors.core.build_prior` (only if ``logprior`` is None).
        prior_bounds: Global hard bounds used to build ``logprior`` via
            :func:`derivkit.forecasting.priors.core.build_prior` (only if ``logprior`` is None).
        logprior: Optional custom log-prior ``logprior(theta)``; returns ``-np.inf`` to reject.
        hard_bounds: Optional hard support bounds; samples outside are discarded.
        label: Label string attached to the returned ``MCSamples``.

    Returns:
        A GetDist ``MCSamples`` object containing weighted samples approximating the
        DALI posterior, constructed from a single 2D sample array of shape ``(n, p)``
        with associated importance ``weights`` and a 1D ``loglikes`` array
        (GetDist convention: ``-log(posterior)`` up to an additive constant).

    Raises:
        ValueError: If tensor shapes or names/labels lengths are inconsistent, or if conflicting
            prior/support options are provided (e.g. `logprior` together with `prior_terms`/`prior_bounds`,
            or `hard_bounds` together with `prior_bounds`/`prior_terms`).
        RuntimeError: If all proposal samples are rejected by bounds, or if all samples are rejected
            after evaluating the DALI posterior and priors.
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if hard_bounds is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous support: you passed `hard_bounds` and also `prior_bounds`/`prior_terms`.\n"
            "Choose ONE support mechanism:\n"
            "  - use `hard_bounds` (set prior_bounds/prior_terms to None), OR\n"
            "  - encode support via the prior (set `hard_bounds=None`)."
        )

    effective_bounds = hard_bounds if hard_bounds is not None else prior_bounds

    # Compile prior once (prevents rebuilding inside logposterior_dali).
    if logprior is None:
        logprior = build_prior(terms=prior_terms, bounds=prior_bounds)

    samples = kernel_samples_from_fisher(
        theta0,
        fisher,
        n_samples=int(nsamp),
        kernel_scale=float(proposal_scale),
        seed=seed,
    )

    samples = apply_parameter_bounds(samples, effective_bounds)
    if samples.shape[0] == 0:
        raise RuntimeError("All proposal samples rejected by bounds (no samples left).")

    logpost = np.array(
        [
            logposterior_dali(
                s,
                theta0,
                fisher,
                g_tensor,
                h_tensor,
                convention=convention,
                logprior=logprior,
            )
            for s in samples
        ],
        dtype=float,
    )

    finite = np.isfinite(logpost)
    samples = samples[finite]
    logpost = logpost[finite]
    if samples.shape[0] == 0:
        raise RuntimeError("All proposal samples rejected by the posterior/prior (logpost=-inf).")

    logq = log_gaussian_kernel(
        samples,
        theta0,
        fisher,
        kernel_scale=float(proposal_scale),
    )
    logw = logpost - logq
    logw -= np.max(logw)
    weights = np.exp(logw)

    # GetDist: loglikes = -log(posterior) up to constant
    loglikes = -logpost
    loglikes -= np.min(loglikes)

    return MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )


def dali_to_getdist_emcee(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsteps: int = 10_000,
    burn: int = 2_000,
    thin: int = 2,
    n_walkers: int | None = None,
    init_scale: float = 0.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (emcee)",
):
    """Runs ``emcee`` on the DALI posterior and returns GetDist ``MCSamples``.

    Initializes walkers from a Fisher-based Gaussian around ``theta0`` and samples the
    DALI log-posterior using an ``emcee.EnsembleSampler``. Hard bounds are enforced by
    returning ``-np.inf`` outside ``hard_bounds``. Priors are applied through
    :func:`derivkit.forecasting.expansions.logposterior_dali`.

    The returned GetDist object is constructed as a list of chains, one per walker,
    with ``loglikes`` set to the GetDist convention ``-log(posterior)`` (up to an additive
    constant), not ``log L``.

    Args:
        theta0: Fiducial parameter vector (shape ``(p,)``) for ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` (shape ``(p, p)``).
        g_tensor: DALI third-order tensor ``G`` (shape ``(p, p, p)``).
        h_tensor: Optional DALI fourth-order tensor ``H`` (shape ``(p, p, p, p)``).
        names: GetDist parameter names (length ``p``).
        labels: GetDist parameter labels (length ``p``).
        nsteps: Total number of MCMC steps to run.
        burn: Number of initial steps to discard.
        thin: Thinning factor applied after burn-in.
        n_walkers: Number of walkers; defaults to ``max(32, 8 * p)``.
        init_scale: Scale controlling the initial scatter of walkers around ``theta0``.
        convention: DerivKit DALI convention passed through to :func:`logposterior_dali`.
        seed: Random seed for walker initialization.
        prior_terms: Prior specification passed through to :func:`logposterior_dali`.
        prior_bounds: Prior bounds passed through to :func:`logposterior_dali`.
        logprior: Optional custom log-prior ``logprior(theta)``; returns ``-np.inf`` to reject.
        hard_bounds: Optional hard support bounds; walkers outside are rejected.
        label: Label string attached to the returned ``MCSamples``.

    Returns:
        A GetDist ``MCSamples`` object containing MCMC samples of the DALI posterior,
        constructed from a list of chains (one per walker), where each chain has shape
        ``(n, p)`` and ``loglikes`` is provided as a matching list of 1D arrays
        (GetDist convention: ``-log(posterior)`` up to an additive constant).

    Raises:
        ValueError: If tensor shapes or names/labels lengths are inconsistent, or if conflicting
            prior/support options are provided.
        RuntimeError: If walker initialization fails (no valid starting points).
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")
    if n_walkers is None:
        n_walkers = max(32, 8 * p)

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if hard_bounds is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous support: you passed `hard_bounds` and also `prior_bounds`/`prior_terms`.\n"
            "Choose ONE support mechanism:\n"
            "  - use `hard_bounds` (set prior_bounds/prior_terms to None), OR\n"
            "  - encode support via the prior (set `hard_bounds=None`)."
        )

    effective_bounds = hard_bounds if hard_bounds is not None else prior_bounds

    # Compile prior once (prevents rebuilding inside logposterior_dali).
    if logprior is None:
        logprior = build_prior(terms=prior_terms, bounds=prior_bounds)

    p0 = init_walkers_from_fisher(
        theta0,
        fisher,
        n_walkers=int(n_walkers),
        init_scale=float(init_scale),
        seed=seed,
        hard_bounds=effective_bounds,
    )

    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 2 or p0.shape[0] == 0 or p0.shape[1] != p:
        raise RuntimeError(
            "Walker initialization failed: no valid starting points. "
            "Your bounds/prior support likely exclude the Fisher proposal around theta0. "
            "Try loosening bounds, increasing init_scale, or checking theta0 is inside support."
        )

    log_prob = partial(
        logposterior_dali,
        theta0=theta0,
        fisher=fisher,
        g_tensor=g_tensor,
        h_tensor=h_tensor,
        convention=convention,
        logprior=logprior,
    )

    sampler = emcee.EnsembleSampler(int(n_walkers), int(p), log_prob)
    sampler.run_mcmc(p0, int(nsteps), progress=True)

    chain = sampler.get_chain(discard=int(burn), thin=int(thin))
    logpost = sampler.get_log_prob(discard=int(burn), thin=int(thin))

    chain_list = [chain[:, i, :] for i in range(chain.shape[1])]
    loglikes_list = [-logpost[:, i] for i in range(logpost.shape[1])]

    return MCSamples(
        samples=chain_list,
        loglikes=loglikes_list,
        names=list(names),
        labels=list(labels),
        label=label,
    )
