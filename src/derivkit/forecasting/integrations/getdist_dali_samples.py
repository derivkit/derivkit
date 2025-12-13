"""Convert DALI posterior to GetDist MCSamples via importance sampling or emcee."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.forecasting.expansions import logposterior_dali
from derivkit.utils.validate import validate_dali_shapes

from .sampling_utils import (
    apply_hard_bounds_mask,
    init_walkers_from_fisher,
    log_gaussian_proposal,
    proposal_samples_from_fisher,
)

Array = NDArray[np.floating]

__all__ = [
    "dali_to_getdist_importance",
    "dali_to_getdist_emcee",
    # optional compat aliases
    "dali_to_mcsamples_importance",
    "dali_to_mcsamples_emcee",
]


def dali_to_getdist_importance(
    theta0: Array,
    fisher: Array,
    g_tensor: Array,
    h_tensor: Array | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 200_000,
    proposal_scale: float = 1.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[Array], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (importance)",
):
    """GetDist MCSamples using importance sampling from a Fisher-Gaussian proposal.

    Target: log p(theta) = logprior(theta) - 0.5*Δχ²_DALI(theta) + const.
    Proposal: q(theta) = N(theta0, (proposal_scale^2) * pinv(F)).
    Weights: w ∝ p/q.

    GetDist expects: loglikes = -log(posterior) up to an additive constant.
    """
    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("Requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    samples = proposal_samples_from_fisher(
        theta0,
        fisher,
        nsamp=int(nsamp),
        proposal_scale=float(proposal_scale),
        seed=seed,
    )
    samples = apply_hard_bounds_mask(samples, hard_bounds)

    logpost = np.array(
        [
            logposterior_dali(
                s,
                theta0,
                fisher,
                g_tensor,
                h_tensor,
                convention=convention,
                prior_terms=prior_terms,
                prior_bounds=prior_bounds,
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

    logq = log_gaussian_proposal(
        samples,
        theta0,
        fisher,
        proposal_scale=float(proposal_scale),
    )

    # importance weights: w ∝ exp(logpost - logq)
    logw = logpost - logq
    logw = logw - np.max(logw)  # stabilize
    weights = np.exp(logw)

    # GetDist: loglikes = -log(posterior) up to constant
    loglikes = -logpost
    loglikes = loglikes - np.min(loglikes)  # safe shift

    return MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )


def dali_to_getdist_emcee(
    theta0: Array,
    fisher: Array,
    g_tensor: Array,
    h_tensor: Array | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsteps: int = 10_000,
    burn: int = 2_000,
    thin: int = 2,
    nwalkers: int | None = None,
    init_scale: float = 0.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[Array], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (emcee)",
):
    """Run emcee on the DALI posterior and return GetDist MCSamples."""
    try:
        import emcee
    except ImportError as e:
        raise ImportError("Requires `emcee` (pip install emcee).") from e

    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("Requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")
    if nwalkers is None:
        nwalkers = max(32, 8 * p)

    p0 = init_walkers_from_fisher(
        theta0,
        fisher,
        nwalkers=int(nwalkers),
        init_scale=float(init_scale),
        seed=seed,
        hard_bounds=hard_bounds,
    )

    def log_prob(th: Array) -> float:
        th = np.asarray(th, float)

        if hard_bounds is not None:
            for j, (lo, hi) in enumerate(hard_bounds):
                if lo is not None and th[j] < lo:
                    return -np.inf
                if hi is not None and th[j] > hi:
                    return -np.inf

        return float(
            logposterior_dali(
                th,
                theta0,
                fisher,
                g_tensor,
                h_tensor,
                convention=convention,
                prior_terms=prior_terms,
                prior_bounds=prior_bounds,
                logprior=logprior,
            )
        )

    # NOTE: avoid random_state= for emcee version-compat
    sampler = emcee.EnsembleSampler(int(nwalkers), int(p), log_prob)
    sampler.run_mcmc(p0, int(nsteps), progress=True)

    chain = sampler.get_chain(discard=int(burn), thin=int(thin))  # (n, nwalkers, p)
    logp = sampler.get_log_prob(discard=int(burn), thin=int(thin))  # log posterior

    # GetDist wants list-of-chains, with each walker as a chain
    chain_list = [chain[:, i, :] for i in range(chain.shape[1])]
    loglikes_list = [-logp[:, i] for i in range(logp.shape[1])]  # -log posterior

    return MCSamples(
        samples=chain_list,
        loglikes=loglikes_list,
        names=list(names),
        labels=list(labels),
        label=label,
    )


# ---------------------------------------------------------------------
# Optional backward-compatible aliases (if you want to keep old names)
# ---------------------------------------------------------------------
dali_to_mcsamples_importance = dali_to_getdist_importance
dali_to_mcsamples_emcee = dali_to_getdist_emcee
