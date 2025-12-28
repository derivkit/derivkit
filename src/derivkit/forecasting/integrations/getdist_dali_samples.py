"""Provides GetDist sampling helpers for DALI approximate posteriors.

This module converts DALI-expanded posteriors into GetDist-compatible
:class:`getdist.MCSamples` for plotting and analysis.

Two backends are provided:

- Importance sampling using a Fisher–Gaussian kernel centered on ``theta0``.
- ``emcee`` ensemble MCMC targeting the same DALI log-posterior.

The target log-posterior is evaluated with
:meth:`derivkit.forecasting.expansions.logposterior_dali`, optionally including
user-defined priors and parameter support bounds.
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
    n_samples: int = 50_000,
    kernel_scale: float = 1.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (importance)",
) -> MCSamples:
    """Returns :class:`getdist.MCSamples` for a DALI posterior via importance sampling.

    The target posterior is evaluated with
    :meth:`derivkit.forecasting.expansions.logposterior_dali`. Samples are drawn from
    a Fisher–Gaussian kernel centered on ``theta0`` and reweighted by the difference
    between the target log-posterior and the kernel log-density.

    Notes:
        :attr:`getdist.MCSamples.loglikes` follows GetDist convention and stores ``-log(posterior)``
        up to an additive constant.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` for ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` with shape ``(p, p)``.
        g_tensor: DALI third-order tensor ``G`` with shape ``(p, p, p)``.
        h_tensor: Optional DALI fourth-order tensor ``H`` with shape ``(p, p, p, p)``.
        names: GetDist parameter names (length ``p``).
        labels: GetDist parameter labels (length ``p``).
        n_samples: Number of kernel samples to draw.
        kernel_scale: Scale factor applied to the Fisher covariance for the kernel.
        convention: DALI convention passed through to ``logposterior_dali``.
        seed: Random seed for kernel sampling.
        prior_terms: Prior terms used to build a prior via ``build_prior`` (ignored if
            ``logprior`` is provided).
        prior_bounds: Bounds used to build a prior via ``build_prior`` (ignored if
            ``logprior`` is provided).
        logprior: Optional custom log-prior ``logprior(theta)``.
        hard_bounds: Optional hard support bounds.
        label: Label attached to the returned :class:`getdist.MCSamples`.

    Returns:
        :class:`getdist.MCSamples` with importance ``weights`` and GetDist-style ``loglikes``.

    Raises:
        ValueError: If shapes are inconsistent or mutually exclusive options are provided.
        RuntimeError: If all samples are rejected by bounds or prior support.
    """
    fiducial = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    dali_g = np.asarray(g_tensor, dtype=float)
    dali_h = None if h_tensor is None else np.asarray(h_tensor, dtype=float)
    validate_dali_shapes(fiducial, fisher_matrix, dali_g, dali_h)

    n_params = int(fiducial.size)
    if len(names) != n_params or len(labels) != n_params:
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

    support_bounds = hard_bounds if hard_bounds is not None else prior_bounds

    # Compile prior once (this prevents rebuilding inside logposterior_dali)
    logprior_fn = logprior if logprior is not None else build_prior(terms=prior_terms, bounds=prior_bounds)

    kernel_samples = kernel_samples_from_fisher(
        fiducial,
        fisher_matrix,
        n_samples=int(n_samples),
        kernel_scale=float(kernel_scale),
        seed=seed,
    )

    kernel_samples = apply_parameter_bounds(kernel_samples, support_bounds)
    if kernel_samples.shape[0] == 0:
        raise RuntimeError("All kernel samples rejected by bounds (no samples left).")

    target_logpost = np.array(
        [
            logposterior_dali(
                theta,
                fiducial,
                fisher_matrix,
                dali_g,
                dali_h,
                convention=convention,
                logprior=logprior_fn,
            )
            for theta in kernel_samples
        ],
        dtype=float,
    )

    keep = np.isfinite(target_logpost)
    kernel_samples = kernel_samples[keep]
    target_logpost = target_logpost[keep]
    if kernel_samples.shape[0] == 0:
        raise RuntimeError("All kernel samples rejected by the posterior/prior (logpost=-inf).")

    kernel_logpdf = log_gaussian_kernel(
        kernel_samples,
        fiducial,
        fisher_matrix,
        kernel_scale=float(kernel_scale),
    )

    log_weights = target_logpost - kernel_logpdf
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)

    loglikes = -target_logpost
    loglikes -= np.min(loglikes)

    return MCSamples(
        samples=kernel_samples,
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
    n_steps: int = 10_000,
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
) -> MCSamples:
    """Returns :class:`getdist.MCSamples` from ``emcee`` sampling of a DALI posterior.

    The target log-posterior is evaluated with :meth:`derivkit.forecasting.expansions.logposterior_dali`.
    Walkers are initialized from a Fisher–Gaussian cloud around ``theta0`` and evolved with
    :class:`emcee.EnsembleSampler`. Optional priors and support bounds are applied through the target log-posterior.

    Notes:
        The returned :class:`getdist.MCSamples` is constructed from a list of per-walker chains.
        :attr:`getdist.MCSamples.loglikes` follows GetDist convention and stores ``-log(posterior)``
        up to an additive constant.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` with shape ``(p, p)``.
        g_tensor: DALI third-order tensor ``G`` with shape ``(p, p, p)``.
        h_tensor: Optional DALI fourth-order tensor ``H`` with shape ``(p, p, p, p)``.
        names: GetDist parameter names (length ``p``).
        labels: GetDist parameter labels (length ``p``).
        n_steps: Total number of MCMC steps.
        burn: Number of initial steps discarded as burn-in.
        thin: Thinning factor applied after burn-in.
        n_walkers: Number of walkers (defaults to ``max(32, 8 * p)``).
        init_scale: Initial scatter scale for walker initialization.
        convention: DALI convention passed through to :meth:`logposterior_dali`.
        seed: Random seed for walker initialization.
        prior_terms: Prior terms used to build a prior via ``build_prior`` (ignored if
            ``logprior`` is provided).
        prior_bounds: Bounds used to build a prior via ``build_prior`` (ignored if
            ``logprior`` is provided).
        logprior: Optional custom log-prior ``logprior(theta)``.
        hard_bounds: Optional hard support bounds.
        label: Label attached to the returned :class:`getdist.MCSamples`.

    Returns:
        :class:`getdist.MCSamples` containing per-walker chains and GetDist-style ``loglikes``.

    Raises:
        ValueError: If shapes are inconsistent or mutually exclusive options are provided.
        RuntimeError: If walker initialization fails (no valid starting points).
    """
    fiducial = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    dali_g = np.asarray(g_tensor, dtype=float)
    dali_h = None if h_tensor is None else np.asarray(h_tensor, dtype=float)
    validate_dali_shapes(fiducial, fisher_matrix, dali_g, dali_h)

    n_params = int(fiducial.size)
    if len(names) != n_params or len(labels) != n_params:
        raise ValueError("names/labels must match number of parameters")
    if n_walkers is None:
        n_walkers = max(32, 8 * n_params)

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if hard_bounds is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous support: you passed `hard_bounds` and also `prior_bounds`/`prior_terms`.\n"
            "Choose ONE support mechanism:\n"
            "  - use `hard_bounds` (set prior_bounds/prior_terms to None), OR\n"
            "  - encode support via the prior (set `hard_bounds=None`)."
        )

    support_bounds = hard_bounds if hard_bounds is not None else prior_bounds

    # Compile prior once (prevents rebuilding inside logposterior_dali).
    logprior_fn = logprior if logprior is not None else build_prior(terms=prior_terms, bounds=prior_bounds)

    walker_init = init_walkers_from_fisher(
        fiducial,
        fisher_matrix,
        n_walkers=int(n_walkers),
        init_scale=float(init_scale),
        seed=seed,
        hard_bounds=support_bounds,
    )

    walker_init = np.asarray(walker_init, dtype=float)
    if walker_init.ndim != 2 or walker_init.shape[0] == 0 or walker_init.shape[1] != n_params:
        raise RuntimeError(
            "Walker initialization failed: no valid starting points. "
            "Your bounds/prior support likely exclude the Fisher kernel around theta0. "
            "Try loosening bounds, increasing init_scale, or checking theta0 is inside support."
        )

    log_prob = partial(
        logposterior_dali,
        theta0=fiducial,
        fisher=fisher_matrix,
        g_tensor=dali_g,
        h_tensor=dali_h,
        convention=convention,
        logprior=logprior_fn,
    )

    sampler = emcee.EnsembleSampler(int(n_walkers), n_params, log_prob)
    sampler.run_mcmc(walker_init, int(n_steps), progress=True)

    chains = sampler.get_chain(discard=int(burn), thin=int(thin))
    log_posteriors = sampler.get_log_prob(discard=int(burn), thin=int(thin))

    chain_list = [chains[:, walker_idx, :] for walker_idx in range(chains.shape[1])]
    loglikes_list = [-log_posteriors[:, walker_idx] for walker_idx in range(log_posteriors.shape[1])]

    return MCSamples(
        samples=chain_list,
        loglikes=loglikes_list,
        names=list(names),
        labels=list(labels),
        label=label,
    )
