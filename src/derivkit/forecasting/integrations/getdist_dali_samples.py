"""Provides GetDist sampling helpers for DALI approximate posteriors.

This module converts DALI-expanded posteriors into GetDist-compatible
:class:`getdist.MCSamples` for plotting and analysis.

Two backends are provided:

- Importance sampling using a Fisher-Gaussian kernel centered on ``theta0``.
- ``emcee`` ensemble MCMC targeting the same DALI log-posterior.

The target log-posterior is evaluated with
:func:`derivkit.forecasting.expansions.logposterior_dali`, optionally including
user-defined priors and parameter support bounds.

Note: GetDist's ``loglikes`` field stores ``-log(posterior)``, not ``-log(likelihood)``.
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


def _resolve_support_bounds(
    *,
    n_params: int,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None,
) -> Sequence[tuple[float | None, float | None]] | None:
    """Computes the effective parameter-support bounds used by the sampler.

    This function combines two optional sources of support constraints:

    - ``prior_bounds``, which truncate the prior (and therefore the posterior support).
    - ``hard_bounds``, which impose an additional hard support constraint used for
      rejection/initialization.

    When both are provided, the returned bounds are their intersection, computed
    component-wise for each parameter. Each bound is a ``(low, high)`` pair where
    either endpoint may be ``None`` to indicate no constraint.

    Args:
        n_params: Number of parameters (expected length of any provided bounds sequence).
        prior_bounds: Optional per-parameter prior truncation bounds.
        hard_bounds: Optional per-parameter hard support bounds.

    Returns:
        The effective per-parameter bounds to use for support filtering, or ``None``
        if neither set of bounds is provided.

    Raises:
        ValueError: If a provided bounds sequence does not have length ``n_params``,
            or if the intersection of ``prior_bounds`` and ``hard_bounds`` is empty
            for any parameter (i.e., the resulting lower bound exceeds the upper bound).
    """
    if hard_bounds is not None and len(hard_bounds) != n_params:
        raise ValueError(
            f"hard_bounds must have length p={n_params}; got len(hard_bounds)={len(hard_bounds)}."
        )

    if prior_bounds is not None and len(prior_bounds) != n_params:
        raise ValueError(
            f"prior_bounds must have length p={n_params}; got len(prior_bounds)={len(prior_bounds)}."
        )

    if hard_bounds is None:
        return prior_bounds
    if prior_bounds is None:
        return hard_bounds

    bounds = []
    for i, ((hlo, hhi), (plo, phi)) in enumerate(zip(hard_bounds, prior_bounds, strict=True)):
        lo = None if (hlo is None and plo is None) else max(x for x in (hlo, plo) if x is not None)
        hi = None if (hhi is None and phi is None) else min(x for x in (hhi, phi) if x is not None)

        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"Empty support for parameter index {i}: "
                f"intersection gives ({lo}, {hi}). "
                "Check prior_bounds and hard_bounds."
            )

        bounds.append((lo, hi))

    return bounds


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

    The target log-posterior is evaluated with
    :func:`derivkit.forecasting.expansions.logposterior_dali`. Samples are drawn from
    a Fisher–Gaussian kernel centered on ``theta0`` and reweighted by the difference
    between the target log-posterior and the kernel log-density.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` for ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` with shape ``(p, p)``.
        g_tensor: Third-order DALI tensor with shape ``(p, p, p)``.
        h_tensor: Optional fourth-order DALI tensor with shape ``(p, p, p, p)``.
        names: Parameter names used to label the returned samples (length ``p``).
        labels: LaTeX-formatted parameter labels used to label the returned samples (length ``p``).
        n_samples: Number of importance samples to draw.
        kernel_scale: Scale factor applied to the Fisher covariance for the kernel.
        convention: DALI convention passed through to :func:`derivkit.forecasting.expansions.logposterior_dali`.
        seed: Random seed for kernel sampling.
        prior_terms: Optional prior term specifications used to build a prior via
            :func:`derivkit.forecasting.priors.core.build_prior`. Mutually exclusive with ``logprior``.
            Can be combined with ``prior_bounds`` to truncate prior support.
        prior_bounds: Optional per-parameter bounds passed to
            :func:`derivkit.forecasting.priors.core.build_prior` to truncate the prior support.
            If provided with no ``prior_terms``, this corresponds to a bounded-uniform (top-hat) prior.
            A bounded-uniform prior is proper (normalizable), unlike an unbounded flat prior which is improper.
            Mutually exclusive with ``logprior``.
        logprior: Optional custom log-prior ``logprior(theta)``. Mutually exclusive with
            ``prior_terms``/``prior_bounds``. If none of these are provided, a flat (typically improper)
            prior is used.
        hard_bounds: Optional per-parameter hard support bounds used for sample rejection/initialization.
            If provided with ``prior_bounds``, the effective support is their intersection.
        label: Label attached to the returned samples output (e.g., used by GetDist in plot legends/titles).

    Returns:
        :class:`getdist.MCSamples` containing the importance ``weights``.
        :attr:`getdist.MCSamples.loglikes` stores ``-log(posterior) (likelihood x prior)``
        up to an additive constant, consistent with GetDist's convention.

    Raises:
        ValueError: If shapes are inconsistent, mutually exclusive options are provided,
            or the effective support bounds are invalid (e.g., an empty intersection).
        RuntimeError: If all samples are rejected by bounds or prior support.
    """
    fiducial = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    dali_g = np.asarray(g_tensor, dtype=float)
    dali_h = None if h_tensor is None else np.asarray(h_tensor, dtype=float)
    validate_dali_shapes(fiducial, fisher_matrix, dali_g, dali_h)

    n_params = int(fiducial.size)
    n_names = len(names)
    n_labels = len(labels)
    if n_names != n_params:
        raise ValueError(f"names must have length p={n_params}; got len(names)={n_names}.")
    if n_labels != n_params:
        raise ValueError(f"labels must have length p={n_params}; got len(labels)={n_labels}.")

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous prior specification: pass either `logprior` or (`prior_terms` and/or `prior_bounds`), not both."
        )

    support_bounds = _resolve_support_bounds(
        n_params=n_params,
        prior_bounds=prior_bounds,
        hard_bounds=hard_bounds,
    )

    # Computing the prior once prevents rebuilding inside logposterior_dali
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
        raise RuntimeError("All kernel samples rejected by bounds.")

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
        raise RuntimeError(
            "All kernel samples were rejected: the target log-posterior was -inf for every sample "
            "(outside prior/support bounds or invalid model evaluation)."
        )

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

    ranges = None
    if support_bounds is not None:
        ranges = {n: [lo, hi] for n, (lo, hi) in zip(names, support_bounds, strict=True)}

    return MCSamples(
        samples=kernel_samples,
        weights=weights,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        ranges=ranges,
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

    The target log-posterior is evaluated with :func:`derivkit.forecasting.expansions.logposterior_dali`.
    Walkers are initialized from a Fisher–Gaussian cloud around ``theta0`` and evolved with
    :class:`emcee.EnsembleSampler`. Optional priors and support bounds are applied through the target log-posterior.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher matrix at ``theta0`` with shape ``(p, p)``.
        g_tensor: Third-order DALI tensor with shape ``(p, p, p)``.
        h_tensor: Optional fourth-order DALI tensor with shape ``(p, p, p, p)``.
        names: Parameter names used to label the returned samples (length ``p``).
        labels: LaTeX-formatted parameter labels used to label the returned samples (length ``p``).
        n_steps: Total number of MCMC steps.
        burn: Number of initial MCMC steps discarded as burn-in, allowing the chain to
            forget its initial starting positions before samples are retained.
        thin: Thinning factor applied after burn-in; only every ``thin``-th sample is kept
            to reduce autocorrelation in the chain.
        n_walkers: Number of walkers. Defaults to ``max(32, 8 * p)``.
        init_scale: Initial scatter scale for walker initialization.
        convention: DALI convention passed through to :func:`derivkit.forecasting.expansions.logposterior_dali`.
        seed: Random seed for walker initialization.
        prior_terms: Optional prior term specifications used to build a prior via
            :func:`derivkit.forecasting.priors.core.build_prior`. Mutually exclusive with ``logprior``.
            Can be combined with ``prior_bounds`` to truncate prior support.
        prior_bounds: Optional per-parameter bounds passed to
            :func:`derivkit.forecasting.priors.core.build_prior` to truncate the prior support.
            If provided with no ``prior_terms``, this corresponds to a bounded-uniform (top-hat) prior.
            A bounded-uniform prior is proper (normalizable), unlike an unbounded flat prior which is improper.
            Mutually exclusive with ``logprior``.
        logprior: Optional custom log-prior ``logprior(theta)``. Mutually exclusive with
            ``prior_terms``/``prior_bounds``. If none of these are provided, a flat (typically improper)
            prior is used.
        hard_bounds: Optional per-parameter hard support bounds used for sample rejection/initialization.
            If provided with ``prior_bounds``, the effective support is their intersection.
        label: Label attached to the returned samples output (e.g., used by GetDist in plot legends/titles).

    Returns:
        :class:`getdist.MCSamples` containing MCMC chains.
        attr:`getdist.MCSamples.loglikes` stores ``-log(posterior) (likelihood x prior)``
        up to an additive constant, consistent with GetDist's convention.

    Raises:
        ValueError: If shapes are inconsistent, mutually exclusive options are provided,
            or the effective support bounds are invalid (e.g., an empty intersection).
        RuntimeError: If walker initialization fails (no valid starting points).
    """
    fiducial = np.asarray(theta0, dtype=float)
    fisher_matrix = np.asarray(fisher, dtype=float)
    dali_g = np.asarray(g_tensor, dtype=float)
    dali_h = None if h_tensor is None else np.asarray(h_tensor, dtype=float)
    validate_dali_shapes(fiducial, fisher_matrix, dali_g, dali_h)

    n_params = int(fiducial.size)
    n_names = len(names)
    n_labels = len(labels)
    if n_names != n_params:
        raise ValueError(f"names must have length p={n_params}; got len(names)={n_names}.")
    if n_labels != n_params:
        raise ValueError(f"labels must have length p={n_params}; got len(labels)={n_labels}.")

    if n_walkers is None:
        n_walkers = max(32, 8 * n_params)

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous prior specification: pass either `logprior` or (`prior_terms` and/or `prior_bounds`), not both."
        )

    support_bounds = _resolve_support_bounds(
        n_params=n_params,
        prior_bounds=prior_bounds,
        hard_bounds=hard_bounds,
    )

    # Compute prior once, as this prevents rebuilding inside logposterior_dali.
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
            "The provided bounds/prior support likely exclude the Fisher kernel around theta0. "
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

    ranges = None
    if support_bounds is not None:
        ranges = {n: [lo, hi] for n, (lo, hi) in zip(names, support_bounds, strict=True)}

    return MCSamples(
        samples=chain_list,
        loglikes=loglikes_list,
        names=list(names),
        labels=list(labels),
        ranges=ranges,
        label=label,
    )
