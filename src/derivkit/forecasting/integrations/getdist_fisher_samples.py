"""Conversion utilities for Fisher-Gaussian forecasts and GetDist outputs.

This module provides helpers to represent a Fisher forecast in GetDist:

- As an analytic Gaussian approximation via ``GaussianND`` (mean ``theta0``,
  covariance from the (pseudo-)inverse Fisher matrix).
- As Monte Carlo samples drawn from the Fisher Gaussian, optionally applying
  priors and hard bounds and storing GetDist-style ``loglikes``.

These utilities are intended for quick visualization (e.g. GetDist triangle
plots) and lightweight prior truncation without running an MCMC sampler.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from getdist import MCSamples
from getdist.gaussian_mixtures import GaussianND
from numpy.typing import NDArray

from derivkit.forecasting.integrations.sampling_utils import (
    apply_parameter_bounds,
    fisher_to_cov,
    kernel_samples_from_fisher,
)
from derivkit.forecasting.priors.core import build_prior
from derivkit.utils.validate import validate_fisher_shapes

__all__ = [
    "fisher_to_getdist_gaussiannd",
    "fisher_to_getdist_samples",
]


def fisher_to_getdist_gaussiannd(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    names: Sequence[str],
    labels: Sequence[str],
    label: str = "Fisher (Gaussian)",
    rcond: float | None = None,
):
    """Returns a GetDist GaussianND object for the Fisher Gaussian."""
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    cov = fisher_to_cov(fisher, rcond=rcond)

    return GaussianND(
        mean=theta0,
        cov=cov,
        names=list(names),
        labels=list(labels),
        label=label,
    )


def fisher_to_getdist_samples(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 30_000,
    seed: int | None = None,
    proposal_scale: float = 1.0,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    store_loglikes: bool = True,
    label: str = "Fisher (samples)",
):
    """Draws samples from the Fisher Gaussian and returns GetDist ``MCSamples``.

    GetDist convention:
        The ``loglikes`` field stores ``-log(posterior)`` (up to an additive constant).
        Despite the name, this is not necessarily ``-log L`` unless the prior is flat.

    In this function (when ``store_loglikes=True``), we store:
        ``-log posterior(theta) = 0.5 * d^T F d - logprior(theta) + const``,
        where ``d = (theta - theta0)``.

    Args:
        theta0: Expansion point (fiducial parameters) as a 1D array of length ``p``.
        fisher: Fisher information matrix as a 2D array with shape ``(p, p)``.
        names: List of parameter names (length ``p``).
        labels: List of parameter labels (length ``p``).
        nsamp: Number of samples to draw.
        seed: Random seed for reproducibility.
        proposal_scale: Scale factor for proposal covariance.
        prior_terms: Prior terms to build a prior from (see ``build_prior``).
        prior_bounds: Prior bounds to build a prior from (see ``build_prior``).
        logprior: Custom log-prior function (overrides ``prior_terms``/``prior_bounds``).
        hard_bounds: Hard bounds to apply to samples (outside samples are dropped).
        store_loglikes: Whether to compute and store loglikes in the output samples.
            If ``store_loglikes=True`` and a prior is provided, samples outside support are dropped
            and the *posterior* loglikes are stored (up to an additive constant).
        label: Label for the GetDist MCSamples object.

    Returns:
        A GetDist MCSamples object containing the drawn samples and GetDist-style ``loglikes``
        (i.e. ``-log posterior`` up to an additive constant, if ``store_loglikes=True``).

    Raises:
        ValueError: If shapes or names/labels lengths are inconsistent, or if conflicting
            prior/support options are provided (e.g. `logprior` together with `prior_terms`/`prior_bounds`,
            or `hard_bounds` together with `prior_bounds`/`prior_terms`).
        RuntimeError: If all samples are rejected by hard bounds, or if all samples are rejected by the prior.
    """
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

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

    # Draw from N(theta0, proposal_scale^2 * pinv(F))
    samples = kernel_samples_from_fisher(
        theta0, fisher, n_samples=int(nsamp), kernel_scale=float(proposal_scale), seed=seed
    )
    samples = apply_parameter_bounds(samples, hard_bounds)
    if samples.shape[0] == 0:
        raise RuntimeError("All samples rejected by hard bounds (no samples left).")

    d = samples - theta0[None, :]
    quad = np.einsum("ni,ij,nj->n", d, fisher, d).astype(float, copy=False)  # d^T F d

    loglikes = None
    if store_loglikes:
        lp_vals: NDArray[np.floating] | None = None
        if logprior is not None or prior_terms is not None or prior_bounds is not None:
            if logprior is None:
                logprior = build_prior(terms=prior_terms, bounds=prior_bounds)

            lp_vals = np.array([float(logprior(s)) for s in samples], dtype=float)
            finite = np.isfinite(lp_vals)
            samples = samples[finite]
            quad = quad[finite]
            lp_vals = lp_vals[finite]
            if samples.shape[0] == 0:
                raise RuntimeError("All samples rejected by the prior (logprior=-inf).")

        # -log posterior = 0.5*quad - logprior + const
        loglikes = 0.5 * quad if lp_vals is None else (0.5 * quad - lp_vals)

        # shift const for numerical stability
        loglikes = loglikes - np.nanmin(loglikes)

    return MCSamples(
        samples=samples,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )
