from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.validate import validate_fisher_shapes

from .sampling_utils import (
    apply_hard_bounds_mask,
    proposal_samples_from_fisher,
)

Array = NDArray[np.floating]

__all__ = [
    "fisher_to_cov",
    "fisher_to_gaussiannd",
    "fisher_to_getdist_samples",
    # compat alias
    "fisher_to_mcsamples",
]


def fisher_to_cov(fisher: Array, *, rcond: float | None = None) -> Array:
    """Convert Fisher to covariance via (pseudo-)inverse."""
    fisher = np.asarray(fisher, float)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"fisher must be square 2D, got {fisher.shape}")
    return np.linalg.pinv(fisher) if rcond is None else np.linalg.pinv(fisher, rcond=rcond)


def fisher_to_gaussiannd(
    theta0: Array,
    fisher: Array,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    label: str = "Fisher (Gaussian)",
    rcond: float | None = None,
):
    """Return a GetDist GaussianND object for the Fisher Gaussian."""
    try:
        from getdist.gaussian_mixtures import GaussianND
    except ImportError as e:
        raise ImportError("Requires `getdist` (pip install getdist).") from e

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
    theta0: Array,
    fisher: Array,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 200_000,
    seed: int | None = None,
    proposal_scale: float = 1.0,
    # priors: either unified spec or direct callable
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[Array], float] | None = None,
    # optional fast reject (pre-prior)
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    # output controls
    store_loglikes: bool = True,
    label: str = "Fisher (samples)",
):
    """Draw samples from the Fisher Gaussian and return GetDist MCSamples.

    If a prior is provided, samples outside support are dropped and the *posterior*
    loglikes are stored (up to an additive constant).

    GetDist convention:
      loglikes = -log(posterior) (up to const)

    For Fisher:
      -log posterior = 0.5 * d^T F d  - logprior + const
    """
    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("Requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    # Draw from N(theta0, proposal_scale^2 * pinv(F))
    samples = proposal_samples_from_fisher(
        theta0, fisher, nsamp=int(nsamp), proposal_scale=float(proposal_scale), seed=seed
    )
    samples = apply_hard_bounds_mask(samples, hard_bounds)

    d = samples - theta0[None, :]
    quad = np.einsum("ni,ij,nj->n", d, fisher, d).astype(float, copy=False)  # d^T F d

    loglikes = None
    if store_loglikes:
        if logprior is not None and (prior_terms is not None or prior_bounds is not None):
            raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

        lp_vals: Array | None = None
        if logprior is not None or prior_terms is not None or prior_bounds is not None:
            if logprior is None:
                # local import avoids cycles
                from derivkit.forecasting.priors.core import build_prior
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


def fisher_to_mcsamples(*args, **kwargs):
    return fisher_to_getdist_samples(*args, **kwargs)
