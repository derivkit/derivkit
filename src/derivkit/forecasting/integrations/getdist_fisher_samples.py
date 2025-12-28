"""Provides conversion helpers for Fisher–Gaussian forecasts and GetDist objects.

This module converts Fisher-matrix Gaussian approximations into
GetDist-compatible representations for plotting and analysis.

Two outputs are supported:

- An analytic Gaussian approximation via :class:`getdist.gaussian_mixtures.GaussianND`
  with mean ``theta0`` and covariance from the (pseudo-)inverse Fisher matrix.
- Monte Carlo samples drawn from the Fisher Gaussian as :class:`getdist.MCSamples`, with
  optional prior support hard bounds, and :attr:`getdist.MCSamples.loglikes`.

These helpers are intended for quick visualization (e.g. triangle plots) and
simple prior truncation without running an MCMC sampler.
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
    names: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    label: str = "Fisher (Gaussian)",
    rcond: float | None = None,
) -> GaussianND:
    """Returns :class:`getdist.gaussian_mixtures.GaussianND` for the Fisher Gaussian.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher matrix with shape ``(p, p)``.
        nnames: Optional parameter names (length ``p``).
            Defaults to ``["p" + str(x) for x in range(len(theta0))]``.
        labels: Optional parameter labels (length ``p``).
            Defaults to ``["p" + str(x) for x in range(len(theta0))]``.
        label: Label attached to the returned object.
        rcond: Cutoff passed to the Fisher (pseudo-)inverse when forming the covariance.

    Returns:
        A :class:`getdist.gaussian_mixtures.GaussianND` with mean ``theta0`` and covariance from ``fisher``.

    Raises:
        ValueError: If shapes or names/labels lengths are inconsistent.
    """
    theta0 = np.asarray(theta0, dtype=float)
    fisher = np.asarray(fisher, dtype=float)
    validate_fisher_shapes(theta0, fisher)

    n_params = int(theta0.size)

    default_names = [f"p{i}" for i in range(n_params)]
    default_labels = [rf"p_{{{i}}}" for i in range(n_params)]

    param_names = list(default_names if names is None else names)
    param_labels = list(default_labels if labels is None else labels)

    if len(param_names) != n_params or len(param_labels) != n_params:
        raise ValueError("names/labels must match number of parameters")

    covariance = fisher_to_cov(fisher, rcond=rcond)

    return GaussianND(
        mean=theta0,
        cov=covariance,
        names=param_names,
        labels=param_labels,
        label=label,
    )


def fisher_to_getdist_samples(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    names: Sequence[str],
    labels: Sequence[str],
    n_samples: int = 30_000,
    seed: int | None = None,
    kernel_scale: float = 1.0,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    store_loglikes: bool = True,
    label: str = "Fisher (samples)",
) -> MCSamples:
    """Draws samples from the Fisher Gaussian as :class:`getdist.MCSamples`.

    Samples are drawn from a multivariate Gaussian with mean ``theta0`` and
    covariance ``kernel_scale**2 * pinv(fisher)``. Optionally, samples are
    truncated by hard bounds and/or by a prior (via ``logprior`` or
    ``prior_terms``/``prior_bounds``).

    GetDist stores :attr:`getdist.MCSamples.loglikes` as ``-log(posterior)`` up to an additive
    constant. When ``store_loglikes=True`` we store:

        ``-log p(theta) = 0.5 * (theta-theta0)^T F (theta-theta0) - logprior(theta) + const``.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)`` with ``p`` parameters.
        fisher: Fisher matrix with shape ``(p, p)``.
        names: Parameter names (length ``p``).
        labels: Parameter labels (length ``p``).
        n_samples: Number of samples to draw.
        seed: Random seed.
        kernel_scale: Multiplicative scale applied to the Gaussian covariance.
        prior_terms: Prior specification for ``build_prior``.
        prior_bounds: Prior bounds for ``build_prior``.
        logprior: Custom log-prior callable. Mutually exclusive with
            ``prior_terms``/``prior_bounds``.
        hard_bounds: Hard bounds applied by rejection (samples outside are dropped).
            Mutually exclusive with encoding support via ``prior_terms``/``prior_bounds``.
        store_loglikes: If True, compute and store :attr:`getdist.MCSamples.loglikes`.
        label: Label for the returned :class:`getdist.MCSamples`.

    Returns:
        :class:`getdist.MCSamples` containing the retained samples and
        optional :attr:`getdist.MCSamples.loglikes`.

    Raises:
        ValueError: If shapes are inconsistent, names/labels lengths mismatch, or
            mutually exclusive options are provided.
        RuntimeError: If all samples are rejected by bounds or prior support.
    """
    theta0 = np.asarray(theta0, dtype=float)
    fisher = np.asarray(fisher, dtype=float)
    validate_fisher_shapes(theta0, fisher)

    n_params = int(theta0.size)
    if len(names) != n_params or len(labels) != n_params:
        raise ValueError("names/labels must match number of parameters")

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if hard_bounds is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError(
            "Ambiguous support: choose either `hard_bounds` or prior-based support "
            "via (`prior_terms`/`prior_bounds`)."
        )

    samples = kernel_samples_from_fisher(
        theta0,
        fisher,
        n_samples=n_samples,
        kernel_scale=float(kernel_scale),
        seed=seed,
    )

    samples = apply_parameter_bounds(samples, hard_bounds)
    if samples.shape[0] == 0:
        raise RuntimeError("All samples rejected by hard bounds (no samples left).")

    delta = samples - theta0[None, :]
    theta_quad = np.einsum("ni,ij,nj->n", delta, fisher, delta).astype(float, copy=False)

    loglikes: NDArray[np.floating] | None = None
    if store_loglikes:
        logprior_fn: Callable[[NDArray[np.floating]], float] | None = None
        if logprior is not None:
            logprior_fn = logprior
        elif prior_terms is not None or prior_bounds is not None:
            logprior_fn = build_prior(terms=prior_terms, bounds=prior_bounds)

        log_prior_vals: NDArray[np.floating] | None = None
        if logprior_fn is not None:
            log_prior_vals = np.array([float(logprior_fn(s)) for s in samples], dtype=float)
            keep = np.isfinite(log_prior_vals)
            samples = samples[keep]
            theta_quad = theta_quad[keep]
            log_prior_vals = log_prior_vals[keep]
            if samples.shape[0] == 0:
                raise RuntimeError("All samples rejected by the prior (logprior=-inf).")

        # GetDist convention: loglikes = -log(posterior) + const
        loglikes = 0.5 * theta_quad if log_prior_vals is None else (0.5 * theta_quad - log_prior_vals)

        # Shift by a constant for numerical stability (preserves relative weights).
        finite_ll = np.isfinite(loglikes)
        if np.any(finite_ll):
            loglikes = loglikes - np.min(loglikes[finite_ll])

    return MCSamples(
        samples=samples,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )
