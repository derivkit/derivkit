"""Utilities for evaluating Fisher and DALI likelihood expansions.

This module provides functional helpers to evaluate approximate likelihood
(or posterior) surfaces from forecast tensors:

- Fisher quadratic approximation (``F``)
- Doublet-DALI cubic/quartic corrections (``F``, ``G``, ``H``)

The functions here do not build tensors; they assume you already have
a Fisher matrix and (optionally) DALI tensors from :mod:`derivkit.forecasting`.

Conventions
-----------

We expose two conventions that are common in the codebase:

- ``convention="delta_chi2"``:
    Uses the standard DALI ``delta_chi2`` form::

        delta_chi2 = d.T @ F @ d + (1/3) einsum(G, d, d, d) + (1/12) einsum(H, d, d, d, d)

    and returns log posterior (up to a constant) as::

        log p = -0.5 * delta_chi2.

- ``convention="matplotlib_loglike"``:
    Matches the prefactors used in some matplotlib contour scripts::

        log p = -0.5 d.T @ F @ d - 0.5 einsum(G, d, d, d) - 0.125 einsum(H, d, d, d, d)

    which corresponds to::

        delta_chi2 = d.T @ F @ d + einsum(G, d, d, d) + 0.25 einsum(H, d, d, d, d)

    so that again ``log p = -0.5 * delta_chi2``.

GetDist convention
------------------

GetDist expects ``loglikes`` to be the negative log posterior::

    loglikes = -log(posterior)

up to an additive constant. Since this module defines::

    log(posterior) = -0.5 * delta_chi2 + const

a compatible choice for GetDist is therefore::

    loglikes = 0.5 * delta_chi2

(optionally shifted by a constant for numerical stability).

Notes
-----

- All log posterior values returned are defined up to an additive constant.
- Priors are optional and are applied via a single unified prior spec:
  (``prior_terms``, ``prior_bounds``) which is compiled using ``build_prior``.
- If no prior is provided, the functions return the *likelihood expansion*
  (i.e. improper flat prior, no hard cutoffs).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.forecasting.priors.core import build_prior
from derivkit.utils.validate import (
    validate_dali_shapes,
    validate_fisher_shapes,
)

__all__ = [
    "submatrix_fisher",
    "submatrix_dali",
    "delta_chi2_fisher",
    "delta_chi2_dali",
    "logposterior_fisher",
    "logposterior_dali",
]


def submatrix_fisher(
    fisher: NDArray[np.floating],
    idx: Sequence[int],
) -> NDArray[np.floating]:
    """Extracts a sub-Fisher matrix for a subset of parameter indices.

    Args:
        fisher: Full Fisher matrix of shape  ``(P, P)`` with ``P`` the number of parameters.
        idx: Sequence of parameter indices to extract.

    Returns:
        Sub-Fisher matrix (len(idx), len(idx)).

    Raises:
        ValueError: If `fisher` is not square 2D.
    """
    idx = list(idx)
    fisher = np.asarray(fisher, float)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"F must be square 2D, got {fisher.shape}")
    return fisher[np.ix_(idx, idx)]


def submatrix_dali(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    idx: Sequence[int],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating] | None,
]:
    """Extracts sub-DALI tensors for a subset of parameter indices.

    Args:
        theta0: Full expansion point with shape ``(P,)`` with ``P`` the number of parameters.
        fisher: Full Fisher matrix with shape ``(P, P)``.
        g_tensor: Full DALI cubic tensor with shape ``(P, P, P)``.
        h_tensor: Full DALI quartic tensor with shape ``(P, P, P, P)`` or ``None``.
        idx: Sequence of parameter indices to extract.

    Returns:
        A tuple ``(theta0_sub, f_sub, g_sub, h_sub)`` where each entry is restricted
        to the specified indices. Shapes are:
        
        - ``theta0_sub``: ``(len(idx),)``
        - ``f_sub``: ``(len(idx), len(idx))``
        - ``g_sub``: ``(len(idx), len(idx), len(idx))``
        - ``h_sub``: ``(len(idx), len(idx), len(idx), len(idx))`` or ``None``.
    """
    idx = list(idx)
    t0 = np.asarray(theta0, float)[idx]
    f2 = np.asarray(fisher, float)[np.ix_(idx, idx)]
    g2 = np.asarray(g_tensor, float)[np.ix_(idx, idx, idx)]
    h2 = None if h_tensor is None else np.asarray(h_tensor, float)[np.ix_(idx, idx, idx, idx)]
    return t0, f2, g2, h2


def delta_chi2_fisher(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
) -> float:
    """Computes a displacement chi-squared under the Fisher approximation.

    Args:
        theta: Parameter vector (trial/evaluation point).
        theta0: Parameter vector at which ``fisher`` was computed.
        fisher: Fisher matrix with shape ``(P, P)`` with ``P`` the number of parameters.

    Returns:
        Scalar delta chi-squared value.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    displacement = theta - theta0
    return float(displacement @ fisher @ displacement)


def _resolve_logprior(
    *,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None,
    logprior: Callable[[NDArray[np.floating]], float] | None,
) -> Callable[[NDArray[np.floating]], float] | None:
    """Resolve the logprior callable from either unified spec or direct callable.

    Rules:
      - If `logprior` is provided, `prior_terms`/`prior_bounds` must be None.
      - If `prior_terms` or `prior_bounds` is provided, compile via `build_prior`.
      - If none provided, return None (no prior).
    """
    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if logprior is None and (prior_terms is not None or prior_bounds is not None):
        return build_prior(terms=prior_terms, bounds=prior_bounds)

    return logprior


def logposterior_fisher(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    *,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
) -> float:
    """Computes the log posterior (up to a constant) under the Fisher approximation.

    If no prior is provided, this returns the Fisher log-likelihood expansion
    (improper flat prior, no hard cutoffs).

    The Fisher approximation corresponds to a purely quadratic delta chi^2 surface,

        delta_chi2 = d^T F d ,

    so the log posterior is

        log p = -0.5 * delta_chi2

    up to an additive constant. This normalization is equivalent to the
    ``convention="delta_chi2"`` used for DALI and should be used for all
    scientific results. In this interpretation, fixed delta_chi2 values correspond
    to fixed probability content (e.g. 68%, 95%) in parameter space, exactly
    as for a Gaussian likelihood.

    This is the same statistical interpretation used in standard Fisher
    forecasts and Gaussian likelihood analyses, making Fisher and DALI
    results directly comparable and statistically well-defined.

    Unlike the DALI case, there is no alternative normalization for the
    Fisher approximation: the likelihood is strictly Gaussian and fully
    described by the quadratic form.

    Args:
        theta: Parameter vector.
        theta0: Expansion point.
        fisher: Fisher matrix (P, P) with P parameters.
        prior_terms: See module docstring.
        prior_bounds: See module docstring.
        logprior: Optional custom log-prior callable. Returns ``-np.inf`` to reject.

    Returns:
        Scalar log posterior value (up to a constant).
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    lp_fn = _resolve_logprior(prior_terms=prior_terms, prior_bounds=prior_bounds, logprior=logprior)

    lp = 0.0
    if lp_fn is not None:
        lp = float(lp_fn(theta))
        if not np.isfinite(lp):
            return -np.inf

    displacement = theta - theta0
    chi2 = float(displacement @ fisher @ displacement)
    return lp - 0.5 * chi2


def delta_chi2_dali(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    *,
    convention: str = "delta_chi2",
) -> float:
    """Computes a displacement chi-squared under the DALI approximation.

    By default this uses ``convention="delta_chi2"``, which should be used for all
    scientific results. In this convention, the DALI log-posterior is treated as a
    standard likelihood, so fixed ``delta_chi2`` values correspond to fixed probability
    content (e.g. 68%, 95%) in parameter space. This is the same interpretation used for
    Gaussian likelihoods and Fisher forecasts, making results directly comparable
    and statistically well-defined.

    The alternative ``convention="matplotlib_loglike"`` follows an older plotting
    normalization based on equal log-likelihood height rather than probability mass.
    It is kept only as a visual sanity check or for reproducing legacy figures. For
    non-Gaussian DALI posteriors, it can change the apparent size and shape of
    contours and should not be used as the default.

    Args:
        theta: Parameter vector.
        theta0: Expansion point.
        fisher: Fisher matrix (P, P) with P parameters.
        g_tensor: DALI cubic tensor (P, P, P).
        h_tensor: DALI quartic tensor (P, P, P, P) or None.
        convention: Which normalization to use (``"delta_chi2"`` or
            ``"matplotlib_loglike"``).

    Returns:
        Scalar delta chi-squared value.
    """
    if convention not in ("delta_chi2", "matplotlib_loglike"):
        raise ValueError(f"Unknown convention='{convention}'. Supported: 'delta_chi2', 'matplotlib_loglike'.")

    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    displacement = theta - theta0
    quad = float(displacement @ fisher @ displacement)
    g3 = float(np.einsum(
        "ijk,i,j,k->",
        g_tensor,
        displacement,
        displacement,
        displacement
    )
    )
    h4 = 0.0 if h_tensor is None else float(np.einsum("ijkl,i,j,k,l->",
                                                      h_tensor,
                                                      displacement,
                                                      displacement,
                                                      displacement,
                                                      displacement
                                                      ))

    conv_dict = {
        "delta_chi2": quad + (1.0 / 3.0) * g3 + (1.0 / 12.0) * h4,
        "matplotlib_loglike": quad + g3 + 0.25 * h4,
    }
    return conv_dict[convention]


def logposterior_dali(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    *,
    convention: str = "delta_chi2",
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
) -> float:
    """Computes the log posterior (up to a constant) under the DALI approximation.

    If no prior is provided, this returns the DALI log-likelihood expansion
    (improper flat prior, no hard cutoffs).

    Args:
        theta: Parameter vector.
        theta0: Expansion point.
        fisher: Fisher matrix (P, P) with P parameters.
        g_tensor: DALI cubic tensor (P, P, P).
        h_tensor: DALI quartic tensor (P, P, P, P) or None.
        convention: See module docstring.
        prior_terms: See module docstring.
        prior_bounds: See module docstring.
        logprior: Optional custom log-prior callable. Returns ``-np.inf`` to reject.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    lp_fn = _resolve_logprior(prior_terms=prior_terms, prior_bounds=prior_bounds, logprior=logprior)

    lp = 0.0
    if lp_fn is not None:
        lp = float(lp_fn(theta))
        if not np.isfinite(lp):
            return -np.inf

    displacement = theta - theta0
    quad = float(displacement @ fisher @ displacement)
    g3 = float(np.einsum("ijk,i,j,k->",
                         g_tensor,
                         displacement,
                         displacement,
                         displacement
                         ))
    h4 = 0.0 if h_tensor is None else float(np.einsum("ijkl,i,j,k,l->",
                                                      h_tensor,
                                                      displacement,
                                                      displacement,
                                                      displacement,
                                                      displacement
                                                      ))

    if convention == "delta_chi2":
        chi2 = quad + (1.0 / 3.0) * g3 + (1.0 / 12.0) * h4
    elif convention == "matplotlib_loglike":
        chi2 = quad + g3 + 0.25 * h4
    else:
        raise ValueError(f"Unknown convention='{convention}'")

    return lp - 0.5 * chi2
