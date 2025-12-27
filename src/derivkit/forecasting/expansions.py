"""Utilities for evaluating Fisher and DALI likelihood expansions.

This module provides functional helpers to evaluate approximate likelihood
(or posterior) surfaces from forecast tensors:

- Fisher quadratic approximation (``F``)
- Doublet-DALI cubic/quartic corrections (``F``, ``G``, ``H``)

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

Notes:
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
) -> NDArray[np.float64]:
    """Extracts a sub-Fisher matrix for a subset of parameter indices.

    The submatrix is constructed by selecting rows and columns of ``fisher``
    at the indices in ``idx`` using ``np.ix_`` such that
    ``F_sub[a, b] = fisher[idx[a], idx[b]]``.

    The indices in ``idx`` may be any subset and any order. They do not need to
    correspond to a contiguous block in the original matrix. For example, selecting
    two parameters that appear at opposite corners of the full Fisher matrix
    produces the full 2x2 Fisher submatrix for those parameters, including their
    off-diagonal correlation.

    This operation is useful for extracting a lower-dimensional slice of a Fisher
    matrix for plotting or evaluation while holding all other parameters fixed at
    their expansion values. It represents a slice through parameter space rather
    than a marginalization. Marginalized constraints instead require operating on
    the covariance matrix obtained by inverting the full Fisher matrix.

    Args:
        fisher: Full Fisher matrix of shape ``(P, P)`` with ``P`` the number of parameters.
        idx: Sequence of parameter indices to extract.

    Returns:
        Sub-Fisher matrix ``(len(idx), len(idx))``.

    Raises:
        ValueError: If ``fisher`` is not square 2D.
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
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
]:
    """Extracts sub-DALI tensors for a subset of parameter indices.

    The tensors are constructed by selecting entries of ``theta0`` and the
    corresponding rows and columns of the Fisher, cubic, and quartic tensors
    using the indices in ``idx``. The indices may be any subset and any order
    and do not need to correspond to a contiguous block.

    This operation is useful for evaluating a DALI expansion on a lower-dimensional
    parameter subspace while holding all other parameters fixed at their expansion
    values. It represents a slice through parameter space rather than a marginalization.

    Args:
        theta0: Full expansion point with shape ``(P,)`` with ``P`` the number of parameters.
        fisher: Full Fisher matrix with shape ``(P, P)``.
        g_tensor: Full DALI cubic tensor with shape ``(P, P, P)``.
        h_tensor: Full DALI quartic tensor with shape ``(P, P, P, P)`` or ``None``.
        idx: Sequence of parameter indices to extract.

    Returns:
        A tuple ``(theta0_sub, f_sub, g_sub, h_sub)`` where each entry is selected
        using the specified indices. Shapes are:

        - ``theta0_sub``: ``(len(idx),)``
        - ``f_sub``: ``(len(idx), len(idx))``
        - ``g_sub``: ``(len(idx), len(idx), len(idx))``
        - ``h_sub``: ``(len(idx), len(idx), len(idx), len(idx))`` or ``None``.

    Raises:
        IndexError: If any index in ``idx`` is out of bounds.
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
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(P, P)`` with ``P`` the number of parameters.

    Returns:
        The scalar delta chi-squared value between ``theta`` and ``theta_0``.
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
    """Determines which log-prior to use for likelihood expansion evaluation.

    This helper allows callers to specify a prior in one of two ways: either by passing
    a pre-built ``logprior(theta)`` callable directly, or by providing a lightweight
    prior specification (``prior_terms`` and/or ``prior_bounds``) that is compiled
    internally using :meth:`derivkit.forecasting.priors.core.build_prior`.

    Only one of these input styles may be used at a time. Providing both results in a
    ``ValueError``. If neither is provided, the function returns ``None``, indicating
    that no prior is applied.

    Args:
        prior_terms: Prior term specification passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
        logprior: Optional custom log-prior callable. Returns ``-np.inf`` to reject

    Returns:
        A function that computes the log-prior contribution to the posterior, or
        ``None`` if the likelihood should be evaluated without a prior.
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
    """Computes the log posterior under the Fisher approximation.

    The returned value is defined up to an additive constant in log space.
    This corresponds to an overall multiplicative normalization of the posterior
    density in probability space.

    If no prior is provided, this returns the Fisher log-likelihood expansion
    (improper flat prior, no hard cutoffs).

    The Fisher approximation corresponds to a purely quadratic ``delta_chi2`` surface::

        delta_chi2 = d.T @ F @ d

    so the log posterior is::

        log p = -0.5 * delta_chi2

    This normalization is equivalent to the ``convention="delta_chi2"`` used for DALI.
    In this interpretation, fixed ``delta_chi2`` values correspond to fixed probability content
    (e.g. 68%, 95%) in parameter space, as for a Gaussian likelihood.
    See :meth:`derivkit.forecasting.expansions.delta_chi2_dali` for the corresponding
    DALI definition of ``delta_chi2`` and its supported conventions.

    Unlike the DALI case, there is no alternative normalization for the Fisher
    approximation: the likelihood is strictly Gaussian and fully described by the
    quadratic form.

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(P, P)`` with ``P`` the number of parameters.
        prior_terms: Prior term specification passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
        logprior: Optional custom log-prior callable. Returns ``-np.inf`` to reject.

    Returns:
        Scalar log posterior value, defined up to an additive constant.
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
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix ``(P, P)`` with ``P`` the number of parameters.
        g_tensor: DALI cubic tensor with shape ``(P, P, P)``.
        h_tensor: DALI quartic tensor ``(P, P, P, P)`` or ``None``.
        convention: The normalization to use (``"delta_chi2"`` or
            ``"matplotlib_loglike"``).

    Returns:
        The scalar delta chi-squared value.
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
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(P, P)`` with ``P`` the number of parameters.
        g_tensor: DALI cubic tensor ``(P, P, P)``.
        h_tensor: DALI quartic tensor ``(P, P, P, P)`` or ``None``.
        convention: The normalization to use (``"delta_chi2"`` or
            ``"matplotlib_loglike"``).
        prior_terms: Prior term specification passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :meth:`derivkit.forecasting.priors.core.build_prior`.
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
