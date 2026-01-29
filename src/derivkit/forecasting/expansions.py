"""Utilities for evaluating Fisher and DALI likelihoods expansions.

This module provides functional helpers to evaluate approximate likelihoods
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
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.forecasting.priors_core import build_prior
from derivkit.utils.validate import (
    validate_dali_shapes,
    validate_fisher_shapes,
)

__all__ = [
    "build_submatrix_fisher",
    "build_submatrix_dali",
    "build_delta_chi2_fisher",
    "build_delta_chi2_dali",
    "build_logposterior_fisher",
    "build_logposterior_dali",
    "build_subspace",
]


def build_submatrix_fisher(
    fisher: NDArray[np.floating],
    idx: Sequence[int],
) -> NDArray[np.float64]:
    """Extracts a sub-Fisher matrix for a subset of parameter indices.

    The submatrix is constructed by selecting rows and columns of ``fisher``
    at the indices such that ``F_sub[a, b] = fisher[idx[a], idx[b]]``.

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
        fisher: Full Fisher matrix of shape ``(p, p)`` with ``p`` the number of parameters.
        idx: Sequence of parameter indices to extract.

    Returns:
        Sub-Fisher matrix ``(len(idx), len(idx))``.

    Raises:
        ValueError: If ``fisher`` is not square 2D.
        TypeError: If ``idx`` contains non-integer indices.
        IndexError: If any index in ``idx`` is out of bounds.
    """
    idx = list(idx)
    fisher = np.asarray(fisher, float)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"fisher must be square 2D, got {fisher.shape}")

    p = int(fisher.shape[0])

    if not all(isinstance(i, (int, np.integer)) for i in idx):
        raise TypeError("idx must contain integer indices")

    if any((i < 0) or (i >= p) for i in idx):
        raise IndexError(f"idx contains out-of-bounds indices for p={p}: {idx}")

    return fisher[np.ix_(idx, idx)]


def build_submatrix_dali(
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
        theta0: Full expansion point with shape ``(p,)`` with ``p`` the number of parameters.
        fisher: Full Fisher matrix with shape ``(p, p)``.
        g_tensor: Full DALI cubic tensor with shape ``(p, p, p)``.
        h_tensor: Full DALI quartic tensor with shape ``(p, p, p, p)`` or ``None``.
        idx: Sequence of parameter indices to extract.

    Returns:
        A tuple ``(theta0_sub, f_sub, g_sub, h_sub)`` where each entry is selected
        using the specified indices. Shapes are:

        - ``theta0_sub``: ``(len(idx),)``
        - ``f_sub``: ``(len(idx), len(idx))``
        - ``g_sub``: ``(len(idx), len(idx), len(idx))``
        - ``h_sub``: ``(len(idx), len(idx), len(idx), len(idx))`` or ``None``.

    Raises:
        ValueError: If input tensors have invalid shapes.
        TypeError: If ``idx`` contains non-integer indices.
        IndexError: If any index in ``idx`` is out of bounds.
    """
    idx = list(idx)

    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)

    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = int(theta0.shape[0])

    if not all(isinstance(i, (int, np.integer)) for i in idx):
        raise TypeError("idx must contain integer indices")

    if any((i < 0) or (i >= p) for i in idx):
        raise IndexError(f"idx contains out-of-bounds indices for p={p}: {idx}")

    t0 = theta0[idx]
    f2 = fisher[np.ix_(idx, idx)]
    g2 = g_tensor[np.ix_(idx, idx, idx)]
    h2 = None if h_tensor is None else h_tensor[np.ix_(idx, idx, idx, idx)]
    return t0, f2, g2, h2


def build_subspace(
    idx: Sequence[int],
    *,
    fisher: NDArray[np.floating],
    theta0: NDArray[np.floating],
    g_tensor: NDArray[np.floating] | None = None,
    h_tensor: NDArray[np.floating] | None = None,
) -> dict[str, Any]:
    """Extracts a parameter subspace for Fisher or DALI expansions.

    This function selects the parameters specified by ``idx`` and extracts the
    corresponding entries of the expansion point and tensors.

    Args:
        idx: Sequence of parameter indices to extract.
        fisher: Full Fisher matrix of shape ``(p, p)`` with ``p`` the number of parameters.
        theta0: Expansion point of shape ``(p,)``.
        g_tensor: Optional DALI cubic tensor of shape ``(p, p, p)``.
        h_tensor: Optional DALI quartic tensor of shape ``(p, p, p, p)`` or ``None``.

    Returns:
        Dictionary containing the sliced objects. Let ``m = len(idx)``. The Fisher
        submatrix has shape ``(m, m)`` and the expansion point has shape ``(m,)``.
        If ``g_tensor`` is provided, the cubic tensor has shape ``(m, m, m)``.
        If ``h_tensor`` is provided, the quartic tensor has shape ``(m, m, m, m)``.
        Keys use the same names as the inputs.

    Raises:
        TypeError: If ``idx`` contains non-integer indices.
        IndexError: If any index in ``idx`` is out of bounds.
        ValueError: If the provided arrays have incompatible shapes, or if ``h_tensor``
            is provided without ``g_tensor``.
    """
    idx = list(idx)
    if not all(isinstance(i, (int, np.integer)) for i in idx):
        raise TypeError("idx must contain integer indices")

    fisher = np.asarray(fisher, float)
    theta0 = np.asarray(theta0, float)

    dali_mode = (g_tensor is not None) or (h_tensor is not None)

    if not dali_mode:
        validate_fisher_shapes(theta0, fisher)
        p = int(theta0.shape[0])
        if any((i < 0) or (i >= p) for i in idx):
            raise IndexError(
                f"idx contains out-of-bounds indices for p={p}: {idx}")

        return {
            "theta0": theta0[idx],
            "fisher": fisher[np.ix_(idx, idx)],
        }

    if g_tensor is None:
        raise ValueError(
            "DALI subspace extraction requires `g_tensor` when `h_tensor` is provided.")

    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)

    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    p = int(theta0.shape[0])
    if any((i < 0) or (i >= p) for i in idx):
        raise IndexError(
            f"idx contains out-of-bounds indices for p={p}: {idx}")

    out = {
        "theta0": theta0[idx],
        "fisher": fisher[np.ix_(idx, idx)],
        "g_tensor": g_tensor[np.ix_(idx, idx, idx)],
    }
    if h_tensor is not None:
        out["h_tensor"] = h_tensor[np.ix_(idx, idx, idx, idx)]
    return out


def build_delta_chi2_fisher(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
) -> float:
    """Computes a displacement chi-squared under the Fisher approximation.

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(p, p)`` with ``p`` the number of parameters.

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
    """Determines which log-prior to use for likelihoods expansion evaluation.

    This helper allows callers to specify a prior in one of two ways: either by passing
    a pre-built ``logprior(theta)`` callable directly, or by providing a lightweight
    prior specification (``prior_terms`` and/or ``prior_bounds``) that is compiled
    internally using :func:`derivkit.forecasting.priors.core.build_prior`.

    Only one of these input styles may be used at a time. Providing both results in a
    ``ValueError``. If neither is provided, the function returns ``None``, indicating
    that no prior is applied.

    Args:
        prior_terms: Prior term specification passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        logprior: Optional custom log-prior callable. If it returns a non-finite value,
            the posterior is treated as zero at that point and the function returns ``-np.inf``.

    Returns:
        A function that computes the log-prior contribution to the posterior, or
        ``None`` if the likelihoods should be evaluated without a prior.
    """
    if logprior is not None and (prior_terms is not None or prior_bounds is not None):
        raise ValueError("Use either `logprior` or (`prior_terms`/`prior_bounds`), not both.")

    if logprior is None and (prior_terms is not None or prior_bounds is not None):
        return build_prior(terms=prior_terms, bounds=prior_bounds)

    return logprior


def build_logposterior_fisher(
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

    If no prior is provided, this returns the Fisher log-likelihoods expansion
    with a flat prior and no hard cutoffs.

    The Fisher approximation corresponds to a purely quadratic ``delta_chi2`` surface::

        delta_chi2 = d.T @ F @ d

    so the log posterior is::

        log p = -0.5 * delta_chi2

    This normalization is equivalent to the ``convention="delta_chi2"`` used for DALI.
    In this interpretation, fixed ``delta_chi2`` values correspond to fixed probability content
    (e.g. 68%, 95%) in parameter space, as for a Gaussian likelihoods.
    See :func:`derivkit.forecasting.expansions.delta_chi2_dali` for the corresponding
    DALI definition of ``delta_chi2`` and its supported conventions.

    Unlike the DALI case, there is no alternative normalization for the Fisher
    approximation: the likelihoods is strictly Gaussian and fully described by the
    quadratic form.

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(p, p)`` with ``p`` the number of parameters.
        prior_terms: prior term specification passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        logprior: Optional custom log-prior callable. If it returns a non-finite value,
            the posterior is treated as zero at that point and the function returns ``-np.inf``.

    Returns:
        Scalar log posterior value, defined up to an additive constant.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shapes(theta0, fisher)

    logprior_fn = _resolve_logprior(prior_terms=prior_terms, prior_bounds=prior_bounds, logprior=logprior)

    logprior_val = 0.0
    if logprior_fn is not None:
        logprior_val = float(logprior_fn(theta))
        if not np.isfinite(logprior_val):
            return -np.inf

    displacement = theta - theta0
    chi2 = float(displacement @ fisher @ displacement)
    return logprior_val - 0.5 * chi2


def build_delta_chi2_dali(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
    *,
    convention: str = "delta_chi2",
) -> float:
    """Computes a displacement chi-squared under the DALI approximation.

    This function evaluates a scalar ``delta_chi2`` from the displacement
    ``d = theta - theta0`` using the Fisher matrix and (optionally) the cubic
    and quartic DALI tensors.

    The ``convention`` parameter controls the numerical prefactors applied to
    the cubic/quartic contractions, i.e. it changes the *scaling* of the higher-
    order corrections relative to the quadratic Fisher term:

    - ``convention="delta_chi2"``:
        ``delta_chi2 = d.T @ F @ d + (1/3) * G[d,d,d] + (1/12) * H[d,d,d,d]``

    - ``convention="matplotlib_loglike"``:
        ``delta_chi2 = d.T @ F @ d + 1 * G[d,d,d] + (1/4) * H[d,d,d,d]``

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix ``(p, p)`` with ``p`` the number of parameters.
        g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
        h_tensor: DALI quartic tensor ``(p, p, p, p)`` or ``None``.
        convention: Controls the prefactors used in the cubic/quartic tensor
            contractions inside ``delta_chi2``.

    Returns:
        The scalar delta chi-squared value.

    Raises:
        ValueError: If an unknown ``convention`` is provided.
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


def build_logposterior_dali(
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

    If no prior is provided, this returns the DALI log-likelihoods expansion with
    a flat prior and no hard cutoffs.

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher/DALI expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix and any
            DALI tensors are assumed to have been computed at this point, and the
            expansion is taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(p, p)`` with ``p`` the number of parameters.
        g_tensor: DALI cubic tensor ``(p, p, p)``.
        h_tensor: DALI quartic tensor ``(p, p, p, p)`` or ``None``.
        convention: The normalization to use (``"delta_chi2"`` or
            ``"matplotlib_loglike"``).
        prior_terms: prior term specification passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        prior_bounds: Global hard bounds passed to
            :func:`derivkit.forecasting.priors.core.build_prior`.
        logprior: Optional custom log-prior callable. If it returns a non-finite value,
            the posterior is treated as zero at that point and the function returns ``-np.inf``.

    Returns:
        Scalar log posterior value, defined up to an additive constant.

    Raises:
        ValueError: If an unknown ``convention`` is provided.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    g_tensor = np.asarray(g_tensor, float)
    h_tensor = None if h_tensor is None else np.asarray(h_tensor, float)
    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)

    logprior_fn = _resolve_logprior(prior_terms=prior_terms, prior_bounds=prior_bounds, logprior=logprior)

    logprior_val = 0.0
    if logprior_fn is not None:
        logprior_val = float(logprior_fn(theta))
        if not np.isfinite(logprior_val):
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

    return logprior_val - 0.5 * chi2
