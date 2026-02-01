"""Utilities for evaluating Fisher and DALI likelihood expansions.

This module provides functional helpers to evaluate approximate likelihoods
(or posterior) surfaces from forecast tensors.

Conventions
-----------

This module uses a single convention throughout:

- ``delta_chi2`` is defined from the displacement ``d = theta - theta0``.
- The log posterior is returned (up to an additive constant) as::

    log p(theta) = logprior(theta) - 0.5 * delta_chi2(theta)

With the forecast tensors returned by :func:`derivkit.forecasting.get_forecast_tensors`
(using the introduced-at-order convention):

- ``dali[1] == (F,)``
- ``dali[2] == (D1, D2)``
- ``dali[3] == (T1, T2, T3)``

the DALI ``delta_chi2`` is:

- order 1 (Fisher): ``d.T @ F @ d``
- order 2 (doublet): add ``(1/3) D1[d,d,d] + (1/12) D2[d,d,d,d]``
- order 3 (triplet): add ``(1/3) T1[d^4] + (1/6) T2[d^5] + (1/36) T3[d^6]``

GetDist convention
------------------

GetDist expects ``loglikes`` to be the negative log posterior, up to a constant.
Since this module defines::

    log p = logprior - 0.5 * delta_chi2 + const

a compatible choice for GetDist is::

    loglikes = -logprior + 0.5 * delta_chi2

(optionally shifted by an additive constant for numerical stability).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.forecasting.forecast_core import SUPPORTED_FORECAST_ORDERS
from derivkit.forecasting.priors_core import build_prior
from derivkit.utils.validate import (
    validate_dali_shape,
    validate_fisher_shape,
)

__all__ = [
    "build_subspace",
    "build_delta_chi2_fisher",
    "build_delta_chi2_dali",
    "build_logposterior_fisher",
    "build_logposterior_dali",
]


def _validate_and_normalize_idx(idx: Sequence[int], *, p: int) -> list[int]:
    """Validates and normalizes a sequence of parameter indices.

    Args:
        idx: Sequence of parameter indices.
        p: Total number of parameters.

    Returns:
        Indices as a list of Python ``int``.

    Raises:
        TypeError: If any entry of ``idx`` is not an integer.
        IndexError: If any index is out of bounds for ``p``.
    """
    idx_list = list(idx)
    if not all(isinstance(i, (int, np.integer)) for i in idx_list):
        raise TypeError("idx must contain integer indices")
    if any((i < 0) or (i >= p) for i in idx_list):
        raise IndexError(f"idx contains out-of-bounds indices for p={p}: {idx_list}")
    return idx_list


def _slice_param_tensor(t: NDArray[np.floating], idx: list[int]) -> NDArray[np.float64]:
    """Slices a parameter-space tensor along all axes.

    This helper assumes ``t`` is a tensor whose every axis indexes parameters,
    e.g. Fisher ``(p, p)``, a cubic tensor ``(p, p, p)``, etc.

    Args:
        t: Tensor to slice.
        idx: Parameter indices to keep.

    Returns:
        Sliced tensor as ``float64``.
    """
    t64 = np.asarray(t, np.float64)
    sl = np.ix_(*([idx] * t64.ndim))
    return t64[sl]


def build_subspace(
    idx: Sequence[int],
    *,
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating] | None = None,
    dali: dict[int, tuple[NDArray[np.floating], ...]] | None = None,
) -> dict[str, Any]:
    """Extracts a parameter subspace for Fisher or DALI expansions.

    This returns a *slice* through parameter space: parameters not in ``idx`` are
    held fixed at their expansion values. This is not a marginalization.

    Provide exactly one of ``fisher`` or ``dali``:

    - Fisher: ``fisher`` has shape ``(p, p)`` and the return dict contains
      ``{"theta0": theta0_sub, "fisher": fisher_sub}``.
    - DALI: ``dali`` is the dict form returned by
      :func:`derivkit.forecasting.get_forecast_tensors` using the introduced-at-order
      convention, and the return dict contains ``{"theta0": theta0_sub, "dali": dali_sub}``.

    Args:
        idx: Parameter indices to extract.
        theta0: Expansion point of shape ``(p,)``.
        fisher: Fisher matrix of shape ``(p, p)``.
        dali: Forecast tensors as a dict mapping ``order -> multiplet``.

    Returns:
        A dict containing the sliced objects. Always includes ``"theta0"``.
        Includes ``"fisher"`` if ``fisher`` was provided, or ``"dali"`` if ``dali``
        was provided.

    Raises:
        ValueError: If not exactly one of ``fisher`` or ``dali`` is provided.
        TypeError: If ``idx`` contains non-integers, or if ``dali`` is not a dict.
        IndexError: If any index in ``idx`` is out of bounds.
        ValueError: If the provided arrays have incompatible shapes.
    """
    theta0_arr = np.asarray(theta0, np.float64).reshape(-1)
    p = int(theta0_arr.shape[0])

    if (fisher is None) == (dali is None):
        raise ValueError("Provide exactly one of `fisher` or `dali`.")

    if fisher is not None:
        fisher_arr = np.asarray(fisher, np.float64)
        validate_fisher_shape(theta0_arr, fisher_arr)
        idx_list = _validate_and_normalize_idx(idx, p=p)
        return {
            "theta0": theta0_arr[idx_list],
            "fisher": fisher_arr[np.ix_(idx_list, idx_list)],
        }

    # dali is not None
    if not isinstance(dali, dict):
        raise TypeError("dali must be the dict form returned by get_forecast_tensors.")

    validate_dali_shape(theta0_arr, dali)
    idx_list = _validate_and_normalize_idx(idx, p=p)

    dali_sub: dict[int, tuple[NDArray[np.float64], ...]] = {}
    for k, multiplet in dali.items():
        dali_sub[int(k)] = tuple(_slice_param_tensor(t, idx_list) for t in multiplet)

    return {
        "theta0": theta0_arr[idx_list],
        "dali": dali_sub,
    }


def build_delta_chi2_fisher(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
) -> float:
    """Computes a displacement chi-squared under the Fisher approximation.

    Args:
        theta: Evaluation point in parameter space. This is the trial parameter vector
            at which the Fisher expansion is evaluated.
        theta0: Expansion point (reference parameter vector). The Fisher matrix
            is assumed to have been computed at this point, and the expansion is
            taken in the displacement ``theta - theta0``.
        fisher: Fisher matrix with shape ``(p, p)`` with ``p`` the number of parameters.

    Returns:
        The scalar delta chi-squared value between ``theta`` and ``theta_0``.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    fisher = np.asarray(fisher, float)
    validate_fisher_shape(theta0, fisher)

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
    validate_fisher_shape(theta0, fisher)

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
    dali: Any,
    *,
    forecast_order: int | None = 2,
) -> float:
    """Compute ``delta_chi2`` under the DALI approximation.

    This evaluates a scalar ``delta_chi2`` from the displacement ``d = theta - theta0``
    using forecast tensors returned by :func:`derivkit.forecasting.get_forecast_tensors`.

    The input must be the dict form using the introduced-at-order convention:

    - ``dali[1] == (F,)`` with ``F`` of shape ``(p, p)``
    - ``dali[2] == (D1, D2)`` with shapes ``(p, p, p)`` and ``(p, p, p, p)``
    - ``dali[3] == (T1, T2, T3)`` with shapes ``(p,)*4``, ``(p,)*5``, ``(p,)*6``

    The evaluated quantity is:

    - order 2: ``d.T @ F @ d + (1/3) D1[d^3] + (1/12) D2[d^4]``
    - order 3: order 2 plus ``(1/3) T1[d^4] + (1/6) T2[d^5] + (1/36) T3[d^6]``.

    Args:
        theta: Evaluation point in parameter space.
        theta0: Expansion point (fiducial parameters).
        dali: Forecast tensors as a dict.
        forecast_order: Maximum order to include. If ``None``, uses the highest key in
            ``dali`` and requires it to be at least 2.

    Returns:
        Scalar ``delta_chi2``.

    Raises:
        TypeError: If ``dali`` is not a dict.
        ValueError: If required tensor orders are missing or have incompatible shapes.
    """
    theta = np.asarray(theta, float).reshape(-1)
    theta0 = np.asarray(theta0, float).reshape(-1)

    if theta.shape != theta0.shape:
        raise ValueError(
            f"theta and theta0 must have the same shape; got {theta.shape} and {theta0.shape}.")

    # DALI evaluation requires the dict form (needs Fisher inside dali[1]).
    if not isinstance(dali, dict):
        raise TypeError(
            "build_delta_chi2_dali expects the dict form from get_forecast_tensors "
            "(needs dali[1]=(F,) plus higher-order tensors)."
        )

    validate_dali_shape(theta0, dali)

    # Choose order
    if forecast_order is None:
        chosen = max(dali.keys())
    else:
        try:
            chosen = int(forecast_order)
        except Exception as e:
            raise TypeError(
                f"forecast_order must be an int or None;"
                f" got {type(forecast_order)}.") from e

    if chosen not in SUPPORTED_FORECAST_ORDERS:
        raise ValueError(
            f"forecast_order={chosen} is not supported."
            f" Supported values: {SUPPORTED_FORECAST_ORDERS}."
        )

    if chosen < 2:
        raise ValueError(
            "build_delta_chi2_dali requires forecast_order >= 2. "
            "Use your Fisher delta-chi2 function for forecast_order=1."
        )

    # Require the needed keys exist
    if 1 not in dali or 2 not in dali:
        raise ValueError(
            "dali must contain keys 1 and 2 (Fisher + doublet tensors).")
    if chosen >= 3 and 3 not in dali:
        raise ValueError(
            "forecast_order=3 requires dali to contain key 3 (triplet tensors).")

    fisher = np.asarray(dali[1][0], dtype=np.float64)
    d = theta - theta0
    chi2 = float(d @ fisher @ d)

    # doublet
    d1 = np.asarray(dali[2][0], dtype=np.float64)
    d2 = np.asarray(dali[2][1], dtype=np.float64)
    chi2 += (1.0 / 3.0) * float(np.einsum("ijk,i,j,k->",
                                          d1, d, d, d))
    chi2 += (1.0 / 12.0) * float(np.einsum("ijkl,i,j,k,l->",
                                           d2, d, d, d, d))

    if chosen == 2:
        return chi2

    t1 = np.asarray(dali[3][0], dtype=np.float64)
    t2 = np.asarray(dali[3][1], dtype=np.float64)
    t3 = np.asarray(dali[3][2], dtype=np.float64)

    t1_4 = float(np.einsum("ijkl,i,j,k,l->",
                           t1, d, d, d, d))
    t2_5 = float(np.einsum("ijklm,i,j,k,l,m->",
                           t2, d, d, d, d, d))
    t3_6 = float(np.einsum("ijklmn,i,j,k,l,m,n->",
                           t3, d, d, d, d, d, d))

    chi2 = chi2 + (1.0 / 3.0) * t1_4 + (1.0 / 6.0) * t2_5 + (1.0 / 36.0) * t3_6
    return chi2


def build_logposterior_dali(
    theta: NDArray[np.floating],
    theta0: NDArray[np.floating],
    dali: Any,
    *,
    forecast_order: int | None = 2,
    prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[NDArray[np.floating]], float] | None = None,
) -> float:
    """Compute the log posterior under the DALI approximation.

    The posterior is evaluated as::

        log p(theta) = logprior(theta) - 0.5 * delta_chi2(theta)

    where ``delta_chi2`` is computed from the dict-form forecast tensors ``dali``
    using :func:`build_delta_chi2_dali`.

    Args:
        theta: Evaluation point in parameter space.
        theta0: Expansion point (fiducial parameters).
        dali: Forecast tensors as a dict in the introduced-at-order convention.
        forecast_order: Maximum order to include in ``delta_chi2``. If ``None``, uses
            the highest key in ``dali``.
        prior_terms: Prior term specification passed to :func:`build_prior`.
        prior_bounds: Global hard bounds passed to :func:`build_prior`.
        logprior: Optional custom log-prior callable.

    Returns:
        Scalar log posterior value (up to an additive constant). If the prior evaluates
        to a non-finite value, returns ``-np.inf``.
    """
    theta = np.asarray(theta, float).reshape(-1)
    theta0 = np.asarray(theta0, float).reshape(-1)

    if theta.shape != theta0.shape:
        raise ValueError(
            f"theta and theta0 must have the same shape; got {theta.shape} and {theta0.shape}."
        )

    if not isinstance(dali, dict):
        raise TypeError(
            "build_logposterior_dali expects the dict form from get_forecast_tensors."
        )

    validate_dali_shape(theta0, dali)

    logprior_fn = _resolve_logprior(
        prior_terms=prior_terms, prior_bounds=prior_bounds, logprior=logprior
    )
    logprior_val = 0.0
    if logprior_fn is not None:
        logprior_val = float(logprior_fn(theta))
        if not np.isfinite(logprior_val):
            return -np.inf

    chi2 = build_delta_chi2_dali(theta, theta0, dali,
                                 forecast_order=forecast_order)

    return logprior_val - 0.5 * chi2
