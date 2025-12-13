"""Numerical utilities."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "central_difference_error_estimate",
    "relative_error",
    "evaluate_logprior",
    "in_bounds_checker",
    "apply_hard_bounds",
    "sum_terms",
    "as_1d_float_array",
    "get_index_value",
    "logsumexp_1d",
]


def central_difference_error_estimate(step_size: float, order: int = 1) -> float:
    """Computes a general heuristic size of the first omitted term in central-difference stencils.

    Uses the general pattern h^2 / ((order + 1) * (order + 2)) as a
    rule-of-thumb O(h^2) truncation-error scale.

    Args:
        step_size: Grid spacing.
        order: Derivative order (positive integer).

    Returns:
        Estimated truncation error scale.
    """
    if order < 1:
        raise ValueError("order must be a positive integer.")

    # if order higher than 4 we do not support it, but we can still compute the estimate
    if order > 4:
        warnings.warn(
            "central_difference_error_estimate called with order > 4,"
            " which is not supported by finite_difference module.",
            UserWarning,
        )
    return step_size**2 / ((order + 1) * (order + 2))


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative error metric between a and b.

    This metric is defined as the maximum over all components of a and b of
    the absolute difference divided by the maximum of 1.0 and the absolute values of
    a and b.

    Args:
        a: First array-like input.
        b: Second array-like input.

    Returns:
        The relative error metric as a float.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))


def evaluate_logprior(
    theta: ArrayLike,
    logprior: Callable[[NDArray[np.floating]], float] | None,
) -> float:
    """Evaluates a log-prior with robust handling of invalid values.

    If no prior is provided, this function represents an improper flat prior
    and returns zero. If a prior is provided, its value is interpreted as a
    log-density defined up to an additive constant. Any non-finite value
    (e.g., ``-inf`` or ``nan``) is treated as zero prior probability.

    Args:
        theta: Parameter vector at which to evaluate the prior.
        logprior: Callable returning the log-prior density, or ``None`` to
            indicate an uninformative (flat) prior.

    Returns:
        The log-prior value at ``theta``, or ``-inf`` if the prior assigns
        zero probability to that point.
    """
    if logprior is None:
        return 0.0
    v = float(logprior(np.asarray(theta)))
    return v if np.isfinite(v) else -np.inf


def in_bounds_checker(
    theta: NDArray[np.floating],
    bounds: Sequence[tuple[float | None, float | None]] | None,
) -> bool:
    """Tests whether parameters lie within optional per-parameter bounds.

    Bounds are interpreted component-wise. For each parameter, either side may
    be unbounded by setting that limit to ``None``. If no bounds are provided,
    all points are considered valid.

    Args:
        theta: Parameter vector to test.
        bounds: Optional sequence of ``(lower, upper)`` pairs, one per parameter.
            Use ``None`` for an unconstrained lower or upper limit.

    Returns:
        True if all parameters satisfy their bounds (or if ``bounds`` is ``None``);
        otherwise False.
    """
    if bounds is None:
        return True
    if len(bounds) != theta.size:
        raise ValueError(f"bounds length {len(bounds)} != theta length {theta.size}")

    for t, (lo, hi) in zip(theta, bounds):
        if (lo is not None and t < lo) or (hi is not None and t > hi):
            return False
    return True


def _apply_hard_bounds_impl(
    theta: NDArray[np.floating],
    *,
    term: Callable[[NDArray[np.floating]], float],
    bounds: Sequence[tuple[float | None, float | None]] | None,
) -> float:
    """Evaluates a log-density term with hard-support truncation.

    This helper enforces a top-hat support region defined by ``bounds``. If
    ``theta`` lies outside the allowed region, the result is ``-inf`` to denote
    zero probability. If ``theta`` lies inside the region, the provided term is
    evaluated and any non-finite output is treated as ``-inf``.

    Args:
        theta: Parameter vector at which to evaluate the term.
        term: Callable returning a log-density contribution.
        bounds: Optional sequence of ``(lower, upper)`` pairs defining the
            allowed support region.

    Returns:
        The term value at ``theta`` if inside bounds and finite; otherwise
        ``-inf``.
    """
    th = np.asarray(theta, dtype=float)
    if not in_bounds_checker(th, bounds):
        return -np.inf
    v = float(term(th))
    return v if np.isfinite(v) else -np.inf


def apply_hard_bounds(
    term: Callable[[NDArray[np.floating]], float],
    *,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Returns a bounded version of a log-density term.

    If ``bounds`` is provided, the returned callable represents the same log
    term but with compact support: points outside the bounds are assigned
    ``-inf`` (zero probability). If ``bounds`` is ``None``, the original term
    is returned unchanged.

    Args:
        term: Callable returning a log-density contribution.
        bounds: Optional sequence of ``(lower, upper)`` pairs defining the
            allowed support region.

    Returns:
        A callable with the same signature as ``term`` that enforces the given
        bounds, or ``term`` itself if no bounds are provided.
    """
    if bounds is None:
        return term
    return partial(_apply_hard_bounds_impl, term=term, bounds=bounds)


def _sum_terms_impl(
    theta: NDArray[np.floating],
    *,
    terms: tuple[Callable[[NDArray[np.floating]], float], ...],
) -> float:
    """Evaluates a sum of log-density terms.

    This helper is meant for constructing composite priors/likelihood pieces as
    additive log terms. If any term evaluates to a non-finite value, the sum is
    treated as ``-inf`` to indicate that the combined density assigns zero
    probability to ``theta``.

    Args:
        theta: Parameter vector at which to evaluate the terms.
        terms: Tuple of callables, each returning a log-density contribution.

    Returns:
        Sum of term values at ``theta`` if all are finite; otherwise ``-inf``.
    """
    th = np.asarray(theta, dtype=float)
    total = 0.0
    for f in terms:
        v = float(f(th))
        if not np.isfinite(v):
            return -np.inf
        total += v
    return total


def sum_terms(
    *terms: Callable[[NDArray[np.floating]], float],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a composite log term by summing multiple contributions.

    The returned callable evaluates each provided term at the same parameter
    vector and adds the results. If any term is non-finite, the composite term
    evaluates to ``-inf``, corresponding to zero probability under the combined
    density.

    Args:
        *terms: One or more callables, each returning a log-density contribution.

    Returns:
        A callable that returns the sum of the provided log terms at ``theta``.

    Raises:
        ValueError: If no terms are provided.
    """
    if len(terms) == 0:
        raise ValueError("sum_terms requires at least one term")
    return partial(_sum_terms_impl, terms=terms)


def as_1d_float_array(x: ArrayLike, *, name: str = "x") -> NDArray[np.float64]:
    """Convert input to a 1D float array.

    This is a small convenience for model/likelihood/prior code that expects
    parameter vectors. It performs a minimal shape check (must be 1D) and
    ensures a float dtype.

    Args:
        x: Input array-like.
        name: Name used in error messages.

    Returns:
        1D NumPy array with dtype float64.

    Raises:
        ValueError: If the converted array is not 1D.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def get_index_value(theta: ArrayLike, index: int, *, name: str = "theta") -> float:
    """Return theta[index] as a float with a clear bounds error.

    This helper is useful for one-dimensional priors/likelihood pieces that
    operate on a single parameter component.

    Args:
        theta: 1D parameter vector.
        index: Index to extract.
        name: Name used in error messages.

    Returns:
        Value at the given index as float.

    Raises:
        ValueError: If theta is not 1D.
        IndexError: If index is out of bounds.
    """
    th = as_1d_float_array(theta, name=name)
    j = int(index)
    if j < 0 or j >= th.size:
        raise IndexError(f"{name} index {j} out of bounds for length {th.size}")
    return float(th[j])


def logsumexp_1d(x: ArrayLike) -> float:
    """Compute log(sum(exp(x))) stably for a 1D array.

    This is a minimal, dependency-free alternative to scipy.special.logsumexp
    for the common 1D case (e.g., mixture models in log-space).

    Args:
        x: 1D array-like values.

    Returns:
        Stable logsumexp as float.

    Raises:
        ValueError: If x is not 1D.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"logsumexp_1d expects a 1D array, got shape {arr.shape}")

    m = float(np.max(arr))
    if not np.isfinite(m):
        return m

    return float(m + np.log(np.sum(np.exp(arr - m))))
