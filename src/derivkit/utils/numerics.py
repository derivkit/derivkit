"""Numerical utilities."""

from __future__ import annotations

<<<<<<< HEAD
from functools import partial
=======
import warnings
>>>>>>> 243daef (utils: clean up numerics helpers)
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.logger import derivkit_logger

__all__ = [
    "central_difference_error_estimate",
    "relative_error",
    "evaluate_logprior",
    "is_in_bounds",
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
        derivkit_logger.warning(
            "central_difference_error_estimate called with order > 4,"
            " which is not supported by finite_difference module.",
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
        logprior: Callable[[NDArray[np.floating]], np.floating] | None,
) -> np.floating:
    """Evaluates a log-prior and attempts to handle non-finite values.

    If ``logprior`` is ``None``, this function assumes a flat prior and returns zero.
    If a prior is provided, its output is interpreted as a log-density
    defined up to an additive constant. Any non-finite value (e.g., ``-np.inf`` or
    ``np.nan``) is treated as zero probability and mapped to ``-np.inf``.

    Args:
        theta: Parameter vector at which to evaluate the prior.
        logprior: Callable returning the log-prior density, or ``None`` to indicate a flat prior.

    Returns:
        The log-prior value at ``theta``. If the prior assigns zero probability to
        that point, returns ``-np.inf``.
    """
    if logprior is None:
        return np.float64(0.0)
    v = np.float64(logprior(np.asarray(theta, dtype=float)))
    return v if np.isfinite(v) else np.float64(-np.inf)


def is_in_bounds(
        theta: NDArray[np.floating],
        bounds: Sequence[tuple[np.floating | None, np.floating | None]] | None,
) -> bool:
    """Checks whether a parameter vector lies within specified bounds.

    If ``bounds`` is ``None``, this returns True by convention so callers can write
    ``if is_in_bounds(theta, bounds): ...`` without special-casing the absence of bounds.

    Bounds are interpreted component-wise. For each parameter, either side may
    be unbounded by setting that limit to ``None``.

    Args:
        theta: Parameter vector to test.
        bounds: Optional sequence of ``(lower, upper)`` pairs, one per parameter.
            Use ``None`` for an unconstrained lower or upper limit.

    Returns:
        A boolean indicating whether all parameters satisfy their bounds
        (or ``True`` if bounds is ``None`).

    Raises:
        ValueError: If ``bounds`` is provided and its length does not match ``theta``.
    """
    if bounds is None:
        return True
    if len(bounds) != theta.size:
        raise ValueError(f"bounds length {len(bounds)} != theta length {theta.size}")

    result = True
    for t, (lo, hi) in zip(theta, bounds):
        if (lo is not None and t < lo) or (hi is not None and t > hi):
            result = False
            break
    return result


def apply_hard_bounds(
        term: Callable[[NDArray[np.floating]], np.floating],
        *,
        bounds: Sequence[tuple[np.floating | None, np.floating | None]] | None = None,
) -> Callable[[NDArray[np.floating]], np.floating]:
    """Returns a bounded version of a log-density contribution.

    A ``term`` is a callable that returns a scalar log-density contribution
    and support refers to the region where the density is non-zero.

    This helper enforces a top-hat support region defined by ``bounds``. If
    ``theta`` lies outside the allowed region, the result is ``-np.inf`` to denote
    zero probability. If ``theta`` lies inside the region, the provided term is
    evaluated and any non-finite output is treated as ``-np.inf``.

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

    def bounded(theta: NDArray[np.floating]) -> np.floating:
        """Evaluates the bounded log-density term."""
        th = np.asarray(theta, dtype=float)
        v = np.float64(-np.inf)
        if is_in_bounds(th, bounds):
            v = np.float64(term(th))
        return v if np.isfinite(v) else np.float64(-np.inf)

    return bounded


def sum_terms(
        *terms: Callable[[NDArray[np.floating]], np.floating]
) -> Callable[[NDArray[np.floating]], np.floating]:
    """Constructs a composite log term by summing multiple contributions.

    The returned callable evaluates each provided term at the same parameter
    vector and adds the results. If any term is non-finite, the composite term
    evaluates to ``-np.inf``, corresponding to zero probability under the combined
    density.

    A ``term`` is a callable that returns a scalar log-density contribution
    and support refers to the region where the density is non-zero.

    Args:
        *terms: One or more callables, each returning a log-density contribution.

    Returns:
        A callable that returns the sum of the provided log terms at ``theta``.

    Raises:
        ValueError: If no terms are provided.
    """
    if len(terms) == 0:
        raise ValueError("sum_terms requires at least one term")

    def summed(theta: NDArray[np.floating]) -> np.floating:
        """Evaluates the summed log-density terms."""
        th = np.asarray(theta, dtype=float)
        total = np.float64(0.0)
        for f in terms:
            v = np.float64(f(th))
            if not np.isfinite(v):
                return np.float64(-np.inf)
            total = np.float64(total + v)
        return total

    return summed


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
    """Computes log(sum(exp(x))) for a 1D array using the max-shift identity.

    This implements the common max-shift trick used to reduce overflow/underflow.

    Args:
        x: 1D array-like values.

    Returns:
        Value of log(sum(exp(x))) as a float.

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
