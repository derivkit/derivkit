"""Extrapolation methods for numerical approximations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "richardson_extrapolate",
    "ridders_extrapolate",
]


def richardson_extrapolate(
        base_values: Sequence[NDArray[np.float64] | float],
        p: int,
        r: float = 2.0,
) -> NDArray[np.float64] | float:
    """Computes Richardson extrapolation on a sequence of approximations.

    Richardson extrapolation improves the accuracy of a sequence of
    numerical approximations that converge with a known leading-order error
    term. Given a sequence of approximations computed with decreasing step sizes,
    this method combines them to eliminate the leading error term, yielding
    a more accurate estimate of the true value.

    Args:
        base_values:
            Sequence of approximations at different step sizes.
            The step sizes are assumed to decrease by a factor of `r`
            between successive entries.
        p:
            The order of the leading error term in the approximations.
        r:
            The step-size reduction factor between successive entries
            (default is 2.0).

    Returns:
        The extrapolated value with improved accuracy.

    Raises:
        ValueError: If `base_values` has fewer than two entries.
    """
    # Work on float arrays for both scalar and vector cases
    n = len(base_values)
    if n < 2:
        raise ValueError("richardson_extrapolate requires at least two base values.")

    vals = [np.asarray(v, dtype=float) for v in base_values]

    for j in range(1, n):
        factor = r ** (p * j)
        for k in range(n - 1, j - 1, -1):
            vals[k] = (factor * vals[k] - vals[k - 1]) / (factor - 1.0)

    result = vals[-1]
    return float(result) if result.ndim == 0 else result


def ridders_extrapolate(
    base_values: Sequence[NDArray[np.float64] | float],
    r: float = 2.0,
    *,
    extrapolator = richardson_extrapolate,
    p: int = 2,
) -> tuple[NDArray[np.float64] | float, float]:
    """Computes a Ridders-style extrapolation on a sequence of approximations.

    This builds the usual Ridders diagonal assuming a central finite-difference
    scheme (leading error is approximately O(h^2)) by repeatedly extrapolating
    prefixes of ``base_values``. By default it uses :func:`richardson_extrapolate`
    with ``p=2``, but a different extrapolator can be passed if needed.

    Args:
        base_values:
            Sequence of derivative approximations at step sizes
            h, h/r, h/r^2, ... (all same shape: scalar, vector, or tensor).
        r:
            Step-size reduction factor (default 2.0).
        extrapolator:
            Function implementing the extrapolation step. Must have the
            signature ``extrapolator(base_values, p, r) -> array_like``.
            Defaults to :func:`richardson_extrapolate`.
        p:
            Leading error order passed to ``extrapolator`` (default 2).

    Returns:
        A tuple ``(best_value, error_estimate)`` where:

        * ``best_value`` is the extrapolated estimate chosen from the
          diagonal entries.
        * ``error_estimate`` is a heuristic scalar error scale given by the
          minimum difference between consecutive diagonal elements.

    Raises:
        ValueError:
            If fewer than two base values are provided.
    """
    n = len(base_values)
    if n < 2:
        raise ValueError("ridders_extrapolate requires at least two base values.")

    diag: list[NDArray[np.float64]] = []
    err_estimates: list[float] = []

    for j in range(n):
        if j == 0:
            d_j = np.asarray(base_values[0], dtype=float)
        else:
            # Use the chosen extrapolator on the first (j+1) base values
            d_j = np.asarray(
                extrapolator(base_values[: j + 1], p=p, r=r),
                dtype=float,
            )

        diag.append(d_j)

        if j == 0:
            err_estimates.append(np.inf)
        else:
            diff = np.asarray(diag[j] - diag[j - 1], dtype=float)
            err_estimates.append(float(np.max(np.abs(diff))))

    # Pick the diagonal element with the smallest estimated error
    best_idx = int(np.argmin(err_estimates))
    best_val = diag[best_idx]
    best_err = err_estimates[best_idx]

    if best_val.ndim == 0:
        return float(best_val), float(best_err)
    return best_val, float(best_err)
