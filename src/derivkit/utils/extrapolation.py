"""Extrapolation methods for numerical approximations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "richardson_extrapolate",
    "ridders_extrapolate",
    "gauss_richardson_extrapolate",
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


def _rbf_kernel_1d(x: NDArray[np.float64],
                   y: NDArray[np.float64],
                   length_scale: float) -> NDArray[np.float64]:
    """Compute the RBF kernel matrix between 1D inputs x and y.

    Args:
        x: 1D array of shape (n,).
        y: 1D array of shape (m,).
        length_scale: Length scale parameter for the RBF kernel.

    Returns:
        Kernel matrix of shape (n, m).
    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    diff2 = (x[:, None] - y[None, :]) ** 2
    return np.exp(-0.5 * diff2 / (length_scale**2))


def gauss_richardson_extrapolate(
    base_values: Sequence[NDArray[np.float64] | float],
    h_values: Sequence[float],
    p: int,
    jitter: float = 1e-10,
) -> tuple[NDArray[np.float64] | float, NDArray[np.float64] | float]:
    """Gauss–Richardson extrapolation for a sequence of approximations f(h_i).

    This method uses a Gaussian-process model with a radial-basis-function (RBF)
    kernel to perform Richardson extrapolation, providing both an improved estimate
    of the true value at h=0 and an uncertainty estimate. For more details, see arXiv:2401.07562.

    Args:
        base_values: Sequence of approximations at different step sizes h_i.
        h_values: Corresponding step sizes (must be positive and same length as base_values).
        p: The order of the leading error term in the approximations.
        jitter: Small positive value added to the diagonal of the kernel matrix for numerical stability.
            Defaults to 1e-10.

    Returns:
        A tuple (extrapolated_value, error_estimate) where:

          - extrapolated_value is the Gauss–Richardson extrapolated estimate at h=0.
          - error_estimate is a heuristic uncertainty estimate for the extrapolated value.

    Raises:
        ValueError:
            If h_values and base_values have different lengths or if any h_value is non-positive.
    """
    h = np.asarray(h_values, dtype=float).ravel()
    if len(base_values) != h.size:
        raise ValueError("base_values and h_values must have the same length.")
    if np.any(h <= 0):
        raise ValueError("All h_values must be > 0.")

    y = np.stack([np.asarray(v, dtype=float) for v in base_values], axis=0)
    n = h.size

    # Error bound b(h) = h^p
    b = h**p

    # crude length scale from spacing
    h_sorted = np.sort(h)
    # then we compute the differences between consecutive sorted h values
    diffs = np.diff(h_sorted)
    # if there are any positive differences, take the median of those
    if np.any(diffs > 0):
        char = np.median(diffs[diffs > 0])
    else:
        char = max(h.max() - h.min(), 1e-12)

    ell = char

    # we then build the kernel matrix kb
    ke = _rbf_kernel_1d(h, h, ell)
    kb = (b[:, None] * b[None, :]) * ke
    kb += jitter * np.eye(n)

    # we then precompute the matrix-vector product kb^{-1} 1
    one = np.ones(n)
    kb_inv_1 = np.linalg.solve(kb, one)

    flat = y.reshape(n, -1)
    means = []
    errs = []

    denom = float(one @ kb_inv_1)
    for j in range(flat.shape[1]):
        col = flat[:, j]

        # reuse kb_inv_1 or recompute:
        kb_inv_y = np.linalg.solve(kb, col)

        num = float(one @ kb_inv_y)
        mean0 = num / denom  # μ̂

        # Residuals
        resid = col - mean0 * one
        kb_inv_resid = np.linalg.solve(kb, resid)

        # Noise variance estimate
        sigma2 = float(resid @ kb_inv_resid) / max(n - 1, 1)

        # Variance at h=0
        var0 = sigma2 / denom if denom > 0 else 0.0
        var0 = max(var0, 0.0)
        std0 = float(np.sqrt(var0))

        means.append(mean0)
        errs.append(std0)

    means_arr = np.array(means).reshape(y.shape[1:])
    errs_arr = np.array(errs).reshape(y.shape[1:])

    if means_arr.ndim == 0:
        return float(means_arr), float(errs_arr)
    return means_arr, errs_arr
