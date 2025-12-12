"""Utilities for evaluating Fisher and DALI likelihood expansions.

This module provides functional helpers to evaluate approximate likelihood
(or posterior) surfaces from forecast tensors:

- Fisher quadratic approximation (F)
- Doublet-DALI cubic/quartic corrections (F, G, H)

The functions here do not build tensors; they assume you already have
a Fisher matrix and (optionally) DALI tensors from :mod:`derivkit.forecasting`.

Conventions
-----------
We expose two conventions that are common in the codebase:

- ``convention="delta_chi2"``:
    Uses the standard DALI Δχ² form

        Δχ² = d^T F d + (1/3) G:d^3 + (1/12) H:d^4

    and returns log posterior (up to a constant) as

        log p = -0.5 * Δχ².

- ``convention="matplotlib_loglike"``:
    Matches the prefactors used in Niko's matplotlib contour scripts:

        log p = -0.5 d^T F d - 0.5 (G:d^3) - 0.125 (H:d^4)

    which corresponds to

        Δχ² = d^T F d + (G:d^3) + 0.25 (H:d^4)

    so that again log p = -0.5 * Δχ².

GetDist convention
------------------
GetDist expects ``loglikes`` to be the negative log posterior,

    ``loglikes = -log(posterior)``

up to an additive constant. Since this module defines

    ``log(posterior) = -0.5 * Δχ² + const``,

a compatible choice for GetDist is therefore

    ``loglikes = 0.5 * Δχ²``

(optionally shifted by a constant for numerical stability).

Notes
-----
- All log posterior values returned are defined up to an additive constant.
- Optional hard bounds implement flat priors: outside bounds returns ``-np.inf``.
"""


from __future__ import annotations

from typing import Sequence, Callable

import numpy as np

from derivkit.utils.validate import validate_fisher_shapes, validate_dali_shapes

Array = np.ndarray

__all__ = [
    "slice_fisher",
    "slice_tensors",
    "delta_chi2_fisher",
    "delta_chi2_dali",
    "logpost_fisher",
    "logpost_dali",
]


def slice_fisher(F: Array, idx: Sequence[int]) -> Array:
    """Slice a Fisher matrix to a subset of parameter indices."""
    idx = list(idx)
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be square 2D, got {F.shape}")
    return F[np.ix_(idx, idx)]


def slice_tensors(
    theta0: Array,
    F: Array,
    G: Array,
    H: Array | None,
    idx: Sequence[int],
) -> tuple[Array, Array, Array, Array | None]:
    """Slice full tensors to a subset of parameter indices."""
    idx = list(idx)
    t0 = np.asarray(theta0, float)[idx]
    F2 = np.asarray(F, float)[np.ix_(idx, idx)]
    G2 = np.asarray(G, float)[np.ix_(idx, idx, idx)]
    H2 = None if H is None else np.asarray(H, float)[np.ix_(idx, idx, idx, idx)]
    return t0, F2, G2, H2


def _apply_bounds(theta: Array, bounds) -> bool:
    if bounds is None:
        return True
    for t, (lo, hi) in zip(theta, bounds):
        if (lo is not None and t < lo) or (hi is not None and t > hi):
            return False
    return True



# ---------------------------------------------------------------------------
# Fisher expansion evaluation
# ---------------------------------------------------------------------------

def delta_chi2_fisher(
    theta: Array,
    theta0: Array,
    F: Array,
) -> float:
    """Compute Fisher Δχ² = d^T F d."""
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)
    validate_fisher_shapes(theta0, F)

    d = theta - theta0
    return float(d @ F @ d)


def logpost_fisher(
    theta: Array,
    theta0: Array,
    F: Array,
    *,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[Array], float] | None = None,
) -> float:
    """Log posterior (up to a constant) under the Fisher approximation."""
    theta = np.asarray(theta, float)

    # hard top-hat bounds (flat prior with cutoffs)
    if hard_bounds is not None:
        for t, (lo, hi) in zip(theta, hard_bounds):
            if (lo is not None and t < lo) or (hi is not None and t > hi):
                return -np.inf

    # additional prior term (optional)
    lp = 0.0
    if logprior is not None:
        lp = float(logprior(theta))
        if not np.isfinite(lp):
            return -np.inf

    # Fisher quadratic term
    d = theta - np.asarray(theta0, float)
    F = np.asarray(F, float)
    chi2 = float(d @ F @ d)

    return lp - 0.5 * chi2


# ---------------------------------------------------------------------------
# DALI expansion evaluation
# ---------------------------------------------------------------------------

def delta_chi2_dali(
    theta: Array,
    theta0: Array,
    F: Array,
    G: Array,
    H: Array | None,
    *,
    convention: str = "delta_chi2",
) -> float:
    """
    Compute an effective Δχ² from DALI tensors.

    Args:
        theta: Parameter vector.
        theta0: Expansion point.
        F: Fisher matrix (P, P).
        G: DALI cubic tensor (P, P, P).
        H: DALI quartic tensor (P, P, P, P) or None.
        convention: See module docstring.

    Returns:
        Scalar Δχ² value.
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    H = None if H is None else np.asarray(H, float)
    validate_dali_shapes(theta0, F, G, H)

    d = theta - theta0
    quad = float(d @ F @ d)
    g3 = float(np.einsum("ijk,i,j,k->", G, d, d, d))
    h4 = 0.0 if H is None else float(np.einsum("ijkl,i,j,k,l->", H, d, d, d, d))

    if convention == "delta_chi2":
        return quad + (1.0 / 3.0) * g3 + (1.0 / 12.0) * h4

    if convention == "matplotlib_loglike":
        return quad + g3 + 0.25 * h4

    raise ValueError(f"Unknown convention='{convention}'")


def logpost_dali(
    theta: Array,
    theta0: Array,
    F: Array,
    G: Array,
    H: Array | None,
    *,
    convention: str = "delta_chi2",
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    logprior: Callable[[Array], float] | None = None,
) -> float:
    """Log posterior (up to a constant) under the DALI approximation."""
    theta = np.asarray(theta, float)

    # hard top-hat bounds (flat prior with cutoffs)
    if hard_bounds is not None:
        for t, (lo, hi) in zip(theta, hard_bounds):
            if (lo is not None and t < lo) or (hi is not None and t > hi):
                return -np.inf

    # additional prior term (optional)
    lp = 0.0
    if logprior is not None:
        lp = float(logprior(theta))
        if not np.isfinite(lp):
            return -np.inf

    # DALI Δχ²
    d = theta - np.asarray(theta0, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    H = None if H is None else np.asarray(H, float)

    quad = float(d @ F @ d)
    g3 = float(np.einsum("ijk,i,j,k->", G, d, d, d))
    h4 = 0.0 if H is None else float(np.einsum("ijkl,i,j,k,l->", H, d, d, d, d))

    if convention == "delta_chi2":
        chi2 = quad + (1.0 / 3.0) * g3 + (1.0 / 12.0) * h4
    elif convention == "matplotlib_loglike":
        chi2 = quad + g3 + 0.25 * h4
    else:
        raise ValueError(f"Unknown convention='{convention}'")

    return lp - 0.5 * chi2
