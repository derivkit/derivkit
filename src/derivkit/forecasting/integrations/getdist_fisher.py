from __future__ import annotations

from typing import Sequence

import numpy as np

Array = np.ndarray

__all__ = [
    "slice_fisher",
    "fisher_to_cov",
    "fisher_to_gaussiannd",
    "fisher_to_mcsamples",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slice_fisher(F: Array, idx: Sequence[int]) -> Array:
    """Slice a Fisher matrix to a subset of parameter indices."""
    idx = list(idx)
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be square 2D, got {F.shape}")
    return F[np.ix_(idx, idx)]


def fisher_to_cov(F: Array, *, rcond: float | None = None) -> Array:
    """
    Convert Fisher matrix to parameter covariance via pseudoinverse.

    Args:
        F: Fisher matrix (p, p)
        rcond: optional rcond passed to np.linalg.pinv
    """
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError(f"F must be square 2D, got {F.shape}")
    if rcond is None:
        return np.linalg.pinv(F)
    return np.linalg.pinv(F, rcond=rcond)


# ---------------------------------------------------------------------------
# GetDist conversion
# ---------------------------------------------------------------------------

def fisher_to_gaussiannd(
    theta0: Array,
    F: Array,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    label: str = "Fisher (Gaussian)",
    rcond: float | None = None,
):
    """
    Return a GetDist GaussianND object corresponding to the Fisher Gaussian.

    Notes:
      - This is the cleanest Fisher->GetDist representation (analytic contours).
      - GaussianND expects a covariance by default; we compute cov = pinv(F).
    """
    try:
        from getdist.gaussian_mixtures import GaussianND
    except ImportError as e:
        raise ImportError("GetDist integration requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)

    p = theta0.size
    if F.shape != (p, p):
        raise ValueError(f"F must have shape {(p,p)}, got {F.shape}")
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    cov = fisher_to_cov(F, rcond=rcond)

    return GaussianND(
        mean=theta0,
        cov=cov,
        names=list(names),
        labels=list(labels),
        label=label,
    )


def fisher_to_mcsamples(
    theta0: Array,
    F: Array,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 200_000,
    seed: int | None = None,
    label: str = "Fisher (samples)",
    rcond: float | None = None,
    store_loglikes: bool = True,
):
    """
    Return a GetDist MCSamples drawn from the Fisher Gaussian.

    Notes:
      - If store_loglikes=True, we store -log(posterior) up to an additive constant:
          -log p = 0.5 * d^T F d  (+ const), for flat priors.
    """
    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("GetDist integration requires `getdist` (pip install getdist).") from e

    rng = np.random.default_rng(seed)

    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)

    p = theta0.size
    if F.shape != (p, p):
        raise ValueError(f"F must have shape {(p,p)}, got {F.shape}")
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    cov = fisher_to_cov(F, rcond=rcond)

    # sample N(theta0, cov)
    jitter = 1e-12 * np.trace(cov) / max(p, 1)
    L = np.linalg.cholesky(cov + jitter * np.eye(p))
    samples = theta0[None, :] + rng.standard_normal((nsamp, p)) @ L.T

    loglikes = None
    if store_loglikes:
        d = samples - theta0[None, :]
        quad = np.einsum("ni,ij,nj->n", d, F, d)
        quad -= np.nanmin(quad)  # shift const (safe)
        loglikes = 0.5 * quad  # -log posterior up to const

    return MCSamples(
        samples=samples,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )
