from __future__ import annotations

from typing import Sequence

import numpy as np

Array = np.ndarray

__all__ = [
    "slice_tensors",
    "dali_to_mcsamples_importance",
    "dali_to_mcsamples_emcee",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _validate_shapes(theta0: Array, F: Array, G: Array, H: Array | None) -> None:
    theta0 = np.asarray(theta0)
    F = np.asarray(F)
    G = np.asarray(G)
    p = theta0.shape[0]
    if theta0.ndim != 1:
        raise ValueError(f"theta0 must be 1D, got {theta0.shape}")
    if F.shape != (p, p):
        raise ValueError(f"F must have shape {(p,p)}, got {F.shape}")
    if G.shape != (p, p, p):
        raise ValueError(f"G must have shape {(p,p,p)}, got {G.shape}")
    if H is not None and np.asarray(H).shape != (p, p, p, p):
        raise ValueError(f"H must have shape {(p,p,p,p)}, got {np.asarray(H).shape}")


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

    convention:
      - "delta_chi2":
          Δχ² = d^T F d + (1/3) G:d^3 + (1/12) H:d^4
          log posterior = -0.5 * Δχ²

      - "matplotlib_loglike":
          Matches the convention in your plotting snippet:
            logp = -0.5 dFd -0.5 (G:d^3) -0.125 (H:d^4)
          which implies:
            Δχ² = dFd + (G:d^3) + 0.25 (H:d^4)
          so that logp = -0.5 * Δχ².
    """
    theta = np.asarray(theta, float)
    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    H = None if H is None else np.asarray(H, float)
    _validate_shapes(theta0, F, G, H)

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
) -> float:
    """Log posterior up to a constant, with optional hard flat bounds."""
    if hard_bounds is not None:
        for t, (lo, hi) in zip(theta, hard_bounds):
            if (lo is not None and t < lo) or (hi is not None and t > hi):
                return -np.inf
    chi2 = delta_chi2_dali(theta, theta0, F, G, H, convention=convention)
    return -0.5 * chi2


# ---------------------------------------------------------------------------
# GetDist conversion
# ---------------------------------------------------------------------------

def dali_to_mcsamples_importance(
    theta0: Array,
    F: Array,
    G: Array,
    H: Array | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsamp: int = 200_000,
    proposal_scale: float = 1.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (importance)",
):
    """
    Return a GetDist MCSamples using Fisher-Gaussian proposal + DALI importance weights.

    Notes
    -----
    - GetDist expects loglikes = -log(posterior). We provide loglikes = 0.5 * Δχ² (shifted).
    - `proposal_scale` inflates the Fisher covariance for broader/non-Gaussian posteriors.
    """
    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("GetDist integration requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    H = None if H is None else np.asarray(H, float)
    _validate_shapes(theta0, F, G, H)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    rng = np.random.default_rng(seed)

    C = np.linalg.pinv(F)
    C_prop = (proposal_scale**2) * C

    # Cholesky with tiny jitter for stability
    jitter = 1e-12 * np.trace(C_prop) / max(p, 1)
    L = np.linalg.cholesky(C_prop + jitter * np.eye(p))

    samples = theta0[None, :] + rng.standard_normal((nsamp, p)) @ L.T

    if hard_bounds is not None:
        mask = np.ones(samples.shape[0], dtype=bool)
        for j, (lo, hi) in enumerate(hard_bounds):
            if lo is not None:
                mask &= samples[:, j] >= lo
            if hi is not None:
                mask &= samples[:, j] <= hi
        samples = samples[mask]
        if samples.shape[0] == 0:
            raise RuntimeError("All proposal samples rejected by hard_bounds.")

    chi2 = np.array(
        [delta_chi2_dali(s, theta0, F, G, H, convention=convention) for s in samples],
        dtype=float,
    )
    chi2 -= np.nanmin(chi2)  # shift const (safe)

    weights = np.exp(-0.5 * chi2)
    loglikes = 0.5 * chi2  # -log posterior (up to const)

    return MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=list(names),
        labels=list(labels),
        label=label,
    )


def dali_to_mcsamples_emcee(
    theta0: Array,
    F: Array,
    G: Array,
    H: Array | None,
    *,
    names: Sequence[str],
    labels: Sequence[str],
    nsteps: int = 10_000,
    burn: int = 2_000,
    thin: int = 2,
    nwalkers: int | None = None,
    proposal_scale: float = 0.5,
    convention: str = "delta_chi2",
    seed: int | None = None,
    hard_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    label: str = "DALI (emcee)",
):
    """
    Run emcee on the DALI posterior and return GetDist MCSamples.

    Requires: `pip install emcee`
    """
    try:
        import emcee
    except ImportError as e:
        raise ImportError("This function requires `emcee` (pip install emcee).") from e

    try:
        from getdist import MCSamples
    except ImportError as e:
        raise ImportError("GetDist integration requires `getdist` (pip install getdist).") from e

    theta0 = np.asarray(theta0, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    H = None if H is None else np.asarray(H, float)
    _validate_shapes(theta0, F, G, H)

    p = theta0.size
    if len(names) != p or len(labels) != p:
        raise ValueError("names/labels must match number of parameters")

    if nwalkers is None:
        nwalkers = max(32, 8 * p)

    rng = np.random.default_rng(seed)

    C = np.linalg.pinv(F)
    C_prop = proposal_scale * C
    jitter = 1e-12 * np.trace(C_prop) / max(p, 1)
    L = np.linalg.cholesky(C_prop + jitter * np.eye(p))

    def log_prob(th: Array) -> float:
        return logpost_dali(
            th, theta0, F, G, H,
            convention=convention,
            hard_bounds=hard_bounds,
        )

    p0 = theta0[None, :] + rng.standard_normal((nwalkers, p)) @ L.T

    sampler = emcee.EnsembleSampler(nwalkers, p, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=True)

    chain = sampler.get_chain(discard=burn, thin=thin)      # (n, nwalkers, p)
    logp = sampler.get_log_prob(discard=burn, thin=thin)    # (n, nwalkers)

    chain_list = [chain[:, i, :] for i in range(chain.shape[1])]
    loglikes_list = [-logp[:, i] for i in range(logp.shape[1])]  # GetDist = -log posterior

    return MCSamples(
        samples=chain_list,
        loglikes=loglikes_list,
        names=list(names),
        labels=list(labels),
        label=label,
    )
