"""Prior utilities (core priors + router/combiner).

Priors are represented as callables:

    logprior(theta) -> float

The return value is interpreted as a log-density defined up to an additive
constant. Returning ``-np.inf`` denotes zero probability (hard exclusion).

These priors are designed to be used when constructing log-posteriors for
sampling (e.g., Fisher/DALI approximate posteriors) or when evaluating posterior
surfaces. GetDist does not apply priors; it only plots the samples or Gaussian
approximations you provide.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.linalg import invert_covariance, normalize_covariance
from derivkit.utils.numerics import (
    apply_hard_bounds,
    get_index_value,
    logsumexp_1d,
    sum_terms,
)

__all__ = [
    "get_prior",
    "prior_none",
    "prior_uniform",
    "prior_gaussian",
    "prior_gaussian_diag",
    "prior_log_uniform",
    "prior_jeffreys",
    "prior_half_normal",
    "prior_half_cauchy",
    "prior_log_normal",
    "prior_beta",
    "prior_gaussian_mixture",
    "make_prior_term",
    "combine_priors",
]


def _prior_none_impl(theta: NDArray[np.floating]) -> float:
    """Internal helper that implements an improper flat prior.

    Args:
        theta: Parameter vector (unused).

    Returns:
        0.0
    """
    _ = theta  # unused
    return 0.0


def _prior_1d_impl(
    theta0: NDArray[np.floating],
    *,
    index: int,
    domain: str,
    kind: str,
    a: float = 0.0,
    b: float = 1.0,
) -> float:
    """A geneeric 1D prior implementation.

    Args:
        theta0: Parameter vector.
        index: Index of the parameter to which the prior applies.
        domain: Domain restriction ("positive", "nonnegative", "unit_open").
        kind: Prior kind ("log_uniform", "half_normal", "half_cauchy", "log_normal", "beta").
        a: First prior parameter (meaning depends on `kind`).
        b: Second prior parameter (meaning depends on `kind`).

    Returns:
        Log-prior value for the specified parameter.
    """
    x = get_index_value(theta0, index, name="theta0")

    # domain gate
    if domain == "positive":
        if x <= 0.0:
            return -np.inf
    elif domain == "nonnegative":
        if x < 0.0:
            return -np.inf
    elif domain == "unit_open":
        if x <= 0.0 or x >= 1.0:
            return -np.inf
    else:
        raise ValueError(f"unknown domain '{domain}'")

    # We define different prior shapes here
    if kind == "log_uniform":
        return float(-np.log(x))

    if kind == "half_normal":
        sigma = float(a)
        # We need to validate here too, in case someone partials directly to _prior_1d_impl
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        return float(-0.5 * (x / sigma) ** 2)

    if kind == "half_cauchy":
        scale = float(a)
        if scale <= 0.0:
            raise ValueError("scale must be > 0")
        t = x / scale
        return float(-np.log1p(t * t))

    if kind == "log_normal":
        mu = float(a)
        sigma = float(b)
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        lx = np.log(x)
        z = (lx - mu) / sigma
        return float(-0.5 * z * z - lx)

    if kind == "beta":
        alpha = float(a)
        beta = float(b)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be > 0")
        return float((alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x))

    raise ValueError(f"unknown prior kind '{kind}'")


def _prior_gaussian_impl(
        theta0: NDArray[np.floating],
        *,
        mean: NDArray[np.floating],
        invcov: NDArray[np.floating]
) -> float:
    """Internal helper that implements a multivariate Gaussian log-prior term (up to an additive constant).

    Args:
        theta0: Parameter vector.
        mean: Mean vector.
        invcov: Inverse covariance matrix.

    Returns:
        Log-prior value.
    """
    th = np.asarray(theta0, dtype=float)
    mu = np.asarray(mean, dtype=float)
    cov_inv = np.asarray(invcov, dtype=float)

    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")
    if cov_inv.ndim != 2 or cov_inv.shape[0] != cov_inv.shape[1] or cov_inv.shape[0] != mu.size:
        raise ValueError(f"invcov must have shape (p, p) with p={mu.size}, got {cov_inv.shape}")
    if th.ndim != 1 or th.size != mu.size:
        raise ValueError(f"theta must have shape ({mu.size},), got {th.shape}")

    d = th - mu
    return float(-0.5 * (d @ cov_inv @ d))


def _prior_gaussian_diag_impl(
    theta: NDArray[np.floating],
    *,
    mean: NDArray[np.floating],
    invcov: NDArray[np.floating],
) -> float:
    """Internal helper that implements a diagonal multivariate Gaussian log-prior term (up to an additive constant).

    Args:
        theta: Parameter vector.
        mean: Mean vector.
        invcov: Inverse covariance matrix (must be diagonal).

    Returns:
        Log-prior value.
    """
    th = np.asarray(theta, dtype=float)
    mu = np.asarray(mean, dtype=float)
    inv_cov = np.asarray(invcov, dtype=float)

    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")
    p = mu.size
    if inv_cov.ndim != 2 or inv_cov.shape != (p, p):
        raise ValueError(f"invcov must have shape ({p},{p}), got {inv_cov.shape}")
    if th.ndim != 1 or th.size != p:
        raise ValueError(f"theta must have shape ({p},), got {th.shape}")

    # Optional strictness: require diagonal
    if not np.allclose(inv_cov, np.diag(np.diag(inv_cov))):
        raise ValueError("invcov must be diagonal for prior_gaussian_diag")

    inv_var = np.diag(inv_cov)
    if np.any(inv_var <= 0.0):
        raise ValueError("diagonal invcov entries must be > 0")

    d = th - mu
    return float(-0.5 * np.sum(d * d * inv_var))


def _prior_gaussian_mixture_impl(
    theta0: NDArray[np.floating],
    *,
    means: NDArray[np.floating],
    inv_covs: NDArray[np.floating],
    log_weights: NDArray[np.floating],
    log_component_norm: NDArray[np.floating],
) -> float:
    """Evaluates a Gaussian mixture log-prior at ``theta0``.

    This method computes the log of a weighted sum of Gaussian components:

        p(theta) = sum_k w_k * N(theta | mean_k, cov_k)

    The result is a log-density defined up to an additive constant.

    Notes:
        - ``inv_covs`` are the per-component inverse covariances (precision matrices).
        - ``log_weights`` are the mixture weights in log-space; they are typically
          normalized so that ``logsumexp(log_weights) = 0``.
        - ``log_component_norm`` controls whether per-component normalization is
          included. If it contains ``-0.5 * log|C_k|`` (or the equivalent computed
          from ``C_k^{-1}``), then components with different covariances get the
          correct relative normalization. If it is all zeros, the mixture keeps
          only the quadratic terms, which can be useful for “shape-only” priors.

    Args:
        theta0: Parameter vector ``theta`` with shape ``(p,)``.
        means: Component means with shape ``(K, p)``.
        inv_covs: Component inverse covariances with shape ``(K, p, p)``.
        log_weights: Log-weights for the ``K`` components with shape ``(K,)``.
        log_component_norm: Per-component log-normalization terms with shape
            ``(K,)`` (often ``-0.5 * log|C_k|``). Use zeros to omit this factor.

    Returns:
        The mixture log-prior value at ``theta0`` (a finite float or ``-np.inf`` if
        the caller enforces hard bounds elsewhere).
    """
    th = np.asarray(theta0, dtype=float)
    mus = np.asarray(means, dtype=float)
    inv_covs = np.asarray(inv_covs, dtype=float)
    lw = np.asarray(log_weights, dtype=float)
    lcn = np.asarray(log_component_norm, dtype=float)

    if th.ndim != 1:
        raise ValueError(f"theta must be 1D, got shape {th.shape}")
    if mus.ndim != 2:
        raise ValueError(f"means must be (K, p), got shape {mus.shape}")
    k, p = mus.shape
    if th.size != p:
        raise ValueError(f"theta length {th.size} != p {p}")
    if inv_covs.ndim != 3 or inv_covs.shape != (k, p, p):
        raise ValueError(f"icovs must be (K, p, p) with K={k}, p={p}, got shape {inv_covs.shape}")
    if lw.ndim != 1 or lw.size != k:
        raise ValueError(f"log_weights must be (K,), got shape {lw.shape}")
    if lcn.ndim != 1 or lcn.size != k:
        raise ValueError(f"log_component_norm must be (K,), got shape {lcn.shape}")

    vals = np.empty(k, dtype=float)
    for i in range(k):
        d = th - mus[i]
        vals[i] = lw[i] + lcn[i] - 0.5 * float(d @ inv_covs[i] @ d)

    return float(logsumexp_1d(vals))


def prior_none() -> Callable[[NDArray[np.floating]], float]:
    """Constructs an improper flat prior (constant log-density).

    Returns:
        Callable log-prior: logp(theta) -> 0.0
    """
    return _prior_none_impl


def prior_uniform(
    *,
    bounds: Sequence[tuple[float | None, float | None]],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a uniform prior with hard bounds.

    This method applies hard bounds to an otherwise flat prior
    (i.e., constant log-density within the bounds, ``-np.inf`` outside).

    Args:
        bounds: Sequence of (min, max) pairs for each parameter.
            Use None for unbounded sides.

    Returns:
        Callable log-prior: logp(theta) -> float
    """
    return apply_hard_bounds(_prior_none_impl, bounds=bounds)


def prior_gaussian(
    *,
    mean: NDArray[np.floating],
    cov: NDArray[np.floating] | None = None,
    inv_cov: NDArray[np.floating] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a multivariate Gaussian prior (up to an additive constant).

    Args:
        mean: Mean vector.
        cov: Covariance matrix (provide exactly one of `cov` or `invcov`).
        inv_cov: Inverse covariance matrix (provide exactly one of `cov` or `invcov`).

    Returns:
        Callable log-prior: logp(theta) -> float.

    Raises:
        ValueError: If neither or both of `cov` and `invcov` are provided,
            or if the provided covariance/inverse covariance cannot be
            normalized/validated.
    """
    if (cov is None) == (inv_cov is None):
        raise ValueError("Provide exactly one of `cov` or `invcov`.")

    mu = np.asarray(mean, dtype=float)
    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")

    if inv_cov is None:
        cov = normalize_covariance(cov, n_parameters=mu.size)
        cov_inv = invert_covariance(cov, warn_prefix="prior_gaussian")
    else:
        cov_inv = np.asarray(inv_cov, dtype=float)
        if cov_inv.ndim != 2 or cov_inv.shape != (mu.size, mu.size):
            raise ValueError(f"invcov must have shape (p,p) with p={mu.size}, got {cov_inv.shape}")
        if not np.all(np.isfinite(cov_inv)):
            raise ValueError("invcov contains non-finite values.")

    return partial(_prior_gaussian_impl, mean=mu, invcov=cov_inv)


def prior_gaussian_diag(
    *,
    mean: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a diagonal Gaussian prior (up to an additive constant).


    Args:
        mean: Mean vector.
        sigma: Standard deviation vector (must be positive).

    Returns:
        Callable log-prior: logp(theta) -> float.

    Raises:
        ValueError: If `mean` and `sigma` have incompatible shapes,
            or if any `sigma` entries are non-positive.
    """
    mu = np.asarray(mean, dtype=float)
    sig = np.asarray(sigma, dtype=float)

    if mu.ndim != 1 or sig.ndim != 1 or mu.shape != sig.shape:
        raise ValueError("mean and sigma must be 1D arrays with the same shape")
    if np.any(sig <= 0.0):
        raise ValueError("all sigma entries must be > 0")

    inv_var = 1.0 / (sig**2)
    return partial(_prior_gaussian_diag_impl, mean=mu, inv_var=inv_var)


def prior_log_uniform(*, index: int) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a log-uniform prior for a single positive parameter.

    This prior has density proportional to 1/x for x > 0, which means it is
    uniform when expressed in log(x). It is commonly used for scale parameters
    when equal weight per order of magnitude is desired.

    Args:
        index: Index of the parameter to which the prior applies.

    Note:
        For a positive scale parameter, the Jeffreys prior has the same form as a
        log-uniform prior. This is why ``prior_log_uniform`` and ``prior_jeffreys``
        use the same implementation here. Both names are kept because they reflect
        different motivations in user code.
    """
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_uniform")


def prior_jeffreys(*, index: int) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a Jeffreys prior for a single positive scale parameter.

    The Jeffreys prior is defined using the Fisher information of the parameter.
    For a positive scale parameter, this leads to a prior proportional to 1/x,
    which is the same functional form as a log-uniform prior.

    Note:
        Although the implementation matches ``prior_log_uniform``, the name
        ``prior_jeffreys`` emphasizes the motivation (reparameterization
        invariance for scale parameters) rather than “uniform in log-space”.
    """
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_uniform")


def prior_half_normal(*, index: int, sigma: float) -> Callable[[NDArray[np.floating]], float]:
    """Construct a half-normal prior for a single non-negative parameter."""
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_normal", a=s)


def prior_half_cauchy(*, index: int, scale: float) -> Callable[[NDArray[np.floating]], float]:
    """Construct a half-Cauchy prior for a single non-negative parameter."""
    s = float(scale)
    if s <= 0.0:
        raise ValueError("scale must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_cauchy", a=s)


def prior_log_normal(*, index: int, mu: float, sigma: float) -> Callable[[NDArray[np.floating]], float]:
    """Construct a log-normal prior for a single positive parameter."""
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_normal", a=float(mu), b=s)


def prior_beta(*, index: int, alpha: float, beta: float) -> Callable[[NDArray[np.floating]], float]:
    """Construct a Beta prior for a single parameter in (0, 1)."""
    a = float(alpha)
    b = float(beta)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("alpha and beta must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="unit_open", kind="beta", a=a, b=b)


def prior_gaussian_mixture(
    *,
    means: NDArray[np.floating],
    covs: NDArray[np.floating] | None = None,
    icovs: NDArray[np.floating] | None = None,
    weights: NDArray[np.floating] | None = None,
    log_weights: NDArray[np.floating] | None = None,
    include_component_norm: bool = True,
) -> Callable[[NDArray[np.floating]], float]:
    """Construct a Gaussian mixture prior (up to an additive constant).

    The mixture is:
        p(theta) = sum_k w_k * N(theta | mean_k, cov_k)

    Provide exactly one of:
        - covs: (K, p, p)
        - icovs: (K, p, p)

    Provide exactly one of:
        - weights: (K,) non-negative (normalized internally)
        - log_weights: (K,) (normalized internally in log-space)

    Args:
        means: Component means with shape (K, p).
        covs: Component covariances with shape (K, p, p).
        icovs: Component inverse covariances with shape (K, p, p).
        weights: Mixture weights with shape (K,). Can include zeros.
        log_weights: Log-weights with shape (K,). Can include -inf entries.
        include_component_norm: If True (default), include the per-component
            Gaussian normalization factor proportional to |C_k|^{-1/2}.
            This is important for *mixtures* when covariances differ.

    Returns:
        Callable log-prior: logp(theta) -> float.
    """
    mus = np.asarray(means, dtype=float)
    if mus.ndim != 2:
        raise ValueError(f"means must be (K, p), got shape {mus.shape}")
    k, p = mus.shape

    include = bool(include_component_norm)

    if (covs is None) == (icovs is None):
        raise ValueError("Provide exactly one of `covs` or `icovs`.")

    if icovs is None:
        C = np.asarray(covs, dtype=float)
        if C.ndim != 3 or C.shape != (k, p, p):
            raise ValueError(f"covs must be (K, p, p) with K={k}, p={p}, got shape {C.shape}")

        # Per-component log-normalization: -0.5 * log|C_k|
        if include:
            log_component_norm = np.empty(k, dtype=float)
            for i in range(k):
                sign, logdet = np.linalg.slogdet(C[i])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(
                        "include_component_norm=True requires each cov_k to be positive-definite "
                        "(slogdet sign>0 and finite)."
                    )
                log_component_norm[i] = -0.5 * logdet
        else:
            log_component_norm = np.zeros(k, dtype=float)

        Ci = np.empty_like(C)
        for i in range(k):
            Ci[i] = invert_covariance(C[i], warn_prefix="prior_gaussian_mixture")


    else:
        Ci = np.asarray(icovs, dtype=float)
        if Ci.ndim != 3 or Ci.shape != (k, p, p):
            raise ValueError(f"icovs must be (K, p, p) with K={k}, p={p}, got shape {Ci.shape}")

        # Using |C| = |C^{-1}|^{-1} => -0.5 log|C| = +0.5 log|C^{-1}|
        if include:
            log_component_norm = np.empty(k, dtype=float)
            for i in range(k):
                sign, logdet = np.linalg.slogdet(Ci[i])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(
                        "include_component_norm=True requires each icov_k to have positive determinant "
                        "(slogdet sign>0 and finite)."
                    )
                log_component_norm[i] = 0.5 * logdet
        else:
            log_component_norm = np.zeros(k, dtype=float)

    if (weights is None) == (log_weights is None):
        raise ValueError("Provide exactly one of `weights` or `log_weights`.")

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size != k:
            raise ValueError(f"weights must be (K,) with K={k}, got shape {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w = w / s
        lw = np.log(w)  # zeros -> -inf is OK
    else:
        lw_in = np.asarray(log_weights, dtype=float)
        if lw_in.ndim != 1 or lw_in.size != k:
            raise ValueError(f"log_weights must be (K,) with K={k}, got shape {lw_in.shape}")
        lw = lw_in - logsumexp_1d(lw_in)

    if not (np.all(np.isfinite(mus)) and np.all(np.isfinite(Ci)) and np.all(np.isfinite(lw))):
        raise ValueError("mixture prior inputs must be finite")
    if not np.all(np.isfinite(log_component_norm)):
        raise ValueError("mixture prior log_component_norm must be finite")

    return partial(
        _prior_gaussian_mixture_impl,
        means=mus,
        icovs=Ci,
        log_weights=lw,
        log_component_norm=log_component_norm,
    )


_PRIOR_REGISTRY: dict[str, Callable[..., Callable[[NDArray[np.floating]], float]]] = {
    "none": prior_none,
    "uniform": prior_uniform,
    "gaussian": prior_gaussian,
    "gaussian_diag": prior_gaussian_diag,
    "log_uniform": prior_log_uniform,
    "jeffreys": prior_jeffreys,
    "half_normal": prior_half_normal,
    "half_cauchy": prior_half_cauchy,
    "log_normal": prior_log_normal,
    "beta": prior_beta,
    "gaussian_mixture": prior_gaussian_mixture,
}


def make_prior_term(spec: dict[str, Any]) -> Callable[[NDArray[np.floating]], float]:
    """Build a single prior term from a configuration dictionary.

    Expected format:
        {"name": "<prior_name>", "params": {...}, "bounds": optional_bounds}

    Notes:
        - For name="uniform", bounds must be provided via params={"bounds": ...}.
          The top-level "bounds" key is reserved for optional *extra* hard bounds
          on non-uniform terms.
    """
    name = str(spec.get("name", "")).strip().lower()
    if not name:
        raise ValueError("prior spec must include non-empty 'name'")
    if name not in _PRIOR_REGISTRY:
        raise ValueError(f"Unknown prior name '{name}'")

    params = dict(spec.get("params", {}))
    term_bounds = spec.get("bounds", None)

    if name == "uniform" and term_bounds is not None:
        raise ValueError(
            "Uniform prior takes bounds via params={'bounds': ...}. "
            "Do not use the top-level 'bounds' field for name='uniform'."
        )

    term = _PRIOR_REGISTRY[name](**params)
    return apply_hard_bounds(term, bounds=term_bounds)



def combine_priors(
    specs: Sequence[dict[str, Any]] | None = None,
    *,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    default: str = "none",
) -> Callable[[NDArray[np.floating]], float]:
    """Combine multiple prior terms into a single log-prior."""
    specs = [] if specs is None else list(specs)

    if len(specs) == 0:
        if bounds is not None:
            return prior_uniform(bounds=bounds)

        key = str(default).strip().lower()
        if key != "none":
            raise ValueError(
                "default prior must be 'none' unless `bounds` is provided "
                "(other priors require parameters; use `specs` instead)."
            )
        return prior_none()

    terms = [make_prior_term(s) for s in specs]
    combined = sum_terms(*terms)
    return apply_hard_bounds(combined, bounds=bounds)


def get_prior(
    name: str = "none",
    /,
    *,
    params: dict[str, Any] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Return a prior callable from a name + params (optionally with hard bounds).

    Examples:
        logp = get_prior("none")
        logp = get_prior("gaussian_diag", params={"mean": mu, "sigma": sig})
        logp = get_prior("log_uniform", params={"index": 0})
        logp = get_prior("uniform", params={"bounds": bounds})  # uniform uses its own bounds
        logp = get_prior("gaussian", params={"mean": mu, "cov": C}, bounds=global_bounds)
    """
    key = str(name).strip().lower()
    if key not in _PRIOR_REGISTRY:
        raise ValueError(f"Unknown prior name '{name}'")

    p = {} if params is None else dict(params)
    prior = _PRIOR_REGISTRY[key](**p)

    # optional *global* hard bounds (separate from e.g. uniform's own bounds)
    return apply_hard_bounds(prior, bounds=bounds)
