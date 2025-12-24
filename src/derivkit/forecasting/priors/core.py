"""Prior utilities (core priors + unified builder).

Priors are represented as callables:

    logprior(theta) -> float

The return value is interpreted as a log-density defined up to an additive
constant. Returning ``-np.inf`` denotes zero probability (hard exclusion).

These priors are designed to be used when constructing log-posteriors for
sampling (e.g., Fisher/DALI approximate posteriors) or when evaluating posterior
surfaces. GetDist plots what is provided; priors must be applied when generating
samples (or explicitly added to the Fisher matrix), not automatically inferred.
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
    "build_prior",
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
    theta: NDArray[np.floating],
    *,
    index: int,
    domain: str,
    kind: str,
    a: float = 0.0,
    b: float = 1.0,
) -> float:
    """A generic 1D prior implementation.

    Args:
        theta: Parameter vector.
        index: Index of the parameter to which the prior applies.
        domain: Domain restriction ("positive", "nonnegative", "unit_open").
        kind: Prior kind ("log_uniform", "half_normal", "half_cauchy", "log_normal", "beta").
        a: First prior parameter (meaning depends on `kind`).
        b: Second prior parameter (meaning depends on `kind`).

    Returns:
        Log-prior value for the specified parameter.
    """
    x = get_index_value(theta, index, name="theta")

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

    # Define different prior shapes here
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
        mean = float(a)
        sigma = float(b)
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        lx = np.log(x)
        z = (lx - mean) / sigma
        return float(-0.5 * z * z - lx)

    if kind == "beta":
        alpha = float(a)
        beta = float(b)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be > 0")
        return float((alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x))

    raise ValueError(f"unknown prior kind '{kind}'")


def _prior_gaussian_impl(
        theta: NDArray[np.floating],
        *,
        mean: NDArray[np.floating],
        inv_cov: NDArray[np.floating]
) -> float:
    """Internal helper that implements a multivariate Gaussian log-prior term (up to an additive constant).

    Args:
        theta: Parameter vector.
        mean: Mean vector.
        inv_cov: Inverse covariance matrix.

    Returns:
        Log-prior value.
    """
    th = np.asarray(theta, dtype=float)
    mu = np.asarray(mean, dtype=float)
    inv_cov = np.asarray(inv_cov, dtype=float)

    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")
    if inv_cov.ndim != 2 or inv_cov.shape[0] != inv_cov.shape[1] or inv_cov.shape[0] != mu.size:
        raise ValueError(f"inv_cov must have shape (p, p) with p={mu.size}, got {inv_cov.shape}")
    if th.ndim != 1 or th.size != mu.size:
        raise ValueError(f"theta must have shape ({mu.size},), got {th.shape}")

    d = th - mu
    return float(-0.5 * (d @ inv_cov @ d))


def _prior_gaussian_diag_impl(
    theta: NDArray[np.floating],
    *,
    mean: NDArray[np.floating],
    inv_cov: NDArray[np.floating],
) -> float:
    """Internal helper that implements a diagonal multivariate Gaussian log-prior term (up to an additive constant).

    Args:
        theta: Parameter vector.
        mean: Mean vector.
        inv_cov: Inverse covariance matrix (must be diagonal).

    Returns:
        Log-prior value.
    """
    th = np.asarray(theta, dtype=float)
    mu = np.asarray(mean, dtype=float)
    inv_cov = np.asarray(inv_cov, dtype=float)

    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")
    p = mu.size
    if inv_cov.ndim != 2 or inv_cov.shape != (p, p):
        raise ValueError(f"inv_cov must have shape ({p},{p}), got {inv_cov.shape}")
    if th.ndim != 1 or th.size != p:
        raise ValueError(f"theta must have shape ({p},), got {th.shape}")

    inv_var = np.diag(inv_cov)
    off_diag = inv_cov.copy()
    np.fill_diagonal(off_diag, 0.0)
    if not np.allclose(off_diag, 0.0, rtol=1e-12, atol=1e-12):
        raise ValueError("inv_cov must be diagonal for prior_gaussian_diag")
    if np.any(inv_var <= 0.0):
        raise ValueError("diagonal inv_cov entries must be > 0")

    d = th - mu
    return float(-0.5 * np.sum(d * d * inv_var))


def _prior_gaussian_mixture_impl(
    theta: NDArray[np.floating],
    *,
    means: NDArray[np.floating],
    inv_covs: NDArray[np.floating],
    log_weights: NDArray[np.floating],
    log_component_norm: NDArray[np.floating],
) -> float:
    """Evaluates a Gaussian mixture log-prior at ``theta``.

    This function computes the log of a weighted sum of Gaussian components:

        p(theta) = sum_n w_n * N(theta | mean_n, cov_n)
    where N(theta | mean, cov) is the multivariate Gaussian density with the
    specified mean and covariance; w_n are the mixture weights (in log-space); and
    the sum runs over the n=1..N components.

    The result is a log-density defined up to an additive constant.

    Notes:
        - ``inv_covs`` are the per-component inverse covariances (precision matrices).
        - ``log_weights`` are the mixture weights in log-space; they are typically
          normalized so that ``logsumexp(log_weights) = 0``.
        - ``log_component_norm`` controls whether per-component normalization is
          included. If it contains ``-0.5 * log|C_n|`` (or the equivalent computed
          from ``C_n^{-1}``), then components with different covariances get the
          correct relative normalization. If it is all zeros, the mixture keeps
          only the quadratic terms, which can be useful for “shape-only” priors.

    Args:
        theta: Parameter vector ``theta`` with shape ``(p,)``.
        means: Component means with shape ``(n, p)``.
        inv_covs: Component inverse covariances with shape ``(n, p, p)``.
        log_weights: Log-weights for the ``n`` components with shape ``(n,)``.
        log_component_norm: Per-component log-normalization terms with shape
            ``(n,)`` (often ``-0.5 * log|C_n|``). Use zeros to omit this factor.

    Returns:
        The mixture log-prior value at ``theta`` (a finite float or ``-np.inf`` if
        the caller enforces hard bounds elsewhere).
    """
    th = np.asarray(theta, dtype=float)
    mus = np.asarray(means, dtype=float)
    inv_covs = np.asarray(inv_covs, dtype=float)
    lw = np.asarray(log_weights, dtype=float)
    lcn = np.asarray(log_component_norm, dtype=float)

    if th.ndim != 1:
        raise ValueError(f"theta must be 1D, got shape {th.shape}")
    if mus.ndim != 2:
        raise ValueError(f"means must be (n, p), got shape {mus.shape}")
    n, p = mus.shape
    if th.size != p:
        raise ValueError(f"theta length {th.size} != p {p}")
    if inv_covs.ndim != 3 or inv_covs.shape != (n, p, p):
        raise ValueError(f"inv_covs must be (n, p, p) with n={n}, p={p}, got shape {inv_covs.shape}")
    if lw.ndim != 1 or lw.size != n:
        raise ValueError(f"log_weights must be (n,), got shape {lw.shape}")
    if lcn.ndim != 1 or lcn.size != n:
        raise ValueError(f"log_component_norm must be (n,), got shape {lcn.shape}")

    vals = np.empty(n, dtype=float)
    for i in range(n):
        d = th - mus[i]
        vals[i] = lw[i] + lcn[i] - 0.5 * float(d @ inv_covs[i] @ d)

    return float(logsumexp_1d(vals))


def prior_none() -> Callable[[NDArray[np.floating]], float]:
    """Constructs an improper flat prior (constant log-density).

    This prior has density proportional to 1 everywhere.

    Returns:
        Callable log-prior: logp(theta) -> 0.0
    """
    return _prior_none_impl


def prior_uniform(
    *,
    bounds: Sequence[tuple[float | None, float | None]],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a uniform prior with hard bounds.

    This prior has density proportional to 1 within the specified bounds
    and zero outside. The log-density is constant (up to an additive constant)
    within the bounds and ``-np.inf`` outside.

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

    This prior has density proportional to
    exp(-0.5 * (x - mean)^T @ inv_cov @ (x - mean)) with inv_cov being the
    inverse of the provided covariance matrix and x being the parameter vector.

    Args:
        mean: Mean vector.
        cov: Covariance matrix (provide exactly one of `cov` or `inv_cov`).
        inv_cov: Inverse covariance matrix (provide exactly one of `cov` or `inv_cov`).

    Returns:
        Callable log-prior: logp(theta) -> float.

    Raises:
        ValueError: If neither or both of `cov` and `inv_cov` are provided,
            or if the provided covariance/inverse covariance cannot be
            normalized/validated.
    """
    if (cov is None) == (inv_cov is None):
        raise ValueError("Provide exactly one of `cov` or `inv_cov`.")

    mu = np.asarray(mean, dtype=float)
    if mu.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {mu.shape}")

    if inv_cov is None:
        cov = normalize_covariance(cov, n_parameters=mu.size)
        inv_cov = invert_covariance(cov, warn_prefix="prior_gaussian")
    else:
        inv_cov = np.asarray(inv_cov, dtype=float)
        if inv_cov.ndim != 2 or inv_cov.shape != (mu.size, mu.size):
            raise ValueError(f"inv_cov must have shape (p,p) with p={mu.size}, got {inv_cov.shape}")
        if not np.all(np.isfinite(inv_cov)):
            raise ValueError("inv_cov contains non-finite values.")

    return partial(_prior_gaussian_impl, mean=mu, inv_cov=inv_cov)


def prior_gaussian_diag(
    *,
    mean: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a diagonal Gaussian prior (up to an additive constant).

    This prior has density proportional to
    exp(-0.5 * sum_i ((x_i - mean_i) / sigma_i)^2), with independent components.

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

    inv_cov = np.diag(1.0 / (sig ** 2))
    return partial(_prior_gaussian_diag_impl, mean=mu, inv_cov=inv_cov)


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
    For a positive scale parameter, this leads to a prior proportional to 1/x
    (with x being the scale parameter that is > 0), which is the same functional
    form as a log-uniform prior.

    Note:
        Although the implementation matches ``prior_log_uniform``, the name
        ``prior_jeffreys`` emphasizes the motivation (reparameterization
        invariance for scale parameters) rather than “uniform in log-space”.
    """
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_uniform")


def prior_half_normal(*, index: int, sigma: float) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a half-normal prior for a single non-negative parameter.

    This prior has density proportional to exp(-0.5 * (x/sigma)^2) for x >= 0.

    Args:
        index: Index of the parameter to which the prior applies.
        sigma: Standard deviation of the underlying normal distribution.

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `sigma` is not positive.
    """
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_normal", a=s)


def prior_half_cauchy(*, index: int, scale: float) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a half-Cauchy prior for a single non-negative parameter.

    This prior has density proportional to 1 / (1 + (x/scale)^2) for x >= 0,
    with x being the parameter.

    Args:
        index: Index of the parameter to which the prior applies.
        scale: Scale parameter of the half-Cauchy distribution.

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `scale` is not positive.
    """
    s = float(scale)
    if s <= 0.0:
        raise ValueError("scale must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_cauchy", a=s)


def prior_log_normal(
        *,
        index: int,
        mean_log: float,
        sigma_log: float
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a log-normal prior for a single positive parameter.

    This prior has density proportional to
    exp(-0.5 * ((log(x) - mean_log) / sigma_log)^2) / x for x > 0.

    Args:
        index: Index of the parameter to which the prior applies.
        mean_log: Mean of the underlying normal distribution (in log-space).
        sigma_log: Standard deviation of the underlying normal distribution (in log-space).

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `sigma_log` is not positive.
    """
    s = float(sigma_log)
    if s <= 0.0:
        raise ValueError("sigma_log must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_normal", a=float(mean_log), b=s)


def prior_beta(*, index: int, alpha: float, beta: float) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a Beta prior for a single parameter in (0, 1).

    This prior has density proportional to x^(alpha-1) * (1-x)^(beta-1) for x in (0, 1).

    Args:
        index: Index of the parameter to which the prior applies.
        alpha: Alpha shape parameter (> 0).
        beta: Beta shape parameter (> 0).

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `alpha` or `beta` are not positive.
    """
    a = float(alpha)
    b = float(beta)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("alpha and beta must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="unit_open", kind="beta", a=a, b=b)


def prior_gaussian_mixture(
    *,
    means: NDArray[np.floating],
    covs: NDArray[np.floating] | None = None,
    inv_covs: NDArray[np.floating] | None = None,
    weights: NDArray[np.floating] | None = None,
    log_weights: NDArray[np.floating] | None = None,
    include_component_norm: bool = True,
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a Gaussian mixture prior (up to an additive constant).

    This prior has density proportional to a weighted sum of Gaussian components.

    The mixture is:
        p(theta) = sum_n w_n * N(theta | mean_n, cov_n)
    where N(theta | mean, cov) is the multivariate Gaussian density with the
    specified mean and covariance; w_n are the mixture weights (non-negative,
    summing to 1); and the sum runs over the n=1..N components.

    Provide exactly one of:
        - covs: (n, p, p)
        - inv_covs: (n, p, p)
    Here, n is the number of components and p is the parameter dimension.

    Provide exactly one of:
        - weights: (n,) non-negative (normalized internally)
        - log_weights: (n,) (normalized internally in log-space)

    Args:
        means: Component means with shape (n, p).
        covs: Component covariances with shape (n, p, p).
        inv_covs: Component inverse covariances with shape (n, p, p).
        weights: Mixture weights with shape (n,). Can include zeros.
        log_weights: Log-weights with shape (n,). Can include -inf entries.
        include_component_norm: If True (default), include the per-component
            Gaussian normalization factor proportional to |C_n|^{-1/2}.
            This is important for *mixtures* when covariances differ.

    Returns:
        Callable log-prior: logp(theta) -> float.
    """
    mus = np.asarray(means, dtype=float)
    if mus.ndim != 2:
        raise ValueError(f"means must be (n, p), got shape {mus.shape}")
    n, p = mus.shape

    include = bool(include_component_norm)

    if (covs is None) == (inv_covs is None):
        raise ValueError("Provide exactly one of `covs` or `inv_covs`.")

    if inv_covs is None:
        cov = np.asarray(covs, dtype=float)
        if cov.ndim != 3 or cov.shape != (n, p, p):
            raise ValueError(f"covs must be (n, p, p) with n={n}, p={p}, got shape {cov.shape}")

        if include:
            log_component_norm = np.empty(n, dtype=float)
            for i in range(n):
                sign, logdet = np.linalg.slogdet(cov[i])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(
                        "include_component_norm=True requires each cov_n to be positive-definite "
                        "(slogdet sign>0 and finite)."
                    )
                log_component_norm[i] = -0.5 * logdet
        else:
            log_component_norm = np.zeros(n, dtype=float)

        inv_cov_n = np.empty_like(cov)
        for i in range(n):
            inv_cov_n[i] = invert_covariance(cov[i], warn_prefix="prior_gaussian_mixture")

    else:
        inv_cov_n = np.asarray(inv_covs, dtype=float)
        if inv_cov_n.ndim != 3 or inv_cov_n.shape != (n, p, p):
            raise ValueError(f"inv_covs must be (n, p, p) with n={n}, p={p}, got shape {inv_cov_n.shape}")

        if include:
            log_component_norm = np.empty(n, dtype=float)
            for i in range(n):
                sign, logdet = np.linalg.slogdet(inv_cov_n[i])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(
                        "include_component_norm=True requires each inv_cov_n to have positive determinant "
                        "(slogdet sign>0 and finite)."
                    )
                # -0.5 log|C| = +0.5 log|C^{-1}|
                log_component_norm[i] = 0.5 * logdet
        else:
            log_component_norm = np.zeros(n, dtype=float)

    if (weights is None) == (log_weights is None):
        raise ValueError("Provide exactly one of `weights` or `log_weights`.")

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size != n:
            raise ValueError(f"weights must be (n,) with n={n}, got shape {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w = w / s
        # Allow zero weights without triggering "divide by zero encountered in log"
        lw = np.full_like(w, -np.inf, dtype=float)
        np.log(w, out=lw, where=(w > 0))
    else:
        lw_in = np.asarray(log_weights, dtype=float)
        if lw_in.ndim != 1 or lw_in.size != n:
            raise ValueError(f"log_weights must be (n,) with n={n}, got shape {lw_in.shape}")
        lw = lw_in - logsumexp_1d(lw_in)

    if not (np.all(np.isfinite(mus)) and np.all(np.isfinite(inv_cov_n))):
        raise ValueError("mixture prior inputs must be finite")

    # allow -inf (zero-weight components), forbid nan / +inf
    if np.any(np.isnan(lw)) or np.any(lw == np.inf):
        raise ValueError("log_weights must not contain nan or +inf")

    return partial(
        _prior_gaussian_mixture_impl,
        means=mus,
        inv_covs=inv_cov_n,
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


def _make_prior_term(spec: dict[str, Any]) -> Callable[[NDArray[np.floating]], float]:
    """Builds a single prior term from a configuration dictionary.

    Expected format:
        {"name": "<prior_name>", "params": {...}, "bounds": optional_bounds}

    Args:
        spec: Prior specification dictionary.

    Returns:
        Callable log-prior term: logp(theta) -> float.

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

    if name == "uniform":
        # allow either params["bounds"] or spec["bounds"] (but not both)
        pb = params.get("bounds", None)
        if pb is not None and term_bounds is not None:
            raise ValueError(
                "uniform prior: provide bounds via either params['bounds'] or top-level 'bounds', not both")
        if pb is None and term_bounds is None:
            raise ValueError("uniform prior requires bounds")
        b = pb if pb is not None else term_bounds
        return prior_uniform(bounds=b)

    term = _PRIOR_REGISTRY[name](**params)
    return apply_hard_bounds(term, bounds=term_bounds)


def build_prior(
    *,
    terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Builds a logprior(theta)->float callable from a single unified spec.

    Args:
        terms: Sequence of prior term specifications (see below).
        bounds: Optional global hard bounds applied to the combined prior.

    The only user-facing API is:
      - `terms`: a list of prior terms, each either
          * ("prior_name", {params})
          * {"name": "prior_name", "params": {...}, "bounds": optional_term_bounds}
      - `bounds`: optional global hard bounds applied to the *combined* prior

    Conventions:
      - If `terms` is None or empty:
          * if `bounds` is None -> improper flat prior (prior_none)
          * else -> uniform top-hat prior over `bounds` (prior_uniform)
      - "uniform" priors must be passed as ("uniform", {"bounds": ...}) or the dict equivalent.
        Global `bounds` is still allowed (it just adds another hard gate).

    Examples:
      build_prior()
      build_prior(bounds=[(0.0, None), (None, None)])
      build_prior(terms=[("gaussian_diag", {"mean": mu, "sigma": sig})])
      build_prior(
          terms=[
              ("gaussian_diag", {"mean": mu, "sigma": sig}),
              ("log_uniform", {"index": 0}),
              {"name": "beta", "params": {"index": 2, "alpha": 2.0, "beta": 5.0}, "bounds": [(0,1), (None,None), (0,1)]},
          ],
          bounds=[(0.0, None), (None, None), (0.0, 1.0)],
      )
    """
    term_list = [] if terms is None else list(terms)

    # Empty -> either flat or bounded-uniform
    if len(term_list) == 0:
        base = prior_none() if bounds is None else prior_uniform(bounds=bounds)
        return base

    specs: list[dict[str, Any]] = []
    for t in term_list:
        if isinstance(t, dict):
            specs.append(t)
            continue

        if not isinstance(t, (tuple, list)) or len(t) != 2:
            raise TypeError(
                "Each term must be either a dict spec or a (name, params) tuple/list of length 2."
            )
        name, params = t
        if not isinstance(params, dict):
            raise TypeError("Term params must be a dict.")
        specs.append({"name": str(name), "params": dict(params)})

    # Use existing builder for dict-spec terms (supports per-term bounds, uniform special case)
    built_terms = [_make_prior_term(s) for s in specs]
    combined = sum_terms(*built_terms)

    # Global bounds apply last
    return apply_hard_bounds(combined, bounds=bounds)
