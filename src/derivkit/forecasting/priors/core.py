"""Prior utilities (core priors + unified builder).

Priors are represented as callables:

    ``logprior(theta) -> float``

The return value is interpreted as a log-density defined up to an additive
constant. Returning ``-np.inf`` denotes zero probability (hard exclusion).

These priors are designed to be used when constructing log-posteriors for
sampling (e.g., from Fisher/DALI Gaussian approximations) or when evaluating
posterior surfaces. GetDist plots what it is given; priors must be applied when
generating samples or, for Gaussian approximations, incorporated explicitly into
the Fisher matrix.
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


def _prior_none_impl(
    theta: NDArray[np.floating]
) -> float:
    """Returns a constant log-prior (improper flat prior).

    This prior is constant in parameter space (up to an additive constant), so it
    contributes zero to the log-posterior at every point.

    Args:
        theta: Parameter vector (unused).

    Returns:
        Log-prior value (always ``0.0``).
    """
    _ = theta
    return 0.0


def _prior_1d_impl(
    theta: NDArray[np.floating],
    *,
    index: int,
    domain: str,
    kind: str,
    a: float | np.floating = 0.0,
    b: float | np.floating = 1.0,
) -> float:
    """A generic 1D prior implementation.

    Args:
        theta: Parameter vector.
        index: Index of the parameter to which the prior applies.
        domain: Domain restriction (``"positive"``, ``"nonnegative"``, ``"unit_open"``).
        kind: Prior kind (``"log_uniform"``, ``"half_normal"``, ``"half_cauchy"``, ``"log_normal"``, ``"beta"``).
        a: First prior parameter (meaning depends on ``kind``).
        b: Second prior parameter (meaning depends on ``kind``).

    Returns:
        Log-prior value for the specified parameter.

    Raises:
        ValueError: If ``domain`` is unknown, if ``kind`` is unknown, or if the
            distribution parameters are invalid (e.g., non-positive scale/sigma or
            non-positive ``a/b``).
    """
    x = get_index_value(theta, index, name="theta")

    match domain:
        case "positive":
            if x <= 0.0:
                return -np.inf
        case "nonnegative":
            if x < 0.0:
                return -np.inf
        case "unit_open":
            if x <= 0.0 or x >= 1.0:
                return -np.inf
        case _:
            raise ValueError(f"unknown domain '{domain}'")

    logp: float
    match kind:
        case "log_uniform":
            logp = float(-np.log(x))

        case "half_normal":
            sigma = float(a)
            if sigma <= 0.0:
                raise ValueError("sigma must be > 0")
            logp = float(-0.5 * (x / sigma) ** 2)

        case "half_cauchy":
            scale = float(a)
            if scale <= 0.0:
                raise ValueError("scale must be > 0")
            t = x / scale
            logp = float(-np.log1p(t * t))

        case "log_normal":
            mean = float(a)
            sigma = float(b)
            if sigma <= 0.0:
                raise ValueError("sigma must be > 0")
            lx = np.log(x)
            z = (lx - mean) / sigma
            logp = float(-0.5 * z * z - lx)

        case "beta":
            alpha = float(a)
            beta = float(b)
            if alpha <= 0.0 or beta <= 0.0:
                raise ValueError("alpha and beta must be > 0")
            logp = float((alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x))

        case _:
            raise ValueError(f"unknown prior kind '{kind}'")

    return logp


def _prior_gaussian_impl(
    theta: NDArray[np.floating],
    *,
    mean: NDArray[np.floating],
    inv_cov: NDArray[np.floating],
) -> float:
    """Evaluates a multivariate Gaussian log-prior (up to an additive constant).

    Args:
        theta: Parameter vector.
        mean: Mean vector.
        inv_cov: Inverse covariance matrix.

    Returns:
        Log-prior value.

    Raises:
        ValueError: If ``theta``/``mean`` are not 1D or if ``inv_cov`` does not have
            shape ``(p, p)`` consistent with ``mean``.
    """
    thetas = np.asarray(theta, dtype=np.float64)
    means = np.asarray(mean, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)

    if means.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {means.shape}")
    if inv_cov.ndim != 2 or inv_cov.shape[0] != inv_cov.shape[1] or inv_cov.shape[0] != means.size:
        raise ValueError(f"inv_cov must have shape (p, p) with p={means.size}, got {inv_cov.shape}")
    if thetas.ndim != 1 or thetas.size != means.size:
        raise ValueError(f"theta must have shape ({means.size},), got {thetas.shape}")

    diff = thetas - means
    return float(-0.5 * (diff @ inv_cov @ diff))


def _prior_gaussian_diag_impl(
    theta: NDArray[np.floating],
    *,
    mean: NDArray[np.floating],
    inv_cov: NDArray[np.floating],
) -> float:
    """Evaluates a diagonal multivariate Gaussian log-prior (up to an additive constant).

    Args:
        theta: Parameter vector.
        mean: Mean vector.
        inv_cov: Inverse covariance matrix (must be diagonal).

    Returns:
        Log-prior value.

    Raises:
        ValueError: If shapes are inconsistent, if ``inv_cov`` is not diagonal, or if
            any diagonal entry of ``inv_cov`` is non-positive.
    """
    thetas = np.asarray(theta, dtype=np.float64)
    means = np.asarray(mean, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)

    if means.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {means.shape}")
    p = means.size
    if inv_cov.ndim != 2 or inv_cov.shape != (p, p):
        raise ValueError(f"inv_cov must have shape ({p},{p}), got {inv_cov.shape}")
    if thetas.ndim != 1 or thetas.size != p:
        raise ValueError(f"theta must have shape ({p},), got {thetas.shape}")

    inv_var = np.diag(inv_cov)
    off_diag = inv_cov.copy()
    np.fill_diagonal(off_diag, 0.0)
    if not np.allclose(off_diag, 0.0, rtol=1e-12, atol=1e-12):
        raise ValueError("inv_cov must be diagonal for prior_gaussian_diag")
    if np.any(inv_var <= 0.0):
        raise ValueError("diagonal inv_cov entries must be > 0")

    diff = thetas - means
    return float(-0.5 * np.sum(diff * diff * inv_var))


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

        ``p(theta) = sum_n w_n * N(theta | means_n, cov_n)``
    where ``N(theta | mean, cov)`` is the multivariate Gaussian density with the
    specified mean and covariance; ``w_n` are the mixture weights (in log-space); and
    the sum runs over the ``n=1..N`` components.

    The result is a log-density defined up to an additive constant.

    Notes:
        - ``inv_covs`` are the per-component inverse covariances (precision matrices).
        - ``log_weights`` are the mixture weights in log-space; they are typically
          normalized so that ``logsumexp_1d(log_weights) = 0``.
        - ``log_component_norm`` controls whether per-component normalization is
          included. If it contains ``-0.5 * log(|C_n|)`` (or the equivalent computed
          from ``C_n^{-1}``), then components with different covariances get the
          correct relative normalization. If it is all zeros, the mixture keeps
          only the quadratic terms, which can be useful for shape-only priors.

    Args:
        theta: Parameter vector ``theta`` with shape ``(p,)``.
        means: Component means with shape ``(n, p)``.
        inv_covs: Component inverse covariances with shape ``(n, p, p)``.
        log_weights: Log-weights for the ``n`` components with shape ``(n,)``.
        log_component_norm: Per-component log-normalization terms with shape
            ``(n,)`` (often ``-0.5 * log(|C_n|)``). Use zeros to omit this factor.

    Returns:
        The mixture log-prior value at ``theta`` (a finite float or ``-np.inf`` if
        the caller enforces hard bounds elsewhere).

    Raises:
        ValueError: If input arrays have incompatible shapes or dimensions.
    """
    thetas = np.asarray(theta, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    inv_covs = np.asarray(inv_covs, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_comp_norm = np.asarray(log_component_norm, dtype=np.float64)

    if thetas.ndim != 1:
        raise ValueError(f"theta must be 1D, got shape {thetas.shape}")

    if means.ndim != 2:
        raise ValueError(f"means must be (n, p), got shape {means.shape}")

    n, p = means.shape
    if thetas.size != p:
        raise ValueError(f"theta length {thetas.size} != p {p}")

    if inv_covs.ndim != 3 or inv_covs.shape != (n, p, p):
        raise ValueError(f"inv_covs must be (n, p, p) with n={n}, p={p}, got shape {inv_covs.shape}")

    if log_weights.ndim != 1 or log_weights.size != n:
        raise ValueError(f"log_weights must be (n,), got shape {log_weights.shape}")

    if log_comp_norm.ndim != 1 or log_comp_norm.size != n:
        raise ValueError(f"log_component_norm must be (n,), got shape {log_comp_norm.shape}")

    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff = thetas - means[i]
        vals[i] = log_weights[i] + log_comp_norm[i] - 0.5 * float(diff @ inv_covs[i] @ diff)

    return float(logsumexp_1d(vals))


def prior_none() -> Callable[[NDArray[np.floating]], float]:
    """Constructs an improper flat prior (constant log-density).

    This prior has density proportional to 1 everywhere.

    Returns:
        Log-prior value (always ``0.0``).
    """
    return _prior_none_impl


def prior_uniform(
    *,
    bounds: Sequence[tuple[float | np.floating | None, float | np.floating | None]],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a uniform prior with hard bounds.

    This prior has density proportional to 1 within the specified bounds
    and zero outside. The log-density is constant (up to an additive constant)
    within the bounds and ``-np.inf`` outside.

    Args:
        bounds: Sequence of (min, max) pairs for each parameter.
            Use None for unbounded sides.

    Returns:
        Callable log-prior: ``logp(theta) -> float``
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
    ``exp(-0.5 * (theta - mean)^T @ inv_cov @ (theta - mean))`` with ``inv_cov`` being the
    inverse of the provided covariance matrix and ``theta`` being the parameter vector.

    Args:
        mean: Mean vector.
        cov: Covariance matrix (provide exactly one of ``cov`` or ``inv_cov``).
        inv_cov: Inverse covariance matrix (provide exactly one of ``cov`` or ``inv_cov``).

    Returns:
        Callable log-prior: logp(theta) -> float.

    Raises:
        ValueError: If neither or both of ``cov`` and ``inv_cov`` are provided,
            or if the provided covariance/inverse covariance cannot be
            normalized/validated.
    """
    if (cov is None) == (inv_cov is None):
        raise ValueError("Provide exactly one of `cov` or `inv_cov`.")

    means = np.asarray(mean, dtype=np.float64)
    if means.ndim != 1:
        raise ValueError(f"mean must be 1D, got shape {means.shape}")

    if inv_cov is None:
        cov = normalize_covariance(cov, n_parameters=means.size)
        inv_cov = invert_covariance(cov, warn_prefix="prior_gaussian")
    else:
        inv_cov = np.asarray(inv_cov, dtype=np.float64)
        if inv_cov.ndim != 2 or inv_cov.shape != (means.size, means.size):
            raise ValueError(f"inv_cov must have shape (p,p) with p={means.size}, got {inv_cov.shape}")
        if not np.all(np.isfinite(inv_cov)):
            raise ValueError("inv_cov contains non-finite values.")

    return partial(_prior_gaussian_impl, mean=means, inv_cov=inv_cov)


def prior_gaussian_diag(
    *,
    mean: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a diagonal Gaussian prior (up to an additive constant).

    This prior has density proportional to
    ``exp(-0.5 * sum_i ((x_i - mean_i) / sigma_i)^2)``, with independent components.

    Args:
        mean: Mean vector.
        sigma: Standard deviation vector (must be positive).

    Returns:
        Callable log-prior: logp(theta) -> float.

    Raises:
        ValueError: If `mean` and `sigma` have incompatible shapes,
            or if any `sigma` entries are non-positive.
    """
    means = np.asarray(mean, dtype=np.float64)
    sigmas = np.asarray(sigma, dtype=np.float64)

    if means.ndim != 1 or sigmas.ndim != 1 or means.shape != sigmas.shape:
        raise ValueError("mean and sigma must be 1D arrays with the same shape")
    if np.any(sigmas <= 0.0):
        raise ValueError("all sigma entries must be > 0")

    inv_cov = np.diag(1.0 / (sigmas ** 2))
    return partial(_prior_gaussian_diag_impl, mean=means, inv_cov=inv_cov)


def prior_log_uniform(
    *,
    index: int,
) -> Callable[[NDArray[np.floating]], float]:
    """Construct a Jeffreys prior for a single positive scale parameter.

    The Jeffreys prior is defined via the Fisher information of the parameter.
    For a positive scale parameter, this results in a prior proportional to
    ``1 / x``, which is the same functional form as a log-uniform prior.

    Args:
        index: Index of the parameter to which the prior applies.

    Returns:
        A callable that evaluates the log-prior at a given parameter vector.

    Note:
        Although this implementation is identical to ``prior_log_uniform``,
        the name ``prior_jeffreys`` emphasizes the statistical motivation
        (reparameterization invariance for scale parameters) rather than
        uniformity in log-space.
    """
    return partial(_prior_1d_impl, index=int(index), domain="positive", kind="log_uniform")


def prior_jeffreys(
        *,
        index: int,
) -> Callable[[NDArray[np.floating]], float]:
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


def prior_half_normal(
        *,
        index: int,
        sigma: float | np.floating,
) -> Callable[[NDArray[np.floating]], float]:
    """"Constructs a half-normal prior for a single non-negative parameter.

    This prior is a standard weakly informative choice for non-negative amplitude
    or scale parameters. It is conceptually different from a normal prior with a
    non-negativity bound, which corresponds to a truncated normal. The half-normal
    instead arises as the distribution of ``|N(0, sigma)|``.

    The (unnormalized) density is proportional to ``exp(-0.5 * (x / sigma)**2)``
    for ``x >= 0``.

    Args:
        index: Index of the parameter to which the prior applies.
        sigma: Standard deviation of the underlying normal distribution.

    Returns:
        A callable that evaluates the log-prior at a given parameter vector.

    Raises:
        ValueError: If `sigma` is not positive.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_normal", a=sigma)


def prior_half_cauchy(
        *,
        index: int,
        scale: float | np.floating
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a half-Cauchy prior for a single non-negative parameter.

    This prior has density proportional to ``1 / (1 + (x/scale)^2)`` for ``x >= 0``,
    with x being the parameter.

    Args:
        index: Index of the parameter to which the prior applies.
        scale: Scale parameter of the half-Cauchy distribution.

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `scale` is not positive.
    """
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be > 0")
    return partial(_prior_1d_impl, index=int(index), domain="nonnegative", kind="half_cauchy", a=scale)


def prior_log_normal(
    *,
    index: int,
    mean_log: float | np.floating,
    sigma_log: float | np.floating
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a log-normal prior for a single positive parameter.

    This prior has density proportional to
    ``exp(-0.5 * ((log(x) - mean_log) / sigma_log) ** 2) / x`` for ``x > 0``.

    Args:
        index: Index of the parameter to which the prior applies.
        mean_log: Mean of the underlying normal distribution in log-space.
        sigma_log: Standard deviation of the underlying normal distribution in log-space.

    Returns:
        Callable log-prior: logp(theta) -> float

    Raises:
        ValueError: If `sigma_log` is not positive.
    """
    sig_log = float(sigma_log)
    if sig_log <= 0.0:
        raise ValueError("sigma_log must be > 0")
    return partial(
        _prior_1d_impl,
        index=int(index),
        domain="positive",
        kind="log_normal",
        a=float(mean_log),
        b=sig_log)


def prior_beta(
    *,
    index: int,
    alpha: float | np.floating,
    beta: float | np.floating,
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs a Beta distribution prior for a single parameter in ``(0, 1)``.

    This prior uses the Beta density on ``x in (0, 1)``, with shape parameters
    ``alpha > 0`` and ``beta > 0``. The returned callable evaluates the
    corresponding log-density up to an additive constant (the normalization
    constant does not depend on ``x`` and is therefore omitted).

    The (unnormalized) density is proportional to::

        x**(alpha - 1) * (1 - x)**(beta - 1)

    Args:
        index: Index of the parameter to which the prior applies.
        alpha: Alpha shape parameter (must be greater than 0).
        beta: Beta shape parameter (must be greater than 0).

    Returns:
        A callable that evaluates the log-prior at a given parameter vector.

    Raises:
        ValueError: If ``alpha`` or ``beta`` are not positive.
    """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be > 0")
    return partial(
        _prior_1d_impl,
        index=int(index),
        domain="unit_open",
        kind="beta",
        a=alpha,
        b=beta)


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
        ``p(theta) = sum_n w_n * N(theta | mean_n, cov_n)``
    where ``N(theta | mean, cov)`` is the multivariate Gaussian density with the
    specified mean and covariance; ``w_n`` are the mixture weights (non-negative,
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

    Raises:
        ValueError: If inputs have incompatible shapes, if both/neither of
            ``covs``/``inv_covs`` are provided, if both/neither of ``weights``/``log_weights``
            are provided, if weights are invalid, or if covariance inputs are not
            compatible with ``include_component_norm=True``.
    """
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError(f"means must be (n, p), got shape {means.shape}")
    n, p = means.shape

    include = bool(include_component_norm)

    if (covs is None) == (inv_covs is None):
        raise ValueError("Provide exactly one of `covs` or `inv_covs`.")

    if inv_covs is None:
        cov = np.asarray(covs, dtype=np.float64)
        if cov.ndim != 3 or cov.shape != (n, p, p):
            raise ValueError(f"covs must be (n, p, p) with n={n}, p={p}, got shape {cov.shape}")

        if include:
            log_component_norm = np.empty(n, dtype=np.float64)
            for i in range(n):
                sign, logdet = np.linalg.slogdet(cov[i])
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError(
                        "include_component_norm=True requires each cov_n to be positive-definite "
                        "(slogdet sign>0 and finite)."
                    )
                log_component_norm[i] = -0.5 * logdet
        else:
            log_component_norm = np.zeros(n, dtype=np.float64)

        inv_cov_n = np.empty_like(cov)
        for i in range(n):
            inv_cov_n[i] = invert_covariance(cov[i], warn_prefix="prior_gaussian_mixture")

    else:
        inv_cov_n = np.asarray(inv_covs, dtype=np.float64)
        if inv_cov_n.ndim != 3 or inv_cov_n.shape != (n, p, p):
            raise ValueError(f"inv_covs must be (n, p, p) with n={n}, p={p}, got shape {inv_cov_n.shape}")

        if include:
            log_component_norm = np.empty(n, dtype=np.float64)
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
            log_component_norm = np.zeros(n, dtype=np.float64)

    if (weights is None) == (log_weights is None):
        raise ValueError("Provide exactly one of `weights` or `log_weights`.")

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != n:
            raise ValueError(f"weights must be (n,) with n={n}, got shape {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w = w / s
        lw = np.full_like(w, -np.inf, dtype=np.float64)
        np.log(w, out=lw, where=(w > 0))
    else:
        lw_in = np.asarray(log_weights, dtype=np.float64)
        if lw_in.ndim != 1 or lw_in.size != n:
            raise ValueError(f"log_weights must be (n,) with n={n}, got shape {lw_in.shape}")
        lw = lw_in - logsumexp_1d(lw_in)

    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(inv_cov_n))):
        raise ValueError("mixture prior inputs must be finite")

    if np.any(np.isnan(lw)) or np.any(lw == np.inf):
        raise ValueError("log_weights must not contain nan or +inf")

    return partial(
        _prior_gaussian_mixture_impl,
        means=means,
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
    """Builds one component of a composite prior from a configuration dictionary.

    A "prior term" here means a single log-prior contribution (for example, a
    Gaussian prior on a subset of parameters) that can be summed with other terms
    to form a complete prior.

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

    Raises:
        ValueError: If the spec is missing a valid prior name, if the name is not
            registered, or if a uniform prior does not include bounds (or includes
            bounds twice).
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
    bounds: Sequence[tuple[float | np.floating | None, float | np.floating | None]] | None = None,
) -> Callable[[NDArray[np.floating]], float]:
    """Constructs and returns a single log-prior callable from a unified prior specification.

    This function takes a list of prior terms (each specified either as a
    ``(name, params)`` tuple or as a dict with keys like ``name``, ``params``,
    and optional per-term ``bounds``), builds each term using the internal
    registry, sums the resulting log-priors, and then applies optional global
    hard bounds.

    If no terms are provided, the behavior falls back to a simple default:
    it returns an improper flat prior when ``bounds`` is ``None``, and a uniform
    top-hat prior over ``bounds`` otherwise.

    Args:
        terms: Sequence of prior term specifications (see below).
        bounds: Optional global hard bounds applied to the combined prior.

    The user-facing API is:
        - ``terms``: a list of prior terms, each either
            * ``("prior_name", {"param": value, ...})``
            * ``{"name": "prior_name", "params": {...}, "bounds": optional_term_bounds}``
        - ``bounds``: optional global hard bounds applied to the combined prior.

    Conventions:
        - If ``terms`` is ``None`` or empty:
            * if ``bounds`` is ``None`` -> improper flat prior (``prior_none``)
            * else -> uniform top-hat prior over ``bounds`` (``prior_uniform``)
        - ``"uniform"`` priors must be passed as ``("uniform", {"bounds": ...})`` or the dict
          equivalent. Global ``bounds`` is still allowed (it adds an additional hard gate).

    Examples:
        ``build_prior()``
        ``build_prior(bounds=[(0.0, None), (None, None)])``
        ``build_prior(terms=[("gaussian_diag", {"mean": mu, "sigma": sig})])``
        ``build_prior(
            terms=[
                ("gaussian_diag", {"mean": mu, "sigma": sig}),
                ("log_uniform", {"index": 0}),
                {"name": "beta", "params": {"index": 2, "alpha": 2.0, "beta": 5.0},
                 "bounds": [(0, 1), (None, None), (0, 1)]},
            ],
            bounds=[(0.0, None), (None, None), (0.0, 1.0)],
        )``

    Raises:
        TypeError: If a term is not a dict spec or a (name, params) tuple/list, or if
            term params are not a dict.
        ValueError: If a dict spec is invalid (see ``_make_prior_term``).
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
