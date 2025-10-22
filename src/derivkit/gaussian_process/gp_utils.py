
import numpy as np
from numpy.linalg import cholesky, solve

from .kernels import (
    rbf_kernel,
)

__all__ = [
    "neg_log_marginal_likelihood",
    "standardize_targets"
]


def neg_log_marginal_likelihood(theta, training_inputs, targets, normalize=True):
    """Negative log marginal likelihood at (log_ell, log_amp, log_noise)."""
    ell = float(np.exp(theta[0]))
    amp = float(np.exp(theta[1]))
    noise = float(np.exp(theta[2]))
    hp = {"ell": ell, "amp": amp}

    x = np.atleast_2d(training_inputs)
    y = np.asarray(targets).reshape(-1)
    y_use = y - np.mean(y) if normalize else y

    k_xx = rbf_kernel(x, x, hp) + noise * np.eye(x.shape[0])
    try:
        l_factor = cholesky(k_xx)
    except Exception:
        return np.inf

    alpha = solve(l_factor.T, solve(l_factor, y_use))
    logdet = 2.0 * np.sum(np.log(np.diag(l_factor)))
    n = y_use.size
    return 0.5 * y_use @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)


def standardize_targets(targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Center targets by their mean."""
    y = np.asarray(targets).reshape(-1)
    mu = float(np.mean(y))
    return y - mu, mu
