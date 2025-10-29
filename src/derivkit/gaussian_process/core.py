"""Gaussian process core ops for RBF kernel (natural names, lowercase keys)."""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky, solve

from .kernels import (
    rbf_kernel,
    rbf_cov_value_train_grad_test,
    rbf_cov_grad_test_grad_test,
    rbf_cov_value_train_hessdiag_test,
    rbf_cov_hessdiag_test_samepoint,
)

from .gp_utils import neg_log_marginal_likelihood, standardize_targets

__all__ = [
    "gp_fit",
    "gp_predict",
    "gp_choose_hyperparams",
    "gp_derivative",
]


def gp_fit(training_inputs, targets, hyper_params, noise_variance, normalize=True):
    """Fit GP state; optionally center targets.

    Args:
      training_inputs: (n, d) array.
      targets: (n,) array.
      hyper_params: dict with keys {"ell", "amp"}.
      noise_variance: float, observation noise variance.
      normalize: if True, subtract mean(targets) and store the mean.

    Returns:
      dict with cached factors:
        - "training_inputs", "hyper_params", "noise_variance"
        - "chol_factor" (Cholesky of K_xx), "alpha" (=K^{-1}(y-mean))
        - "target_mean", "normalize"
    """
    x = np.atleast_2d(training_inputs)
    y = np.asarray(targets).reshape(-1)

    if normalize:
        y_centered, y_mean = standardize_targets(y)
    else:
        y_centered, y_mean = y, 0.0

    k_xx = rbf_kernel(x, x, hyper_params) + float(noise_variance) * np.eye(x.shape[0])
    l_factor = cholesky(k_xx)
    alpha = solve(l_factor.T, solve(l_factor, y_centered))

    return {
        "training_inputs": x,
        "hyper_params": dict(hyper_params),
        "noise_variance": float(noise_variance),
        "chol_factor": l_factor,
        "alpha": alpha,
        "target_mean": y_mean,
        "normalize": bool(normalize),
    }


def gp_predict(state, test_locations):
    """Function mean/variance at test locations (not derivatives).

    Args:
      state: dict returned by gp_fit.
      test_locations: (m, d) array.

    Returns:
      (mean, var) with shape (m,), (m,).
    """
    x = state["training_inputs"]
    hp = state["hyper_params"]
    l_factor = state["chol_factor"]
    alpha = state["alpha"]

    x_star = np.atleast_2d(test_locations)
    k_xxstar = rbf_kernel(x, x_star, hp)
    mean = k_xxstar.T @ alpha

    v = solve(l_factor, k_xxstar)
    k_xstarxstar = rbf_kernel(x_star, x_star, hp)
    var = np.clip(np.diag(k_xstarxstar - v.T @ v), 0.0, np.inf)

    if state.get("normalize", False):
        mean = mean + state.get("target_mean", 0.0)

    return mean.squeeze(), var


def gp_choose_hyperparams(training_inputs, targets, init_hp=None, init_noise=1e-6, normalize=True):
    """Pick hyperparams by NLML (L-BFGS-B if available, else tiny grid).

    Returns:
      (hyper_params_dict, noise_variance_float)
    """
    x = np.atleast_2d(training_inputs)
    y = np.asarray(targets).reshape(-1)

    span = max(1e-9, float(x[:, 0].max() - x[:, 0].min()))
    ell0 = init_hp.get("ell", span / 2.0) if init_hp else span / 2.0
    amp0 = init_hp.get("amp", float(np.std(y) or 1.0)) if init_hp else float(np.std(y) or 1.0)
    noise0 = float(init_noise)

    try:
        from scipy.optimize import minimize  # type: ignore

        theta0 = np.log([max(ell0, 1e-8), max(amp0, 1e-12), max(noise0, 1e-12)])
        bounds = [
            (np.log(1e-6), np.log(10.0 * span + 1e-6)),
            (np.log(1e-6), np.log(1e3 * max(amp0, 1e-3))),
            (np.log(1e-12), np.log(1e-1)),
        ]
        res = minimize(
            lambda th: neg_log_marginal_likelihood(th, x, y, normalize),
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not res.success:
            raise RuntimeError("lbfgs failed")
        ell, amp, noise = np.exp(res.x)
        return {"ell": float(ell), "amp": float(amp)}, float(noise)

    except Exception:
        ells = np.geomspace(max(span / 8.0, 1e-4), max(span * 2.0, 1e-3), 10)
        noises = np.geomspace(1e-10, 1e-4, 6)
        best = (np.inf, ell0, amp0, noise0)
        for ell in ells:
            for noise in noises:
                nll = neg_log_marginal_likelihood(np.log([ell, amp0, noise]), x, y, normalize)
                if nll < best[0]:
                    best = (nll, ell, amp0, noise)
        _, ell, amp, noise = best
        return {"ell": float(ell), "amp": float(amp)}, float(noise)


def gp_derivative(state, test_locations, order=1, axis=0):
    """Derivative mean/variance at test points for order=1 or 2 along `axis`."""
    x = state["training_inputs"]
    hp = state["hyper_params"]
    l_factor = state["chol_factor"]
    alpha = state["alpha"]

    x_star = np.atleast_2d(test_locations)

    if order == 1:
        k_cross = rbf_cov_value_train_grad_test(x, x_star, hp, axis=axis)     # (n, m)
        mean = k_cross.T @ alpha                                              # (m,)
        v = solve(l_factor, k_cross)
        k_dd = rbf_cov_grad_test_grad_test(x_star, x_star, hp, axis=axis)     # (m, m)
        k_dd = k_dd + 1e-12 * np.eye(k_dd.shape[0])
        var = np.clip(np.diag(k_dd - v.T @ v), 0.0, np.inf)

    elif order == 2:
        k_cross = rbf_cov_value_train_hessdiag_test(x, x_star, hp, axis=axis) # (n, m)
        mean = k_cross.T @ alpha
        v = solve(l_factor, k_cross)
        k_dd = rbf_cov_hessdiag_test_samepoint(x_star, hp, axis=axis)         # (m, m) diagonal
        k_dd = k_dd + 1e-12 * np.eye(k_dd.shape[0])
        var = np.clip(np.diag(k_dd - v.T @ v), 0.0, np.inf)

    else:
        raise NotImplementedError("only order=1 or 2 supported")

    return mean.squeeze(), var
