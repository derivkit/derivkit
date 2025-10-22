"""Squared-Exponential (RBF) kernel and derivative covariances (clean API)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "rbf_kernel",
    "rbf_cov_value_train_grad_test",
    "rbf_cov_grad_test_grad_test",
    "rbf_cov_value_train_hessdiag_test",
    "rbf_cov_hessdiag_test_samepoint",
]


def rbf_kernel(
    training_inputs: np.ndarray,
    other_inputs: np.ndarray,
    hyper_parameters: dict,
) -> np.ndarray:
    """Squared-Exponential / RBF kernel k(X, Y).

    Uses the isotropic SE form:
        k(x, y) = amp^2 * exp(- ||x - y||^2 / (2 * ell^2))

    Args:
      training_inputs: Array of input locations with shape (n, d).
      other_inputs:   Second array of input locations with shape (m, d).
      hyper_parameters: Hyperparameters dict with keys:
                        - "ell": length scale (float)
                        - "amp": signal amplitude (float)

    Returns:
      Kernel matrix with shape (n, m): k(training_inputs, other_inputs).
    """
    x = np.atleast_2d(training_inputs)
    y = np.atleast_2d(other_inputs)

    ell2 = float(hyper_parameters["ell"]) ** 2
    amp2 = float(hyper_parameters["amp"]) ** 2

    sqdist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return amp2 * np.exp(-0.5 * sqdist / ell2)


def rbf_cov_value_train_grad_test(
    training_inputs: np.ndarray,
    test_locations: np.ndarray,
    hyper_parameters: dict,
    axis: int = 0,
) -> np.ndarray:
    """Cross-covariance Cov[f(x_train), ∂f/∂x_axis(x_test)] for an RBF kernel.

    For the isotropic SE (RBF) kernel (component along `axis`):
        Cov = k(x, x*) * ( (x_axis - x*_axis) / ell^2 )

    Args:
      training_inputs: Training locations, shape (n_train, d).
      test_locations:  Test locations, shape (n_test, d).
      hyper_parameters: Hyperparameters dict with "ell" and "amp".
      axis: Coordinate index along which the derivative is taken
        at the test location. Defaults to 0.

    Returns:
      Cross-covariance matrix with shape (n_train, n_test).
      Element (i, j) = Cov[f(training_inputs[i]), ∂f/∂x_axis(test_locations[j])].
    """
    x = np.atleast_2d(training_inputs)
    x_star = np.atleast_2d(test_locations)
    kernel = rbf_kernel(x, x_star, hyper_parameters)
    ell2_axis = _ell2_for_axis(hyper_parameters, axis)
    delta = x[:, None, axis] - x_star[None, :, axis]
    cov = (delta / ell2_axis) * kernel
    return cov


def rbf_cov_grad_test_grad_test(
    test_locations1: np.ndarray,
    test_locations2: np.ndarray,
    hyper_parameters: dict,
    axis: int = 0,
) -> np.ndarray:
    """Covariance Cov[∂f/∂x_axis(x*_i), ∂f/∂x_axis(x*'_j)] for an RBF kernel.

    For the isotropic SE (RBF) kernel (component along `axis`):
        Cov = k(x*, x*') * ( 1/ell^2 - (Δ^2)/ell^4 ),  where Δ = x*_axis - x*'_axis

    Args:
      test_locations1: First set of test locations, shape (m1, d).
      test_locations2: Second set of test locations, shape (m2, d).
      hyper_parameters: Hyperparameters dict with "ell" and "amp".
      axis: Coordinate index along which the derivatives are taken.
        Defaults to 0.

    Returns:
      Covariance matrix with shape (m1, m2).
    """
    x1 = np.atleast_2d(test_locations1)
    x2 = np.atleast_2d(test_locations2)
    kernel = rbf_kernel(x1, x2, hyper_parameters)
    ell2_axis = _ell2_for_axis(hyper_parameters, axis)
    delta = x1[:, None, axis] - x2[None, :, axis]
    cov = (1.0 / ell2_axis - (delta ** 2) / (ell2_axis ** 2)) * kernel
    return cov


def rbf_cov_value_train_hessdiag_test(
    training_inputs: np.ndarray,
    test_locations: np.ndarray,
    hyper_parameters: dict,
    axis: int = 0,
) -> np.ndarray:
    """Cross-covariance Cov[f(x_train), ∂²f/∂x_axis²(x_test)] for an RBF kernel.

    For the isotropic SE (RBF) kernel (component along `axis`):
        Cov = k(x, x*) * ( (Δ^2)/ell^4 - 1/ell^2 ),  where Δ = x_axis - x*_axis

    Args:
      training_inputs: Training locations, shape (n_train, d).
      test_locations:  Test locations, shape (n_test, d).
      hyper_parameters: Hyperparameters dict with "ell" and "amp".
      axis: Coordinate index for the second derivative at the test
        location. Defaults to 0.

    Returns:
      Cross-covariance matrix with shape (n_train, n_test).
    """
    x = np.atleast_2d(training_inputs)
    x_star = np.atleast_2d(test_locations)
    kernel = rbf_kernel(x, x_star, hyper_parameters)
    ell2_axis = _ell2_for_axis(hyper_parameters, axis)
    delta = x[:, None, axis] - x_star[None, :, axis]
    cov = ((delta ** 2) / (ell2_axis ** 2) - 1.0 / ell2_axis) * kernel
    return cov


def rbf_cov_hessdiag_test_samepoint(
    test_locations: np.ndarray,
    hyper_parameters: dict,
    axis: int = 0,
) -> np.ndarray:
    """Same-point covariance of ∂²f/∂x_axis² at test inputs for an RBF kernel.

    For the isotropic SE (RBF) kernel in 1D along a single `axis`, the variance
    of the second derivative at a point is:
        Var[∂²f/∂x_axis²] = amp^2 * 3 / ell^4

    We return a diagonal matrix with that value for each test location.

    Args:
      test_locations: Test input locations, shape (m, d).
      hyper_parameters: Hyperparameters dict with "ell" and "amp".
      axis: Coordinate index of the second derivative (kept for API
        symmetry; the closed-form here does not depend on it in
        the isotropic case). Defaults to 0.

    Returns:
      Diagonal covariance matrix with shape (m, m).
    """
    x_star = np.atleast_2d(test_locations)
    ell2_axis = _ell2_for_axis(hyper_parameters, axis)
    amp2 = float(hyper_parameters["amp"]) ** 2
    value = amp2 * 3.0 / (ell2_axis ** 2)
    cov = np.eye(x_star.shape[0]) * value
    return cov


def _ell2_for_axis(hyper_params: dict, axis: int) -> float:
    """Return ℓ² for the chosen axis; supports scalar or per-dim ℓ.

    Args:
        hyper_params: Hyperparameter dict with "ell" and "amp".
        axis: Coordinate index of the second derivative

    Returns:
        Covariance matrix with shape (m, m).
    """
    ell = hyper_params["ell"]
    if np.ndim(ell) == 0:
        return float(ell) ** 2
    ell_vec = np.asarray(ell, dtype=float)
    cov = float(ell_vec[axis]) ** 2
    return cov
