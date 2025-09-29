"""Differential calculus helpers."""

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function


def gradient(function, theta0, n_workers=1):
    """Returns the gradient of a scalar-valued function.

    Args:
        function: f(theta) -> scalar.
        theta0: 1D parameter point (array-like).
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        1D array of gradient values.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    def compute_deriv(i: int) -> float:
        """Compute partial derivative with respect to theta0[i]."""
        kit = DerivativeKit(get_partial_function(function, i, theta0), theta0[i])
        # NOTE: n_workers controls inner 1D differentiation
        return kit.adaptive.differentiate(order=1, n_workers=n_workers)

    grad = np.array([compute_deriv(i) for i in range(theta0.size)], dtype=float)
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in gradient.")
    return grad


def jacobian(*args, **kwargs):
    """This is a placeholder for a Jacobian computation function."""
    raise NotImplementedError
def hessian(*args, **kwargs):
    """This is a placeholder for a Hessian computation function."""
    raise NotImplementedError
def hessian_diag(*args, **kwargs):
    """This is a placeholder for a Hessian diagonal computation function."""
    raise NotImplementedError
def jacobian_diag(*args, **kwargs):
    """This is a placeholder for a Jacobian diagonal computation function."""
    raise NotImplementedError
def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError
