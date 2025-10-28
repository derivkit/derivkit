"""Contains functions used to construct the gradient of scalar-valued functions."""

from collections.abc import Callable

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.sandbox import get_partial_function
from derivkit.utils.validate import check_scalar_valued


def build_gradient(function, theta0, n_workers=1):
    """Returns the gradient of a scalar-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the gradient is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 1D array representing the gradient.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # One-time scalar check for build_gradient()
    check_scalar_valued(function, theta0, 0, n_workers)

    # n_workers controls inner 1D differentiation (not across parameters).
    grad = np.array(
        [
            _grad_component(function, theta0, i, n_workers)
            for i in range(theta0.size)
        ],
        dtype=float,
    )
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in build_gradient.")
    return grad


def _grad_component(
        function: Callable,
        theta0: np.ndarray, i:int,
        n_workers: int
) -> float:
    """Returns one entry of the gradient for a scalar-valued function.

    Used inside ``build_gradient`` to find how the function changes with respect
    to a single parameter while keeping the others fixed.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: The index of the parameter being varied.
        n_workers: Number of workers used for the internal derivative step.

    Returns:
        A single number showing how the function changes with that parameter.
    """
    partial_vec = get_partial_function(function, i, theta0)

    kit = DerivativeKit(partial_vec, theta0[i])
    return kit.adaptive.differentiate(order=1, n_workers=n_workers)
