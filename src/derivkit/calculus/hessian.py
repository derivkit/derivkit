"""Contains functions used in constructing the Hessian of a scalar-valued function."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.sandbox import get_partial_function


def build_hessian(function: Callable,
                  theta0: np.ndarray,
                  n_workers: int=1
) -> NDArray[np.floating]:
    """Returns the hessian of a scalar-valued function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the hessian is evaluated.
        n_workers: Number of workers used by DerivativeKit.adaptive.differentiate.
            This setting does not parallelize across parameters.

    Returns:
        A 2D array representing the hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a scalar value.
    """
    theta = np.asarray(theta0, dtype=float).ravel()
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    f0 = np.asarray(function(theta), dtype=float)
    if f0.size != 1:
        raise TypeError("build_hessian() expects a scalar-valued function.")

    n_parameters = theta.size
    hess = np.empty((n_parameters, n_parameters), dtype=float)

    # Diagonals here (pure second orders)
    for i in range(n_parameters):
        hess[i, i] = _hessian_component(function, theta, i, i, n_workers)

    # Off-diagonals here (mixed second orders).
    for i in range(n_parameters):
        for j in range(i + 1, n_parameters):
            hij = _hessian_component(function, theta, i, j, n_workers)
            hess[i, j] = hij
            hess[j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in hessian.")
    return hess


def _hessian_component(function: Callable, theta0: np.ndarray, i: int, j:int, n_workers: int) -> float:
    """Return one entry of the Hessian for a scalar-valued function.

    Used inside ``build_hessian`` to measure how the function’s change in one
    parameter depends on changes in another. This can describe both pure
    second derivatives and mixed ones.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        n_workers: Number of workers used for the internal derivative step.

    Returns:
        A single number showing how the rate of change in one parameter
        depends on another.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    if i == j:
        # 1 parameter to differentiate twice, and n_parameters-1 parameters to hold fixed
        partial_vec1 = get_partial_function(
            function, i, theta0
        )

        # One-time scalar check for build_hessian()
        probe = np.asarray(partial_vec1(theta0[i]), dtype=float)
        if probe.size != 1:
            raise TypeError(
                "build_hessian() expects a scalar-valued function; "
                f"got shape {probe.shape} from full_function(params)."
            )

        kit1 = DerivativeKit(
            partial_vec1, theta0[i]
        )
        return kit1.adaptive.differentiate(
                order=2, n_workers=n_workers
            )

    else:
        # 2 parameters to differentiate once, with other parameters held fixed
        def partial_vec2(y):
            theta0_y = theta0.copy()
            theta0_y[j] = y
            partial_vec1 = get_partial_function(
                function, i, theta0_y
            )
            kit1 = DerivativeKit(
                partial_vec1, theta0[i]
            )
            return kit1.adaptive.differentiate(order=1, n_workers=n_workers)

        kit2 = DerivativeKit(
            partial_vec2, theta0[j]
        )
        return kit2.adaptive.differentiate(
                order=1, n_workers=n_workers
            )


def hessian_diag(*args, **kwargs):
    """This is a placeholder for a Hessian diagonal computation function."""
    raise NotImplementedError


def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError
