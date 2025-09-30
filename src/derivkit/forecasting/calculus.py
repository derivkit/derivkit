"""Differential calculus helpers."""

from collections.abc import Callable

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function


def gradient(function, theta0, n_workers=1):
    """Returns the gradient of a scalar-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the gradient is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 1D array representing the gradient.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # n_workers controls inner 1D differentiation (not across parameters).
    grad = np.array(
        [_grad_component(function, theta0, i, n_workers) for i in range(theta0.size)],
        dtype=float,
    )
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in gradient.")
    return grad

def _grad_component(function, theta0: np.ndarray, i: int, n_workers: int) -> float:
    """Compute one component of the gradient of a scalar-valued function.

    Helper used by ``gradient``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

    Args:
        function (callable): The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return a scalar observable value.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Returns:
        float: The ith component of the gradient of function evaluated at ``theta0``.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)

    # One-time scalar check for gradient()
    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )

    kit = DerivativeKit(partial_vec, theta0[i])
    return kit.adaptive.differentiate(order=1, n_workers=n_workers)

def jacobian(function, theta0, n_workers=1):
    """Returns the jacobian of a vector-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the jacobian is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 2D array representing the jacobian.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # n_workers controls inner 1D differentiation (not across parameters).
    cols = [_jacobian_component(function, theta0, i, n_workers) for i in range(theta0.size)]
    jac = np.stack(cols, axis=1)  # (m_outputs, n_params)
    if not np.isfinite(jac).all():
        raise FloatingPointError("Non-finite values encountered in jacobian.")
    return jac

def _jacobian_component(function: Callable, theta0: np.ndarray, i: int, n_workers: int) -> np.ndarray:
    """Compute one column of the jacobian of a vector-valued function.

    Helper used by ``jacobian``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

    Args:
        function: The vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return an array of observable values.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Returns:
        The ith column of the jacobian of function evaluated at ``theta0``.
    """
    partial_vec = get_partial_function(function, i, theta0)

    kit = DerivativeKit(partial_vec, theta0[i])
    return np.atleast_1d(np.asarray(kit.adaptive.differentiate(order=1, n_workers=n_workers), dtype=float))

def hessian(function, theta0, n_workers=1):
    """Returns the hessian of a scalar-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 2D array representing the hessian.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    f0 = np.asarray(function(theta0), dtype=float)
    if f0.size != 1:
        raise TypeError("hessian() expects a scalar-valued function.")

    n_parameters = theta0.size
    hess = np.empty((n_parameters, n_parameters), dtype=float)

    # Diagonals here (pure second orders)
    for i in range(n_parameters):
        hess[i, i] = _hessian_component(function, theta0, i, i, n_workers)

    # Off-diagonals here (mixed second orders).
    for i in range(n_parameters):
        for j in range(i + 1, n_parameters):
            hij = _hessian_component(function, theta0, i, j, n_workers)
            hess[i, j] = hij
            hess[j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in hessian.")
    return hess

def _hessian_component(function: Callable, theta0: np.ndarray, i: int, j:int, n_workers: int) -> float:
    """Compute one component of the hessian of a scalar-valued function.

    Helper used by ``hessian``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

    Args:
        function: The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return an array of observable values.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the first parameter with respect to which to differentiate.
        j: Zero-based index of the second parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Returns:
        The ith, jth component of the hessian of function evaluated at ``theta0``.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    if i == j:
        # 1 parameter to differentiate twice, and n_parameters-1 parameters to hold fixed
        partial_vec1 = get_partial_function(
            function, i, theta0
        )

        # One-time scalar check for hessian()
        probe = np.asarray(partial_vec1(theta0[i]), dtype=float)
        if probe.size != 1:
            raise TypeError(
                "hessian() expects a scalar-valued function; "
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
def jacobian_diag(*args, **kwargs):
    """This is a placeholder for a Jacobian diagonal computation function."""
    raise NotImplementedError
def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError
