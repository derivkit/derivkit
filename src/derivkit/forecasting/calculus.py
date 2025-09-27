import warnings
from copy import deepcopy

import numpy as np

from derivkit.derivative_kit import DerivativeKit

"""Differential calculus helpers."""

def gradient(function, theta0, n_workers=1):
    """Returns the gradient of a function.

    Args:
        function (callable): The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return a scalar observable value.
        theta0 (class:`np.ndarray`): The points at which the
            derivative is evaluated. A 1D array or list of parameter values
            matching the expected input of the function.

    Returns:
        :class:`np.ndarray`: the gradient of the function, as an array.
    """

    n_parameters = theta0.shape[0]
    gradient = np.zeros(n_parameters, dtype=float)

    for m in range(n_parameters):
        # 1 parameter to differentiate, and n_parameters-1 parameters to hold fixed
        theta0_x = deepcopy(theta0)
        function_to_diff = get_partial_function(
            function, m, theta0_x
        )
        kit = DerivativeKit(function_to_diff, theta0[m])
        gradient[m] = kit.adaptive.differentiate(
            order=1, n_workers=n_workers
        )
    return gradient

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
    
def get_partial_function(
    full_function, variable_index, fixed_values
):
    """Returns a single-variable version of a multivariate function.

    A single parameter must be specified by index. All others parameters
    are held fixed.

    Args:
        full_function (callable): A function that takes a list of
            n_parameters parameters and returns a vector of n_observables
            observables.
        variable_index (int): The index of the parameter to treat as the
            variable.
        fixed_values (list or np.ndarray): The list of parameter values to
            use as fixed inputs for all parameters except the one being
            varied.

    Returns:
        callable: A function of a single variable, suitable for use in
            differentiation.
    """

    def partial_function(x):
        params = deepcopy(fixed_values)
        params[variable_index] = x
        return np.atleast_1d(full_function(params))

    return partial_function
