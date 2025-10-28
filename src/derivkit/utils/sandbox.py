"""Sandbox utilities for experimentation and testing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "get_partial_function",
    "generate_test_function",
]


def get_partial_function(
    full_function: Callable,
    variable_index: int,
    fixed_values: list | np.ndarray,
) -> Callable:
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

    Raises:
        ValueError: If ``fixed_values`` is not 1D or if `variable_index`` is out of bounds.
        TypeError: If ``variable_index`` is not an integer.
        IndexError: If ``variable_index`` is out of bounds for the size of ``fixed_values``.
    """
    fixed_arr = np.asarray(fixed_values, dtype=float)
    if fixed_arr.ndim != 1:
        raise ValueError(
            f"fixed_values must be 1D; got shape {fixed_arr.shape}."
        )
    if not isinstance(variable_index, (int, np.integer)):
        raise TypeError(
            f"variable_index must be an integer; got {type(variable_index).__name__}."
        )
    if variable_index < 0 or variable_index >= fixed_arr.size:
        raise IndexError(
            f"variable_index {variable_index} out of bounds for size {fixed_arr.size}."
        )

    def partial_function(x):
        params = fixed_arr.copy()
        params[variable_index] = x
        return np.atleast_1d(full_function(params))

    return partial_function


def generate_test_function(name: str = "sin"):
    """Return (f, f', f'') tuple for a named test function.

    Args:
        name: One of {"sin"}; more may be added.

    Returns:
        Tuple of callables (f, df, d2f) for testing.
    """
    if name == "sin":
        return lambda x: np.sin(x), lambda x: np.cos(x), lambda x: -np.sin(x)
    raise ValueError(f"Unknown test function: {name!r}")
