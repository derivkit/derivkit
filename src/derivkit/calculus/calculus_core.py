"""Core utilities for calculus-based derivative computations.

This module provides shared helper functions for building derivative
objects (gradients, Jacobians, Hessians, higher-order tensors) of scalar-
and tensor-valued functions using DerivativeKit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "component_scalar_eval",
]


def component_scalar_eval(
    theta_vec: NDArray[np.float64],
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    idx: int,
) -> float:
    """Evaluates a single element of the model output ``function(theta_vec)`` as a scalar.

    This helper is used internally when building derivatives of models that
    return multiple outputs (e.g. vectors or arrays). Derivative routines
    operate on scalar-valued functions, so one output component is selected
    and treated as a scalar function of the parameters.

    Args:
        theta_vec: 1D parameter vector.
        function: Original function.
        idx: Index of the output component to differentiate, where the output is first
            flattened with NumPy C-order (i.e. ``np.ravel(y, order="C")``).

    Returns:
        Scalar value of the specified output component.

    Raises:
        IndexError: If ``idx`` is out of bounds for the model output.
    """
    theta_vec = np.asarray(theta_vec, dtype=np.float64)
    val = np.asarray(function(theta_vec))

    flat = np.ravel(val, order="C")
    i = int(idx)

    if i < 0 or i >= flat.size:
        raise IndexError(
            f"Output index {i} out of bounds for model output of size {flat.size}."
        )

    return float(flat[i])
