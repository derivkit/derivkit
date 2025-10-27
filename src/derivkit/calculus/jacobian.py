"""Contains functions used to construct the Jacobian matrix."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function


def build_jacobian(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    n_workers: int | None = 1,
) -> NDArray[np.floating]:
    """Computes the Jacobian of a vector-valued function.

    Each column in the Jacobian is the derivative with respect to one parameter.

    Args:
        function: The vector-valued function to be differentiated.
            It should accept a list or array of parameter values as input and
            return an array of observable values.
        theta0: The parameter vector at which the jacobian is evaluated.
        n_workers: Number of workers used to parallelize across
            parameters. If None or 1, no parallelization is used.
            If greater than 1, this many threads will be used to compute
            derivatives with respect to different parameters in parallel.

    Returns:
        A 2D array representing the jacobian. Each column corresponds to
            the derivative with respect to one parameter.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a vector value.
    """
    # Validate and flatten theta0
    theta = np.asarray(theta0, dtype=float).ravel()
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # Use flattened theta when calling the function
    y0 = np.asarray(function(theta), dtype=float)
    if y0.ndim != 1:
        raise TypeError(
            f"build_jacobian expects f: R^n -> R^m with 1-D vector output; got shape {y0.shape}"
        )
    if not np.isfinite(y0).all():
        raise FloatingPointError("Non-finite values in model output at theta0.")

    m = y0.size
    n = int(theta.size)

    # Resolve worker count robustly
    try:  # first try to convert to int
        work = max(1, int(n_workers or 1))
    except (TypeError, ValueError):
        work = 1  # fallback to serial

    if work == 1:  # serial computation
        cols = [_grad_for_param(function, theta, j) for j in range(n)]
    else:  # parallel computation
        worker = partial(_grad_for_param, function, theta)
        with ThreadPoolExecutor(max_workers=work) as ex:
            cols = list(ex.map(worker, range(n)))

    # Ensure all columns are finite and stack into 2D array
    jacobian = np.column_stack([np.asarray(c, dtype=float).reshape(m) for c in cols])
    return jacobian

def _grad_for_param(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    j: int,
) -> NDArray[np.floating]:
    """Return how all outputs of a function change when one input changes.

    This helper is used by ``jacobian`` to compute the effect of changing
    a single parameter while keeping the others fixed.

    Args:
        function: The vector-valued function to be differentiated.
        theta0: The parameter vector at which the derivative is evaluated.
        j: Zero-based index of the parameter with respect to which to differentiate.

    Returns:
        A 1D array representing the derivative of the function with respect to theta[j].
    """
    theta_x = deepcopy(np.asarray(theta0, dtype=float).reshape(-1))
    f_j = get_partial_function(function, j, theta_x)  # this sets theta[j]=y
    kit = DerivativeKit(f_j, theta_x[j])
    g = kit.adaptive.differentiate(order=1, n_workers=1)
    # Keep inner differentiation serial to avoid nested pools.
    g = np.atleast_1d(np.asarray(g, dtype=float))
    if not np.isfinite(g).all():
        raise FloatingPointError(f"Non-finite derivative for parameter index {j}.")
    return g
