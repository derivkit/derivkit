"""Differential calculus helpers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function

__all__ = [
    "gradient",
    "jacobian",
    "hessian_diag",
    "build_hessian",
    "gauss_newton_hessian",
]


def gradient(function, theta0, n_workers=1):
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

    # One-time scalar check for gradient()
    _check_scalar_valued(function, theta0, 0, n_workers)

    # n_workers controls inner 1D differentiation (not across parameters).
    grad = np.array(
        [
            _grad_component(function, theta0, i, n_workers)
            for i in range(theta0.size)
        ],
        dtype=float,
    )
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in gradient.")
    return grad


def _grad_component(
    function, theta0: np.ndarray, i: int, n_workers: int
) -> float:
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
    """
    partial_vec = get_partial_function(function, i, theta0)

    kit = DerivativeKit(partial_vec, theta0[i])
    return kit.adaptive.differentiate(order=1, n_workers=n_workers)


def jacobian(
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
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    y0 = np.atleast_1d(function(theta0))
    m, n = int(y0.size), int(theta0.size)

    if not np.isfinite(y0).all():
        raise FloatingPointError("Non-finite values in model output at theta0.")

    try:
        work = int(n_workers) if n_workers is not None else 1
    except (TypeError, ValueError):
        work = 1
    if work < 1:
        work = 1

    if work == 1:
        cols = [_grad_for_param(function, theta0, j) for j in range(n)]
    else:
        worker = partial(_grad_for_param, function, theta0)
        with ThreadPoolExecutor(max_workers=work) as ex:
            cols = list(ex.map(worker, range(n)))

    # Ensure each column is length-m; works even when m==1 or n==1
    jacobian = np.column_stack([np.asarray(c, dtype=float).reshape(m) for c in cols])
    return jacobian


def _jacobian_component(
    function: Callable, theta0: np.ndarray, i: int, n_workers: int
) -> np.ndarray:
    """Compute one column of the jacobian of a scalar-valued function.

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
    return kit.adaptive.differentiate(order=1, n_workers=n_workers)

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
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    f0 = np.asarray(function(theta0), dtype=float)
    if f0.size != 1:
        raise TypeError("build_hessian() expects a scalar-valued function.")

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
        TypeError: If the function is not scalar-valued.
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


def _check_scalar_valued(function, theta0: np.ndarray, i: int, n_workers: int):
    """Helper used by ``gradient`` and ``build_hessian``.

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

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)

    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )

def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError


def _grad_wrt_param(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    idx: int,
) -> NDArray[np.floating]:
    """Derivative of a vector-valued function wrt a single parameter theta[idx].

     Helper used by ``jacobian``. Wraps ``function`` into a single-variable callable via
    ``derivkit.utils.get_partial_function`` and differentiates it with
    ``DerivativeKit.adaptive.differentiate``.

    Args:
        function: The vector-valued function to be differentiated.
        theta0: The parameter vector at which the derivative is evaluated.
        idx: Zero-based index of the parameter with respect to which to differentiate.

    Returns:
        A 1D array representing the derivative of the function with respect to theta[idx].
    """
    theta0_x = deepcopy(np.atleast_1d(theta0))
    f_i = get_partial_function(function, idx, theta0_x)   # this sets theta[idx]=y
    kit = DerivativeKit(f_i, theta0_x[idx])
    # Keep inner serial to avoid nested pools; adaptive can still batch-eval internally.
    gi = kit.adaptive.differentiate(order=1)
    gi = np.asarray(gi, dtype=float).reshape(-1)  # ensure (m,)
    return gi


def _grad_for_param(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    j: int,
) -> NDArray[np.floating]:
    """Derivative of a vector-valued function wrt a single parameter theta[j].

    Helper used by ``jacobian``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

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
