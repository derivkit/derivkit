"""Differential calculus helpers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function

__all__ = [
    "build_gradient",
    "build_jacobian",
    "hessian_diag",
    "build_hessian",
    "gauss_newton_hessian",
]


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
    # Here I am forcing the symmetry
    hess = 0.5 * (hess + hess.T)
    return hess


def _hessian_component(function: Callable, theta0: np.ndarray, i: int, j: int, n_workers: int) -> float:
    """Central-difference entry (i,j) of the Hessian for scalar f:R^n->R.

    Diagonals:
        (f(x+h e_i) - 2 f(x) + f(x-h e_i)) / h^2
    Off-diagonals:
        (f(x+hi e_i + hj e_j) - f(x+hi e_i - hj e_j)
         - f(x-hi e_i + hj e_j) + f(x-hi e_i - hj e_j)) / (4 hi hj)
    """
    x = np.asarray(theta0, dtype=float).ravel()
    eps = np.finfo(float).eps
    base = eps ** 0.25  # ~1.2e-4
    scale = 8.0  # was 1.0; increase to tame 1/h^2 roundoff
    hi = max(1e-6, scale * base * (1.0 + abs(x[i])))

    if i == j:
        ei = np.zeros_like(x)
        ei[i] = 1.0
        f0 = float(function(x))
        fp = float(function(x + hi * ei))
        fm = float(function(x - hi * ei))
        if not (np.isfinite(fp) and np.isfinite(fm) and np.isfinite(f0)):
            raise FloatingPointError("Non-finite in diagonal stencil.")
        return (fp - 2.0 * f0 + fm) / (hi ** 2)

    # mixed partials
    hj = max(1e-6, scale * base * (1.0 + abs(x[j])))
    ei = np.zeros_like(x)
    ei[i] = 1.0
    ej = np.zeros_like(x)
    ej[j] = 1.0

    fpp = float(function(x + hi*ei + hj*ej))
    fpm = float(function(x + hi*ei - hj*ej))
    fmp = float(function(x - hi*ei + hj*ej))
    fmm = float(function(x - hi*ei - hj*ej))

    if not (np.isfinite(fpp) and np.isfinite(fpm) and np.isfinite(fmp) and np.isfinite(fmm)):
        raise FloatingPointError("Non-finite in mixed-partial stencil.")

    return (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)


def hessian_diag(*args, **kwargs):
    """This is a placeholder for a Hessian diagonal computation function."""
    raise NotImplementedError


def _check_scalar_valued(function, theta0: np.ndarray, i: int, n_workers: int):
    """Helper used by ``build_gradient`` and ``build_hessian``.

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
            "build_gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )

def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError


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
    theta = np.asarray(theta0, dtype=float).ravel().copy()

    # Per-parameter step (good default for central differences)
    sqrt_eps = np.sqrt(np.finfo(float).eps)  # ~1.49e-8
    h = max(1e-8, sqrt_eps * (1.0 + abs(theta[j])))

    # Evaluate f at theta ± h e_j
    tp = theta.copy()
    tp[j] += h
    tm = theta.copy()
    tm[j] -= h

    fp = np.asarray(function(tp), dtype=float)
    fm = np.asarray(function(tm), dtype=float)

    # Must be finite 1-D vectors
    if fp.ndim != 1 or fm.ndim != 1:
        raise TypeError("Function must return a 1-D vector.")
    if (not np.isfinite(fp).all()) or (not np.isfinite(fm).all()):
        raise FloatingPointError("Non-finite values in function outputs.")

    return (fp - fm) / (2.0 * h)
