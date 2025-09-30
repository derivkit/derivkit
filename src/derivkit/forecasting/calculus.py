"""Differential calculus helpers."""

from collections.abc import Callable, Iterable
from multiprocessing import Pool

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
    jacobian = np.array(
        [
            _jacobian_component(function, theta0, i, n_workers)
            for i in range(theta0.size)
        ],
        dtype=float,
    )
    if not np.isfinite(jac).all():
        raise FloatingPointError("Non-finite values encountered in jacobian.")
    return jac.T


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


def build_hessian(function, theta0, n_workers=1):
    """Returns the hessian of a scalar-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 2D array representing the hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
    """
    theta0 = np.asarray(theta0, dtype=float)
    p = theta0.size
    hess = np.empty((p, p), dtype=float)

    i, j = np.triu_indices(p)  # upper triangle incl. diagonal
    pairs = list(zip(i, j))
    vals = np.asarray(_compute_hessian_entries(function, theta0, pairs, n_workers))

    hess[i, j] = vals
    hess[j, i] = vals

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in hessian.")
    return hess


def _hess_entry(function: Callable, theta0: np.ndarray, i: int, j: int) -> float:
    """Compute one component of the hessian of a scalar-valued function.

    Helper used by ``build_hessian``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        i (int): Zero-based index of the first parameter with respect to which to differentiate.
        j (int): Zero-based index of the second parameter with respect to which to differentiate.

    Returns:
        (``float``): The (i, j) component of the hessian of function evaluated at ``theta0``.
    """
    # IMPORTANT: do not spawn workers inside; this is the unit of parlelism
    return _hessian_component(function, theta0, i, j, n_workers=1)


def _compute_hessian_entries(
    function: Callable,
    theta0: np.ndarray,
    pairs: Iterable[tuple[int, int]],
    n_workers: int = 1,
) -> list[float]:
    """Compute Hessian entries for given (i, j) index pairs.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        pairs (Iterable[tuple[int, int]]): Iterable of (i, j) index
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``list``): List of hessian entries corresponding to the input pairs.
    """
    tasks = [(function, theta0, int(i), int(j)) for (i, j) in pairs]
    if n_workers and n_workers > 1:
        with Pool(processes=n_workers) as pool:
            return pool.starmap(_hess_entry, tasks)
    else:
        return [_hess_entry(*t) for t in tasks]


def build_hessian_diag(function: Callable, theta0, n_workers: int = 1) -> np.ndarray:
    """Diagonal of the Hessian at theta0 as shape (p, ).

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 1D array representing the diagonal of the hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
    """
    theta0 = np.asarray(theta0, dtype=float)
    p = theta0.size
    idx = np.arange(p, dtype=int)
    pairs = list(zip(idx, idx))  # (i, i)
    vals = np.asarray(_compute_hessian_entries(function, theta0, pairs, n_workers))
    if not np.isfinite(vals).all():
        raise FloatingPointError("Non-finite values encountered in hessian_diag.")
    return vals


def hessian_triangle(
    function: Callable, theta0, which: str = "upper", n_workers: int = 1
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return only the specified triangle (incl. diagonal) entries plus their indices.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the hessian is evaluated.
        which (str): One of {"upper", "lower"} to specify which triangle to return
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        (``np.array``): 1D array of the requested triangle entries of the h
        (``tuple``): Tuple of two 1D arrays (i, j) of zero-based indices corresponding to the entries.

    Raises:
        ValueError: If ``which`` is not one of {"upper", "lower"}.
    """
    theta0 = np.asarray(theta0, dtype=float)
    p = theta0.size
    if which == "upper":
        i, j = np.triu_indices(p)
    elif which == "lower":
        i, j = np.tril_indices(p)
    else:
        raise ValueError("which must be 'upper' or 'lower'")

    pairs = list(zip(i, j))
    vals = np.asarray(_compute_hessian_entries(function, theta0, pairs, n_workers))
    return vals, (i, j)


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
        partial_vec1 = get_partial_function(function, i, theta0)

        # One-time scalar check for hessian()
        probe = np.asarray(partial_vec1(theta0[i]), dtype=float)
        if probe.size != 1:
            raise TypeError(
                "hessian() expects a scalar-valued function; "
                f"got shape {probe.shape} from full_function(params)."
            )

        kit1 = DerivativeKit(partial_vec1, theta0[i])
        deriv = kit1.adaptive.differentiate(order=2, n_workers=n_workers)
        return deriv

    else:
        # 2 parameters to differentiate once, with other parameters held fixed
        kit2 = DerivativeKit(mixed_first_deriv_wrt_i, theta0[j])
        deriv = kit2.adaptive.differentiate(order=1, n_workers=n_workers)
        return deriv

def mixed_first_deriv_wrt_i(y, function, theta0, i, j, n_workers):
    """Helper for computing mixed second derivatives for Hessian.

    Args:
        y: The value to substitute for theta[j].
        function: The scalar-valued function to differentiate.
        theta0: The points at which the derivative is evaluated.
        i: Zero-based index of the first parameter with respect to which to differentiate.
        j: Zero-based index of the second parameter with respect to which to differentiate.
        n_workers: Number of workers used inside ``DerivativeKit.adaptive.differentiate``.
                   This does not parallelize across parameters.

    Returns:
        The first derivative with respect to parameter i, evaluated at theta[j] = y.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    theta[j] = y
    g_i = get_partial_function(function, i, theta)
    dk = DerivativeKit(g_i, theta[i])
    deriv = dk.adaptive.differentiate(order=1, n_workers=n_workers)
    return deriv


def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError
