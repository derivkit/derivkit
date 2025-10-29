"""Contains functions used in constructing the Hessian of a scalar-valued function."""

from collections.abc import Callable
from functools import partial
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function

__all__ = [
    "build_hessian",
    "build_hessian_diag",
]


def build_hessian(function: Callable,
                  theta0: np.ndarray,
                  method: str | None = None,
                  n_workers: int=1,
                  return_diag: bool = False,
                  **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Returns the hessian of a scalar-valued function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Parallel tasks across Hessian entries (outer).
        return_diag: If True, compute and return only the diagonal as a 1D array.
        **dk_kwargs: Extra options forwarded to `DerivativeKit.differentiate`.

    Returns:
        A 2D array representing the hessian. If ``return_diag`` is True, returns only the
        diagonal as a 1D array.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a scalar value.
    """
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    f0 = np.asarray(function(theta), dtype=float)
    if f0.size != 1:
        raise TypeError("build_hessian() expects a scalar-valued function.")

    n = theta.size
    inner = resolve_inner_from_outer(n_workers)

    # Build tasks (include dk_kwargs so options propagate)
    if return_diag:
        tasks: list[Tuple[Any, ...]] = [
            (function, theta, i, i, method, inner, dk_kwargs) for i in range(n)
        ]
    else:
        tasks = [(function, theta, i, i, method, inner, dk_kwargs) for i in range(n)]
        tasks += [
            (function, theta, i, j, method, inner, dk_kwargs)
            for i in range(n) for j in range(i + 1, n)
        ]

    vals = parallel_execute(
        _hessian_component_worker,
        tasks,
        outer_workers=n_workers,
        inner_workers=inner,
    )

    if return_diag:
        diag = np.asarray(vals, dtype=float)
        if not np.isfinite(diag).all():
            raise FloatingPointError("Non-finite values encountered in Hessian diagonal.")
        return diag

    # Assemble full symmetric matrix
    hess = np.empty((n, n), dtype=float)
    k = 0
    for i in range(n):
        hess[i, i] = float(vals[k])
        k += 1
    for i in range(n):
        for j in range(i + 1, n):
            hij = float(vals[k])
            k += 1
            hess[i, j] = hij
            hess[j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    return hess


def build_hessian_diag(
        function: Callable,
        theta0: np.ndarray,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the diagonal of the hessian of a scalar-valued function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers used by DerivativeKit.adaptive.differentiate.
        **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A 1D array representing the diagonal of the hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a scalar value.
    """
    return build_hessian(
        function=function,
        theta0=theta0,
        method=method,
        n_workers=n_workers,
        return_diag=True,
        **dk_kwargs,
    )


def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError


def _hessian_component_worker(
    function: Callable,
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict,
) -> float:
    """Returns one entry of the Hessian for a scalar-valued function.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        inner_workers: Number of workers used for the internal derivative step.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A single number showing how the rate of change in one parameter
        depends on another.
    """
    return _hessian_component(
        function=function,
        theta0=theta0,
        i=i,
        j=j,
        method=method,
        n_workers=inner_workers,
        **dk_kwargs,
    )


def _hessian_component(
        function: Callable,
        theta0: np.ndarray,
        i: int,
        j: int,
        method: str | None = None,
        n_workers: int | None = 1,
        **dk_kwargs: Any,
) -> float:
    """Return one entry of the Hessian for a scalar-valued function.

    Used inside ``build_hessian`` to measure how the function’s change in one
    parameter depends on changes in another. This can describe both pure
    second derivatives and mixed ones.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers used for the internal derivative step.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A single number showing how the rate of change in one parameter
        depends on another.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    if i == j:
        partial_vec1 = get_partial_function(function, i, theta0)
        probe = np.asarray(partial_vec1(float(theta0[i])), dtype=float)
        if probe.size != 1:
            raise TypeError("build_hessian() expects a scalar-valued function.")
        kit1 = DerivativeKit(partial_vec1, float(theta0[i]))
        return kit1.differentiate(order=2, method=method, n_workers=n_workers, **dk_kwargs)

    # Here we make a mixed partial derivative
    path = partial(
        _mixed_partial_value,
        function=function,
        theta0=theta0,
        i=i,
        j=j,
        method=method,
        inner_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    kit2 = DerivativeKit(path, float(theta0[j]))
    return kit2.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)


def _mixed_partial_value(
    y: float,
    function: Callable,
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict,
) -> float:
    """Returns the value of the partial derivative w.r.t. theta_i at theta_j = y.

    Args:
        y: The value to set for parameter j.
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        inner_workers: Number of workers used for the internal derivative step.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        The value of the partial derivative with respect to parameter i
        when parameter j is set to y.
    """
    theta = theta0.copy()
    theta[j] = y
    partial_vec1 = get_partial_function(function, i, theta)
    kit1 = DerivativeKit(partial_vec1, float(theta[i]))
    return float(kit1.differentiate(order=1, method=method, n_workers=inner_workers, **dk_kwargs))
