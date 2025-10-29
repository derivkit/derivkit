"""Contains functions used to construct the gradient of scalar-valued functions."""

from collections.abc import Callable
from functools import partial

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function
from derivkit.utils.validate import check_scalar_valued


def build_gradient(function: Callable,
                   theta0: np.ndarray,
                   method: str | None = None,
                   n_workers=1,
                   **dk_kwargs: dict
                   ) -> np.ndarray:
    """Returns the gradient of a scalar-valued function.

    Args:
        function (Callable): The function to be differentiated.
        theta0  (array-like): The parameter vector at which the gradient is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters. Default is 1.
        dk_kwargs (dict, optional): Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A 1D array representing the gradient.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    check_scalar_valued(function, theta0, 0, n_workers)

    inner = resolve_inner_from_outer(n_workers)

    # Bind shared kwargs once; tasks only carry (function, theta0, i)
    worker = partial(
        _grad_component,
        method=method,
        n_workers=inner,
        dk_kwargs=dk_kwargs,
    )
    tasks = [(function, theta0, i) for i in range(theta0.size)]

    vals = parallel_execute(worker, tasks, outer_workers=n_workers, inner_workers=inner)
    grad = np.asarray(vals, dtype=float)
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in build_gradient.")
    return grad


def _grad_component(
        function: Callable,
        theta0: np.ndarray,
        i:int,
        method: str | None = None,
        n_workers: int = 1,
        dk_kwargs: dict | None = None,
) -> float:
    """Returns one entry of the gradient for a scalar-valued function.

    Used inside ``build_gradient`` to find how the function changes with respect
    to a single parameter while keeping the others fixed.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: The index of the parameter being varied.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers used for the internal derivative step. Default is 1.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A single number showing how the function changes with that parameter.
    """
    partial_vec = get_partial_function(function, i, theta0)
    kit = DerivativeKit(partial_vec, float(theta0[i]))
    return kit.differentiate(order=1, method=method, n_workers=n_workers, **(dk_kwargs or {}))
