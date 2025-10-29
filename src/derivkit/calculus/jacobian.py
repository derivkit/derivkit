"""Contains functions used to construct the Jacobian matrix."""

from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function


def build_jacobian(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    method: str | None = None,
    n_workers: int | None = 1,
    dk_kwargs: dict | None = None,
) -> NDArray[np.floating]:
    """Computes the Jacobian of a vector-valued function.

    Each column in the Jacobian is the derivative with respect to one parameter.

    Args:
        function: The vector-valued function to be differentiated.
            It should accept a list or array of parameter values as input and
            return an array of observable values.
        theta0: The parameter vector at which the jacobian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers used to parallelize across
            parameters. If None or 1, no parallelization is used.
            If greater than 1, this many threads will be used to compute
            derivatives with respect to different parameters in parallel.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A 2D array representing the jacobian. Each column corresponds to
            the derivative with respect to one parameter.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a vector value.
    """
    # Validate inputs and evaluate baseline output
    theta = np.asarray(theta0, dtype=float).ravel()
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    y0 = np.asarray(function(theta), dtype=float)
    if y0.ndim != 1:
        raise TypeError(
            f"build_jacobian expects f: R^n -> R^m with 1-D vector output; got shape {y0.shape}"
        )
    if not np.isfinite(y0).all():
        raise FloatingPointError("Non-finite values in model output at theta0.")

    m = int(y0.size)
    n = int(theta.size)

    # Resolve parallelism policy
    try:
        outer_workers = max(1, int(n_workers or 1))
    except (TypeError, ValueError):
        outer_workers = 1
    inner_workers = resolve_inner_from_outer(outer_workers)

    # Prepare worker
    worker = partial(
        _column_derivative,
        function=function,
        theta0=theta,
        method=method,
        inner_workers=inner_workers,
        dk_kwargs=dk_kwargs,
        expected_m=m,
    )

    # Parallelize across parameters
    cols = parallel_execute(
        worker,
        arg_tuples=[(j,) for j in range(n)],
        outer_workers=outer_workers,
        inner_workers=inner_workers,  # passed for context; we also pass explicitly to worker
    )

    # Stack columns → (m, n)
    jac = np.column_stack([np.asarray(c, dtype=float).reshape(m) for c in cols])
    return jac


def _column_derivative(
    j: int,
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict | None,
    expected_m: int,
) -> NDArray[np.floating]:
    """Derivative of function with respect to parameter j.

    Args:
        j: Index of the parameter to differentiate with respect to.
        function: The vector-valued function to be differentiated.
        theta0: The parameter vector at which the jacobian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        inner_workers: Number of workers used by DerivativeKit.adaptive.differentiate.
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.
        expected_m: Expected length of the derivative vector.

    Returns:
        A 1D array representing the derivative with respect to parameter j.

    Raises:
        TypeError: If the derivative does not have the expected length.
        FloatingPointError: If non-finite values are encountered.
    """
    theta_x = np.asarray(theta0, dtype=float).ravel().copy()

    # Single-variable view: f_j(y) where theta_x[j] = y, others fixed
    f_j = get_partial_function(function, j, theta_x)

    # Normalize method aliases for the new DK API
    # (let DK handle validation; we only map common shorthands)
    method_norm = None
    if method is not None:
        m = method.lower()
        alias = {"auto": "adaptive", "fd": "finite"}
        method_norm = alias.get(m, m)

    # Differentiate via new unified API (passes method through)
    kit = DerivativeKit(f_j, theta_x[j])
    g = kit.differentiate(method=method_norm, order=1, n_workers=inner_workers, **(dk_kwargs or {}))

    g = np.atleast_1d(np.asarray(g, dtype=float)).reshape(-1)
    if g.size != expected_m:
        raise TypeError(
            f"Expected derivative of length {expected_m} but got {g.size} for parameter index {j}."
        )
    if not np.isfinite(g).all():
        raise FloatingPointError(f"Non-finite derivative for parameter index {j}.")

    return g
