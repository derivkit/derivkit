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


def build_hessian(
    function: Callable[[ArrayLike], np.float64 | np.ndarray],
    theta0: np.ndarray,
    method: str | None = None,
    n_workers: int = 1,
    return_diag: bool = False,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Returns the hessian of a function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Parallel tasks across output components / Hessian entries (outer).
        return_diag: If True, compute and return only the diagonal as a 1D array per output.
        **dk_kwargs: Extra options forwarded to `DerivativeKit.differentiate`.
            You may optionally pass `inner_workers=<int>` here to override the inner policy.

    Returns:
        If ``function(theta0)`` is scalar, returns a 2D array (p, p) representing the Hessian.
        If ``return_diag`` is True, returns only the diagonal as (p,).

        If ``function(theta0)`` is tensor-valued with shape ``out_shape``,
        the full Hessian has shape ``(*out_shape, p, p)``. If ``return_diag`` is True,
        returns only the diagonal as ``(*out_shape, p)``.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If a scalar component path does not return a scalar value.
    """
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # Evaluate once to determine output shape
    y0 = np.asarray(function(theta))
    out_shape = y0.shape

    # Worker policy: outer is this call; inner inherited via concurrency utils
    inner_override = dk_kwargs.pop("inner_workers", None)
    outer = int(n_workers) if n_workers is not None else 1
    inner = int(inner_override) if inner_override is not None else resolve_inner_from_outer(outer)

    # Scalar-output path: identical behavior to your original implementation
    if y0.ndim == 0:
        if return_diag:
            return _build_hessian_scalar_diag(function, theta, method, outer, inner, **dk_kwargs)
        return _build_hessian_scalar_full(function, theta, method, outer, inner, **dk_kwargs)

    # Tensor-output path: flatten outputs; compute one Hessian per component; stack/reshape back
    m = y0.size
    tasks = [
        (i, theta, method, outer, inner, return_diag, dk_kwargs, function)
        for i in range(m)
    ]
    vals = parallel_execute(
        _compute_component_hessian,
        tasks,
        outer_workers=outer,
        inner_workers=inner,
    )
    arr = np.stack(vals, axis=0)                 # (m, p) or (m, p, p)
    arr = arr.reshape(out_shape + arr.shape[1:]) # -> (*out_shape, p) or (*out_shape, p, p)
    if not np.isfinite(arr).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    return arr


def build_hessian_diag(
    function: Callable[[ArrayLike], np.float64 | np.ndarray],
    theta0: np.ndarray,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the diagonal of the Hessian of a function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Parallel tasks across output components / Hessian entries (outer).
        **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.
            You may optionally pass `inner_workers=<int>` here to override the inner policy.

    Returns:
        If ``function(theta0)`` is scalar, returns a 1D array (p,) representing the diagonal.
        If ``function(theta0)`` is tensor-valued with shape ``out_shape``,
        returns an array with shape ``(*out_shape, p)``.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If a scalar component path does not return a scalar value.
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


def _compute_component_hessian(
        idx: int,
        theta: NDArray[np.floating],
        method: str | None,
        _outer_workers: int,
        inner_workers: int | None,
        return_diag: bool,
        dk_kwargs: dict,
        function: Callable,) -> NDArray[np.floating]:
    """Computes the Hessian (or its diagonal) for a single flattened output component.

    Args:
        idx: The index of the flattened output component.
        theta: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite").
        _outer_workers: Number of outer parallel workers (not used here).
        inner_workers: Number of inner workers (set via context by parallel_execute).
        return_diag: If True, compute and return only the diagonal.
        dk_kwargs: Additional keyword arguments for DerivativeKit.differentiate.
        function: The original function that may return a tensor.

    Returns:
        The Hessian (2D array) or its diagonal (1D array) for the specified component.
    """
    # Adapter to scalar component
    g = partial(_component_scalar_eval, function=function, idx=int(idx))

    if return_diag:
        return _build_hessian_scalar_diag(g, theta, method, 1, inner_workers, **dk_kwargs)
    return _build_hessian_scalar_full(g, theta, method, 1, inner_workers, **dk_kwargs)


def _component_scalar_eval(
    theta_vec: NDArray[np.floating],
    *,
    function: Callable,
    idx: int,
) -> float:
    """Returns the scalar value of the `idx`-th flattened component of f(theta_vec).

    Args:
        theta_vec: The parameter vector at which the function is evaluated.
        function: The original function that may return a tensor.
        idx: The index of the flattened output component to extract.

    Returns:
        The scalar value of the specified component.
    """
    val = np.asarray(function(theta_vec))
    return float(val.ravel()[idx])


def _build_hessian_scalar_full(
    function: Callable,
    theta: np.ndarray,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the full (p, p) Hessian for a scalar-valued function.

    Args:
        function: The function to be differentiated (must return a scalar).
        theta: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite").
        outer_workers: Number of outer parallel workers for Hessian entries.
        inner_workers: Number of inner workers (set via context by parallel_execute).
        **dk_kwargs: Additional keyword arguments for DerivativeKit.differentiate.

    Returns:
        A 2D array representing the Hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        TypeError: If ``function`` does not return a scalar value.
    """
    p = int(theta.size)

    # Assemble tasks for diagonal and upper triangle; symmetry is enforced in assembly
    tasks: list[Tuple[Any, ...]] = [(function, theta, i, i, method, inner_workers, dk_kwargs) for i in range(p)]
    tasks += [
        (function, theta, i, j, method, inner_workers, dk_kwargs)
        for i in range(p) for j in range(i + 1, p)
    ]

    vals = parallel_execute(
        _hessian_component_worker,
        tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    hess = np.empty((p, p), dtype=float)
    k = 0
    for i in range(p):
        hess[i, i] = float(vals[k])
        k += 1
    for i in range(p):
        for j in range(i + 1, p):
            hij = float(vals[k])
            k += 1
            hess[i, j] = hij
            hess[j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    return hess


def _build_hessian_scalar_diag(
    function: Callable,
    theta: np.ndarray,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the diagonal (p,) of the Hessian for a scalar-valued function.

    Args:
        function: The function to be differentiated (must return a scalar).
        theta: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., "adaptive", "finite").
        outer_workers: Number of outer parallel workers for diagonal entries.
        inner_workers: Number of inner workers (set via context by parallel_execute).
        **dk_kwargs: Additional keyword arguments for DerivativeKit.differentiate.

    Returns:
        A 1D array representing the diagonal of the Hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        TypeError: If ``function`` does not return a scalar value.
    """
    p = int(theta.size)

    tasks: list[Tuple[Any, ...]] = [(function, theta, i, i, method, inner_workers, dk_kwargs) for i in range(p)]
    vals = parallel_execute(
        _hessian_component_worker,
        tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    diag = np.asarray(vals, dtype=float)
    if not np.isfinite(diag).all():
        raise FloatingPointError("Non-finite values encountered in Hessian diagonal.")
    return diag


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
        inner_workers: Number of inner workers (set via context by parallel_execute).
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
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> float:
    """Returns one entry of the Hessian for a scalar-valued function.

    This function measures how the rate of change in one parameter depends
    on another. It handles both the pure and mixed second derivatives:
      - If i == j, this is the second derivative with respect to a single parameter.
      - If i != j, this is the mixed derivative, computed by first finding
        how the function changes with parameter i while holding parameter j fixed,
        and then differentiating that result with respect to parameter j.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of inner workers (set via context by parallel_execute).
        **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        A single number showing how the rate of change in one parameter
        depends on another.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    # Mixed derivative path:
    # Define a helper that computes how the function changes with parameter i
    # when parameter j is temporarily set to a specific value.
    # Then we take the derivative of that helper with respect to parameter j.
    if i == j:
        partial_vec1 = get_partial_function(function, i, theta0)
        probe = np.asarray(partial_vec1(float(theta0[i])), dtype=float)
        if probe.size != 1:
            raise TypeError("build_hessian() expects a scalar-valued function.")
        kit1 = DerivativeKit(partial_vec1, float(theta0[i]))
        return kit1.differentiate(order=2, method=method, n_workers=n_workers, **dk_kwargs)

    # Mixed derivative path:
    path = partial(
        _mixed_partial_value,
        function=function,
        theta0=theta0,
        i=i,
        j=j,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    kit2 = DerivativeKit(path, float(theta0[j]))
    return kit2.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)


def _mixed_partial_value(
    y: float,
    *,
    function: Callable[[ArrayLike], np.float64],
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    n_workers: int | None,
    dk_kwargs: dict,
) -> float:
    """Returns the first derivative with respect to parameter i while temporarily setting parameter j to a given value.

    This helper does not compute the second derivative itself. It only returns
    the first derivative of the function with respect to one parameter while
    holding another fixed. The caller then takes the derivative of this result
    with respect to that fixed parameter to get the mixed second derivative.

    Args:
        y: The value to set for parameter j.
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., "adaptive", "finite"). If None,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of inner workers (set via context by parallel_execute).
        dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.

    Returns:
        The value of the partial derivative with respect to parameter i
        when parameter j is set to y.
    """
    theta = theta0.copy()
    theta[j] = float(y)
    partial_vec1 = get_partial_function(function, i, theta)
    kit1 = DerivativeKit(partial_vec1, float(theta[i]))
    return float(kit1.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs))
