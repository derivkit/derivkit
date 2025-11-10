"""Experimental JAX-based autodiff helpers for DerivKit.

This module does not register any DerivKit backend by default.

Use these functions directly, or see `jax_backend.register_jax_autodiff`
for an opt-in integration.

Use only with JAX-differentiable functions. For arbitrary models, prefer
the "adaptive" or "finite" methods.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    from jax.errors import ConcretizationTypeError
except Exception:  # JAX not installed / misconfigured
    jax = None
    jnp = None
    ConcretizationTypeError = RuntimeError  # type: ignore[assignment]
    _HAS_JAX = False
else:
    _HAS_JAX = True

has_jax: bool = _HAS_JAX

__all__ = [
    "has_jax",
    "AutodiffUnavailable",
    "autodiff_derivative",
    "autodiff_gradient",
    "autodiff_jacobian",
    "autodiff_hessian",
]


class AutodiffUnavailable(RuntimeError):
    """Raised when JAX autodiff cannot be used."""


def _require_jax() -> None:
    """Raises AutodiffUnavailable if JAX is not available."""
    if not _HAS_JAX:
        raise AutodiffUnavailable(
            "JAX autodiff requires `jax` + `jaxlib`.\n"
            "Install them with `pip install jax jaxlib`, or use "
            "DerivKit's 'adaptive' or 'finite' methods instead."
        )


def _to_jax_scalar(y: Any, *, where: str) -> jnp.ndarray:
    """Ensures that output is scalar and returns as JAX array.

    Args:
        y: Output to check.
        where: Context string for error messages.

    Returns:
        JAX array with shape ().
    """
    arr = jnp.asarray(y)
    if arr.ndim != 0:
        raise TypeError(
            f"{where}: expected scalar output; got shape {tuple(arr.shape)}."
        )
    return arr


def _to_jax_array(y: Any, *, where: str) -> jnp.ndarray:
    """Ensures that output is array-like (not scalar) and returns as JAX array.

    Args:
        y: Output to check.
        where: Context string for error messages.

    Returns:
        JAX array with shape (m,) or higher.
    """
    try:
        arr = jnp.asarray(y)
    except TypeError as exc:
        raise TypeError(
            f"{where}: output could not be converted to a JAX array."
        ) from exc
    if arr.ndim == 0:
        raise TypeError(
            f"{where}: output is scalar; use autodiff_derivative/gradient instead."
        )
    return arr


def _apply_scalar_1d(func: Callable[[float], Any], where: str, x: jnp.ndarray) -> jnp.ndarray:
    """Adapts the function f: R -> R with scalar output enforcement.

    Args:
        func: User function mapping float -> scalar.
        where: Context string for error messages.
        x: JAX array with shape () (scalar input).

    Returns:
        JAX array with shape () (scalar output).
    """
    y = func(x)
    return _to_jax_scalar(y, where=where)


def _apply_scalar_nd(func: Callable, where: str, theta: jnp.ndarray) -> jnp.ndarray:
    """Adapts the function f: R^n -> R with scalar output enforcement.

    Args:
        func: User function mapping array-like -> scalar.
        where: Context string for error messages.
        theta: JAX array with shape (n,) (input vector).

    Returns:
        JAX array with shape () (scalar output).
    """
    y = func(theta)
    return _to_jax_scalar(y, where=where)


def _apply_array_nd(func: Callable, where: str, theta: jnp.ndarray) -> jnp.ndarray:
    """Adapts the function f: R^n -> R^m with array output enforcement.

    Args:
        func: User function mapping array-like -> array-like (non-scalar).
        where: Context string for error messages.
        theta: JAX array with shape (n,) (input vector).

    Returns:
        JAX array with shape (m,) or higher (output vector/tensor).
    """
    y = func(theta)
    return _to_jax_array(y, where=where)


def autodiff_derivative(func: Callable, x0: float, order: int = 1) -> float:
    """Derivative of a scalar function at a scalar point via JAX autodiff.

    Args:
        func: Callable mapping float -> scalar.
        x0: Point at which to evaluate the derivative.
        order: Derivative order (>=1); uses repeated grad for higher orders.

    Returns:
        Derivative value as a float.

    Requirements:
        - func must be JAX-differentiable at x0.
        - order >= 1.

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        ValueError: If order < 1.
    """
    _require_jax()

    if order < 1:
        raise ValueError("autodiff_derivative: order must be >= 1.")

    # make sure we have a JAX-wrapped version of func
    f_jax = partial(_apply_scalar_1d, func, "autodiff_derivative")

    g = f_jax
    for _ in range(order):
        g = jax.grad(g)

    try:
        val = g(x0)
    except (ConcretizationTypeError, TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_derivative: function is not JAX-differentiable at x0. "
            "Use JAX primitives / jax.numpy or fall back to 'adaptive'/'finite'."
        ) from exc

    return float(val)


def autodiff_gradient(func: Callable, x0) -> _np.ndarray:
    """Computes the gradient of f: R^n -> R at x0 via JAX autodiff.

    Args:
        func: Callable mapping array-like -> scalar.
        x0: Point at which to evaluate the gradient; array-like, shape (n,
            where n is the number of parameters.

    Returns:
        1D numpy.ndarray with shape (n_params,).
    Requirements:
        - func(theta) returns a scalar.
        - func must be JAX-compatible.
        - x0 is array-like; treated as 1D vector.

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
    """
    _require_jax()

    x0_arr = _np.asarray(x0, float).ravel()

    # f_jax(theta) = scalar output
    f_jax = partial(_apply_scalar_nd, func, "autodiff_gradient")
    grad_f = jax.grad(f_jax)

    try:
        g = grad_f(jnp.asarray(x0_arr))
    except (ConcretizationTypeError, TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_gradient: function is not JAX-differentiable."
        ) from exc

    return _np.asarray(g, dtype=float)


def autodiff_jacobian(
    func: Callable,
    x0,
    *,
    mode: str | None = None,
) -> _np.ndarray:
    """Calculates the Jacobian of a vector/tensor function f: R^n -> R^m via JAX autodiff.

    Args:
        func: Callable mapping array-like -> array-like (non-scalar).
        x0: Point at which to evaluate the Jacobian; array-like, shape (n,).
        mode: Differentiation mode; None (auto), 'fwd', or 'rev'.
            If None, chooses 'rev' if m <= n, else 'fwd'.

    Returns:
        2D numpy.ndarray with shape (m, n), where m is the output dimension.

    Requirements:
        - func must be JAX-differentiable at x0.
        - x0 is array-like; treated as 1D vector.

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        ValueError: If mode is invalid.
    """
    _require_jax()

    x0_arr = _np.asarray(x0, float).ravel()
    x0_jax = jnp.asarray(x0_arr)

    # ensure here we have a JAX-wrapped version of func that returns array output
    f_jax = partial(_apply_array_nd, func, "autodiff_jacobian")

    try:
        y0 = f_jax(x0_jax)
    except (ConcretizationTypeError, TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_jacobian: function is not JAX-differentiable at x0."
        ) from exc

    out_shape = y0.shape
    in_dim = x0_arr.size
    out_dim = int(_np.prod(out_shape))

    if mode is None:
        use_rev = out_dim <= in_dim  # reverse-mode if output not "much bigger" than input
    elif mode == "rev":
        use_rev = True
    elif mode == "fwd":
        use_rev = False
    else:
        raise ValueError("autodiff_jacobian: mode must be None, 'fwd', or 'rev'.")

    jac_fun = jax.jacrev if use_rev else jax.jacfwd

    try:
        jac = jac_fun(f_jax)(x0_jax)
    except (ConcretizationTypeError, TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_jacobian: failed to trace function with JAX."
        ) from exc

    return _np.asarray(jac, dtype=float)


def autodiff_hessian(func: Callable, x0) -> _np.ndarray:
    """Calcualtes the Hessian of f: R^n -> R at x0 via JAX autodiff.

    Args:
        func: Callable mapping array-like -> scalar.
        x0: Point at which to evaluate the Hessian; array-like, shape (n
            where n is the number of parameters.

    Returns:
        2D numpy.ndarray with shape (n, n).

    Requirements:
        - func(theta) returns a scalar.
        - func must be JAX-compatible.
        - x0 is array-like; treated as 1D vector.

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
    """
    _require_jax()

    x0_arr = _np.asarray(x0, float).ravel()
    x0_jax = jnp.asarray(x0_arr)

    # f_jax(theta) = scalar output
    f_jax = partial(_apply_scalar_nd, func, "autodiff_hessian")

    try:
        hess = jax.hessian(f_jax)(x0_jax)
    except (ConcretizationTypeError, TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_hessian: function is not JAX-differentiable."
        ) from exc

    return _np.asarray(hess, dtype=float)
