r"""JAX-based autodiff helpers for DerivKit.

This module does not register any DerivKit backend by default.

Use these functions directly, or see
:func:`derivkit.autodiff.jax_autodiff.register_jax_autodiff_backend`
for an opt-in integration.

Use only with JAX-differentiable functions. For arbitrary models, prefer
the "adaptive" or "finite" methods.

Shape conventions (aligned with DerivKit calculus builders):

- ``autodiff_derivative``:
  :math:`f:\\mathbb{R}\\mapsto\\mathbb{R}` → returns ``float`` (scalar)

- ``autodiff_gradient``:
  :math:`f:\\mathbb{R}^n\\mapsto\\mathbb{R}` → returns array of shape ``(n,)``

- ``autodiff_jacobian``:
  :math:`f:\\mathbb{R}^n\\mapsto\\mathbb{R}^m` (or tensor output) → returns array of
  shape ``(m, n)``, where ``m = \\prod(\text{out\\_shape})``

- ``autodiff_hessian``:
  :math:`f:\\mathbb{R}^n\\mapsto\\mathbb{R}` → returns array of shape ``(n, n)``
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from derivkit.autodiff.jax_utils import (
    AutodiffUnavailable,
    apply_array_nd,
    apply_scalar_1d,
    apply_scalar_nd,
    jax,
    jnp,
    require_jax,
)

__all__ = [
    "autodiff_derivative",
    "autodiff_gradient",
    "autodiff_jacobian",
    "autodiff_hessian",
]


def autodiff_derivative(func: Callable, x0: float, order: int = 1) -> float:
    """Calculates the k-th derivative of a function f: R -> R via JAX autodiff.

    Args:
        func: Callable mapping float -> scalar.
        x0: Point at which to evaluate the derivative.
        order: Derivative order (>=1); uses repeated grad for higher orders.

    Returns:
        Derivative value as a float.

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        ValueError: If order < 1.
        TypeError: If func(x) is not scalar-valued.
    """
    require_jax()

    if order < 1:
        raise ValueError("autodiff_derivative: order must be >= 1.")

    f_jax = partial(apply_scalar_1d, func, "autodiff_derivative")

    g = f_jax
    for _ in range(order):
        g = jax.grad(g)

    try:
        val = g(x0)
    except (TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_derivative: function is not JAX-differentiable at x0. "
            "Use JAX primitives / jax.numpy or fall back to 'adaptive'/'finite'."
        ) from exc

    return float(val)


def autodiff_gradient(func: Callable, x0) -> np.ndarray:
    """Computes the gradient of a scalar-valued function f: R^n -> R via JAX autodiff.

    Args:
        func: Function to be differentiated.
        x0: Point at which to evaluate the gradient.

    Returns:
        A gradient vector as a 1D numpy.ndarray with shape (n,).

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        TypeError: If func(theta) is not scalar-valued.
    """
    require_jax()

    x0_arr = np.asarray(x0, float).ravel()

    f_jax = partial(apply_scalar_nd, func, "autodiff_gradient")
    grad_f = jax.grad(f_jax)

    try:
        g = grad_f(jnp.asarray(x0_arr))
    except (TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_gradient: function is not JAX-differentiable."
        ) from exc

    return np.asarray(g, dtype=float).reshape(-1)


def autodiff_jacobian(
    func: Callable,
    x0,
    *,
    mode: str | None = None,
) -> np.ndarray:
    """Calculates the Jacobian of a vector-valued function via JAX autodiff.

    Output convention matches DerivKit Jacobian builders: we flatten the function
    output to length m = prod(out_shape), and return a 2D Jacobian of shape (m, n),
    with n = input dimension.

    Args:
        func: Function to be differentiated.
        x0: Point at which to evaluate the Jacobian; array-like, shape (n,).
        mode: Differentiation mode; None (auto), 'fwd', or 'rev'.
            If None, chooses 'rev' if m <= n, else 'fwd'. For more details about
            modes, see JAX documentation for `jax.jacrev` and `jax.jacfwd`.

    Returns:
        A Jacobian matrix as a 2D numpy.ndarray with shape (m, n).

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        ValueError: If mode is invalid.
        TypeError: If func(theta) is scalar-valued.
    """
    require_jax()

    x0_arr = np.asarray(x0, float).ravel()
    x0_jax = jnp.asarray(x0_arr)

    f_jax = partial(apply_array_nd, func, "autodiff_jacobian")

    try:
        y0 = f_jax(x0_jax)
    except (TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_jacobian: function is not JAX-differentiable at x0."
        ) from exc

    in_dim = x0_arr.size
    out_dim = int(np.prod(y0.shape))

    if mode is None:
        use_rev = out_dim <= in_dim
    elif mode == "rev":
        use_rev = True
    elif mode == "fwd":
        use_rev = False
    else:
        raise ValueError("autodiff_jacobian: mode must be None, 'fwd', or 'rev'.")

    jac_fun = jax.jacrev if use_rev else jax.jacfwd

    try:
        jac = jac_fun(f_jax)(x0_jax)
    except (TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_jacobian: failed to trace function with JAX."
        ) from exc

    jac_np = np.asarray(jac, dtype=float)
    return jac_np.reshape(out_dim, in_dim)


def autodiff_hessian(func: Callable, x0) -> np.ndarray:
    """Calculates the full Hessian of a scalar-valued function.

    Args:
        func: A function to be differentiated.
        x0: Point at which to evaluate the Hessian; array-like, shape (n,) with
            n = input dimension.

    Returns:
        A Hessian matrix as a 2D numpy.ndarray with shape (n, n).

    Raises:
        AutodiffUnavailable: If JAX is not available or function is not differentiable.
        TypeError: If func(theta) is not scalar-valued.
    """
    require_jax()

    x0_arr = np.asarray(x0, float).ravel()
    x0_jax = jnp.asarray(x0_arr)

    f_jax = partial(apply_scalar_nd, func, "autodiff_hessian")

    try:
        hess = jax.hessian(f_jax)(x0_jax)
    except (TypeError, ValueError) as exc:
        raise AutodiffUnavailable(
            "autodiff_hessian: function is not JAX-differentiable."
        ) from exc

    return np.asarray(hess, dtype=float)
