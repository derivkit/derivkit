"""JAX-based autodiff backend for DerivativeKit.

This backend is intentionally minimal: it only supports scalar derivatives
$f: R \mapsto R$ via JAX autodiff, and must be registered explicitly as explained in the example.

Example:
--------

Basic usage (opt-in registration):

    >>> from derivkit.derivative_kit import DerivativeKit
    >>> from derivkit.autodiff.jax_autodiff import register_jax_autodiff_backend
    >>> register_jax_autodiff_backend()  # must be called first to enable JAX backend
    >>>
    >>> def func(x):
    ...     # Use jax.numpy inside the function body for best compatibility.
    ...     import jax.numpy as jnp
    ...     return jnp.sin(x) + 0.5 * x**2
    ...
    >>> dk = DerivativeKit(func, 1.0)
    >>> dk.differentiate(method="autodiff", order=1)
    1.0...
    >>> dk.differentiate(method="autodiff", order=2)
    0.5...

Notes:
------

- This backend is scalar-only. For gradients/Jacobians/Hessians of functions
  with vector inputs/outputs, use the standalone helpers in
  ``derivkit.autodiff.jax_core`` (e.g. ``autodiff_gradient``).
- To enable this backend, install the JAX extra: ``pip install "derivkit[jax]"``.
"""


from __future__ import annotations

from typing import Any, Callable

from derivkit.autodiff.jax_core import autodiff_derivative
from derivkit.autodiff.jax_utils import require_jax
from derivkit.derivative_kit import register_method

__all__ = [
    "AutodiffDerivative",
    "register_jax_autodiff_backend",
]


class AutodiffDerivative:
    """DerivativeKit engine for JAX-based autodiff.

    Supports scalar functions f: R -> R with JAX-differentiable bodies.
    """

    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initializes the JAX autodiff derivative engine."""
        self.function = function
        self.x0 = float(x0)

    def differentiate(self, *, order: int = 1, **_: Any) -> float:
        """Computes the k-th derivative via JAX autodiff.

        Args:
            order: Derivative order (>=1).

        Returns:
            Derivative value as a float.
        """
        return autodiff_derivative(self.function, self.x0, order=order)


def register_jax_autodiff_backend(
    *,
    name: str = "autodiff",
    aliases: tuple[str, ...] = ("jax", "jax-autodiff", "jax-diff", "jd"),
) -> None:
    """Registers the experimental JAX autodiff backend with DerivativeKit.

    After calling this, you can use:

        DerivativeKit(f, x0).differentiate(method=name, order=...)

    Args:
        name: Name of the method to register.
        aliases: Alternative names for the method.

    Returns:
        None

    Raises:
        AutodiffUnavailable: If JAX is not available.
    """
    require_jax()

    register_method(
        name=name,
        cls=AutodiffDerivative,
        aliases=aliases,
    )
