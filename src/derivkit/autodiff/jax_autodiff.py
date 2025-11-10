"""JAX-based autodiff backend for DerivativeKit (experimental)."""

from __future__ import annotations

from typing import Any, Callable

from derivkit.derivative_kit import register_method

from .jax_core import (
    AutodiffUnavailable,
    autodiff_derivative,
    has_jax,
)

__all__ = [
    "AutodiffDerivative",
    "register_jax_autodiff_backend",
]


class AutodiffDerivative:
    """DerivativeKit engine for JAX-based autodiff (experimental).

    This uses :func:`autodiff_derivative` under the hood and therefore
    supports scalar functions f: R -> R with JAX-differentiable bodies.
    """

    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initializes the JAX autodiff derivative engine.

        Args:
            function: Callable mapping a float to a scalar output.
            x0: Expansion point about which derivatives are computed.
        """
        self.function = function
        self.x0 = x0

    def differentiate(self, *, order: int = 1, **_: Any) -> float:
        """Compute the k-th derivative via JAX autodiff.

        Args:
            order: Derivative order (>=1).

        Returns:
            The k-th derivative at x0 as a float.

        """
        return autodiff_derivative(self.function, self.x0, order=order)


def register_jax_autodiff_backend(
    *,
    name: str = "autodiff",
    aliases: tuple[str, ...] = ("jax", "jax-autodiff", "jax-diff"),
) -> None:
    """Register the experimental JAX autodiff backend with DerivativeKit.

    After calling this, you can use:

        DerivativeKit(f, x0).differentiate(method=name, order=...)

    This is opt-in on purpose: JAX is an optional dependency and only
    works for JAX-compatible functions.

    Raises:
        AutodiffUnavailable: If JAX is not available.
    """
    if not has_jax:
        # Reuse your existing error type/message style.
        raise AutodiffUnavailable(
            "Cannot register JAX autodiff backend: JAX is not available.\n"
            "Install `jax` + `jaxlib`, or use 'adaptive' / 'finite' instead."
        )

    register_method(
        name=name,
        cls=AutodiffDerivative,
        aliases=aliases,
    )
