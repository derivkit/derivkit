"""Utilities for JAX-based autodiff in DerivKit."""

from __future__ import annotations

from typing import Any, Callable

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
    _HAS_JAX = False
else:
    _HAS_JAX = True

has_jax: bool = _HAS_JAX

__all__ = [
    "AutodiffUnavailable",
    "require_jax",
    "to_jax_scalar",
    "to_jax_array",
    "apply_scalar_1d",
    "apply_scalar_nd",
    "apply_array_nd",
]


class AutodiffUnavailable(RuntimeError):
    """Raises when JAX-based autodiff is unavailable."""


def require_jax() -> None:
    """Raises if JAX is not available.

    Args:
        None.

    Returns:
        None.

    Raises:
        AutodiffUnavailable: If JAX is not installed.
    """
    if not _HAS_JAX:
        raise AutodiffUnavailable(
            "JAX autodiff requires `jax` + `jaxlib`.\n"
            'Install with `pip install "derivkit[jax]"` '
            "(or follow JAX's official install instructions for GPU)."
        )


def to_jax_scalar(y: Any, *, where: str) -> "jnp.ndarray":
    """Ensures that output is scalar and returns as JAX array.

    Args:
        y: Output to check.
        where: Context string for error messages.

    Returns:
        JAX array with shape ().

    Raises:
        TypeError: If output is not scalar.
    """
    arr = jnp.asarray(y)
    if arr.ndim != 0:
        raise TypeError(f"{where}: expected scalar output; got shape {tuple(arr.shape)}.")
    return arr


def to_jax_array(y: Any, *, where: str) -> "jnp.ndarray":
    """Ensures that output is array-like (not scalar) and returns as JAX array.

    Args:
        y: Output to check.
        where: Context string for error messages.

    Returns:
        JAX array with shape (m,) or higher.

    Raises:
        TypeError: If output is scalar or cannot be converted to JAX array.
    """
    try:
        arr = jnp.asarray(y)
    except TypeError as exc:
        raise TypeError(f"{where}: output could not be converted to a JAX array.") from exc
    if arr.ndim == 0:
        raise TypeError(f"{where}: output is scalar; use autodiff_derivative/gradient instead.")
    return arr


def apply_scalar_1d(
    func: Callable[[float], Any],
    where: str,
    x: "jnp.ndarray",
) -> "jnp.ndarray":
    """Takes an input function and maps it over a 1D array with scalar output enforcement.

    Args:
        func: Function to apply.
        where: Context string for error messages.
        x: 1D JAX array of inputs.

    Returns:
        JAX array of scalar outputs.
    """
    return to_jax_scalar(func(x), where=where)


def apply_scalar_nd(
    func: Callable,
    where: str,
    theta: "jnp.ndarray",
) -> "jnp.ndarray":
    """Takes an input function and maps it over an ND array with scalar output enforcement.

    Args:
        func: Function to apply.
        where: Context string for error messages.
        theta: ND JAX array of inputs.

    Returns:
        JAX array of scalar outputs.
    """
    return to_jax_scalar(func(theta), where=where)


def apply_array_nd(
    func: Callable,
    where: str,
    theta: "jnp.ndarray",
) -> "jnp.ndarray":
    """Takes an input function and maps it over an ND array with array output enforcement.

    Args:
        func: Function to apply.
        where: Context string for error messages.
        theta: ND JAX array of inputs.

    Returns:
        JAX array of array outputs.
    """
    return to_jax_array(func(theta), where=where)
