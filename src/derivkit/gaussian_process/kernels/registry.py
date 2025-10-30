"""Kernel registry for Gaussian Process backends.

This module provides a tiny registry so kernels can be discovered and selected
by name. Kernels are instantiated once and reused (they are expected to be
stateless).
"""

from __future__ import annotations

from typing import Dict, Iterable, Type

import numpy as np

from derivkit.gaussian_process.kernels.base import Kernel

__all__ = ["register_kernel", "get_kernel", "list_kernels", "validate_kernel_params"]

_registry: Dict[str, Kernel] = {}


def register_kernel(name: str):
    """Register a kernel class under a given name.

    This is a class decorator. The class is instantiated once and stored in the
    registry. Names are normalized to lowercase.

    Args:
        name: Short identifier for the kernel (e.g., "rbf").

    Raises:
        ValueError: If the name is already registered to a different class.

    Returns:
        The decorator that registers the class.
    """
    norm = name.lower().strip()

    def deco(cls: Type[Kernel]):
        if norm in _registry and _registry[norm].__class__ is not cls:
            raise ValueError(
                f"Kernel name '{norm}' is already registered to "
                f"{_registry[norm].__class__.__name__}; got {cls.__name__}."
            )
        # Kernels should be stateless; keep a single instance.
        _registry[norm] = cls()  # type: ignore[call-arg]
        return cls

    return deco


def get_kernel(name: str) -> Kernel:
    """Return a registered kernel instance by name.

    Args:
        name: Kernel name (case-insensitive).

    Returns:
        The kernel instance associated with the name.

    Raises:
        ValueError: If the name is unknown.
    """
    norm = name.lower().strip()
    try:
        return _registry[norm]
    except KeyError as exc:
        available = ", ".join(sorted(_registry)) or "<none>"
        raise ValueError(
            f"Unknown kernel '{name}'. Available: {available}"
        ) from exc


def list_kernels() -> Iterable[str]:
    """List registered kernel names.

    Returns:
        Tuple of kernel names sorted alphabetically.
    """
    return tuple(sorted(_registry.keys()))


def validate_kernel_params(kernel: Kernel, params: dict) -> None:
    """Validate parameter keys for a specific kernel.

    This checks that provided keys are a subset of the kernel's declared
    ``param_names``. It also enforces that ``length_scale`` is a scalar unless
    the kernel advertises (automatic relevance determination) ARD support via ``ard_ok``.

    Args:
        kernel: Kernel instance whose schema to validate against.
        params: Parameter dictionary to validate.

    Raises:
        ValueError: If unknown parameters are present or ARD is not supported.
    """
    allowed = set(kernel.param_names)
    extra = set(params) - allowed
    if extra:
        raise ValueError(
            f"Unexpected parameter(s) {sorted(extra)} for '{kernel.name}'. "
            f"Expected a subset of {sorted(allowed)}."
        )

    if "length_scale" in params and not kernel.ard_ok:

        if np.ndim(params["length_scale"]) > 0:
            raise ValueError(
                f"Kernel '{kernel.name}' does not support array length_scale."
            )
