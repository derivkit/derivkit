"""Kernel registry and built-in kernels for GaussianProcess.

This subpackage provides:
  * A tiny **registry** to look up kernels by name.
  * A built-in **RBFKernel** (squared-exponential) with analytic first/second
    derivative support.
  * Convenience re-exports so callers can do:
        from derivkit.gaussian_process.kernels import get_kernel, RBFKernel

Kernels are **stateless** objects that implement the :class:`Kernel` protocol
(see `kernels/base.py`). They expose methods to compute covariances and the
covariances involving first/second derivatives. Because kernels are stateless,
the registry stores a single shared instance per kernel type.

Typical usage:
    >>> from derivkit.gaussian_process.kernels import list_kernels, get_kernel
    >>> list_kernels()
    ('rbf',)
    >>> k = get_kernel("rbf")   # returns the registered RBF kernel instance

Registering a custom kernel:
    >>> from derivkit.gaussian_process.kernels import register_kernel
    ...
    >>> @register_kernel("matern32")
    ... class Matern32:
    ...     \"\"\"Example kernel; implement the Kernel protocol methods.\"\"\"
    ...     name = "matern32"
    ...     param_names = ("length_scale", "output_scale")
    ...     ard_ok = True
    ...     # implement k(...), cov_value_grad(...), cov_grad_grad(...),
    ...     # cov_value_hessdiag(...), cov_hessdiag_samepoint(...)

Parameter validation:
    Use :func:`validate_kernel_params` to check that a parameter dictionary only
    contains keys a kernel expects (e.g., ``length_scale``, ``output_scale``),
    and that shapes are compatible with the kernel’s ARD support.

Public API re-exports:
    - :func:`register_kernel`
    - :func:`get_kernel`
    - :func:`list_kernels`
    - :func:`validate_kernel_params`
    - :class:`RBFKernel`
"""

from __future__ import annotations

# Ensure the built-in RBF kernel class is importable from this package.
from derivkit.gaussian_process.kernels.rbf_kernel import (
    RBFKernel,  # noqa: F401
)
from derivkit.gaussian_process.kernels.registry import (
    get_kernel,
    list_kernels,
    register_kernel,
    validate_kernel_params,
)

__all__ = [
    "register_kernel",
    "get_kernel",
    "list_kernels",
    "validate_kernel_params",
    "RBFKernel",
]
