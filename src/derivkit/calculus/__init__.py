"""Calculus utilities.

Provides constructors for gradient, Jacobian, and Hessian computations.
"""

from derivkit.calculus.gradient import build_gradient
from derivkit.calculus.hessian import build_hessian, build_hessian_diag
from derivkit.calculus.hyper_hessian import build_hyper_hessian
from derivkit.calculus.jacobian import build_jacobian

__all__ = [
    "build_gradient",
    "build_jacobian",
    "build_hessian",
    "build_hessian_diag",
    "build_hyper_hessian",
]
