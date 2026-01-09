"""Unit tests for derivkit.autodiff.jax_core functions."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivatives.autodiff.jax_utils import (
    AutodiffUnavailable,
    require_jax,
)

try:
    require_jax()
except AutodiffUnavailable:
    pytest.skip(
        'JAX not installed; install with `pip install "derivkit[jax]"`.',
        allow_module_level=True,
    )

import jax.numpy as jnp

from derivkit.derivatives.autodiff.jax_core import (
    autodiff_derivative,
    autodiff_gradient,
    autodiff_hessian,
    autodiff_jacobian,
)


def _skip_if_no_jax() -> None:
    """Skips the test if JAX is not available."""
    try:
        require_jax()
    except AutodiffUnavailable:
        pytest.skip("JAX not installed")


def square(x):
    """Square function."""
    return x**2


def cube(x):
    """Cubic function."""
    return x**3


def identity(x):
    """Identity function."""
    return x


def grad_func(t):
    """Gradient function: f(t) = t[0]^2 + 3*t[1]."""
    return t[0] ** 2 + 3.0 * t[1]


def jac_func(t):
    """Jacobian function: f(t) = [t[0] + t[1], t[0] * t[1]]."""
    return jnp.array([t[0] + t[1], t[0] * t[1]])


def hess_func(t):
    """Hessian function: f(t) = t[0]^2 + 2*t[0]*t[1] + t[1]^2."""
    return t[0] ** 2 + 2.0 * t[0] * t[1] + t[1] ** 2


def nondiff_func(t):
    """Non-differentiable function: f(t) = |t[0]|."""
    if float(t[0]) > 0:
        return 1.0
    return 0.0


def test_autodiff_derivative_first_order() -> None:
    """Tests that autodiff_derivative computes first-order derivative correctly."""
    _skip_if_no_jax()

    val = autodiff_derivative(square, 3.0)
    assert val == pytest.approx(6.0)


def test_autodiff_derivative_higher_order() -> None:
    """Tests that autodiff_derivative computes higher-order derivative correctly."""
    _skip_if_no_jax()

    val = autodiff_derivative(cube, 2.0, order=2)
    assert val == pytest.approx(12.0)


def test_autodiff_derivative_invalid_order() -> None:
    """Tests that autodiff_derivative raises ValueError for invalid order."""
    _skip_if_no_jax()

    with pytest.raises(ValueError):
        autodiff_derivative(identity, 1.0, order=0)


def test_autodiff_gradient_simple() -> None:
    """Tests that autodiff_gradient computes gradient correctly."""
    _skip_if_no_jax()

    g = autodiff_gradient(grad_func, [2.0, 1.0])
    assert g.shape == (2,)
    assert np.allclose(g, [4.0, 3.0])


def test_autodiff_jacobian_simple() -> None:
    """Tests that autodiff_jacobian computes Jacobian correctly."""
    _skip_if_no_jax()

    j = autodiff_jacobian(jac_func, [2.0, 3.0])
    assert j.shape == (2, 2)
    assert np.allclose(j, [[1.0, 1.0], [3.0, 2.0]])


def test_autodiff_jacobian_invalid_mode() -> None:
    """Tests that autodiff_jacobian raises ValueError for invalid mode."""
    _skip_if_no_jax()

    with pytest.raises(ValueError):
        autodiff_jacobian(identity, [1.0], mode="bad")


def test_autodiff_hessian_simple() -> None:
    """Tests that autodiff_hessian computes Hessian correctly."""
    _skip_if_no_jax()

    h = autodiff_hessian(hess_func, [1.0, 2.0])
    assert h.shape == (2, 2)
    assert np.allclose(h, [[2.0, 2.0], [2.0, 2.0]])


def test_autodiff_gradient_nondifferentiable_raises() -> None:
    """Tests that autodiff_gradient raises AutodiffUnavailable for non-differentiable function."""
    _skip_if_no_jax()

    with pytest.raises(AutodiffUnavailable):
        autodiff_gradient(nondiff_func, [1.0])
