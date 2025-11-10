"""Unit tests for the JAX-based autodiff backend in DerivativeKit."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.autodiff.jax_autodiff import register_jax_autodiff_backend
from derivkit.autodiff.jax_core import (
    AutodiffUnavailable,
    autodiff_gradient,
    autodiff_hessian,
    has_jax,
)
from derivkit.derivative_kit import DerivativeKit, available_methods

if has_jax:
    # Only register the backend when JAX is actually available.
    register_jax_autodiff_backend()

    import jax
    import jax.numpy as jnp

    # Make autodiff “high precision” so it’s a good reference.
    jax.config.update("jax_enable_x64", True)
else:
    # For type-checkers / function definitions; tests that use jnp are skipped.
    jnp = None  # type: ignore[assignment]


def _rel_err(a: float, b: float) -> float:
    """Computes relative error between a and b."""
    a = float(a)
    b = float(b)
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) / scale


def _analytic_derivative(name: str, order: int, x: float) -> float:
    """Analytic k-th derivative for our small test set."""
    if name == "sin":
        # cycle: sin, cos, -sin, -cos, ...
        k = order % 4
        if k == 1:
            return np.cos(x)
        elif k == 2:
            return -np.sin(x)
        elif k == 3:
            return -np.cos(x)
        else:  # k == 0
            return np.sin(x)

    if name == "cos":
        # cycle: cos, -sin, -cos, sin, ...
        k = order % 4
        if k == 1:
            return -np.sin(x)
        elif k == 2:
            return -np.cos(x)
        elif k == 3:
            return np.sin(x)
        else:  # k == 0
            return np.cos(x)

    if name == "exp":
        # all derivatives of exp are exp
        return np.exp(x)

    if name == "poly3":
        # f(x) = x^3 - 2x + 0.5
        if order == 1:
            return 3 * x**2 - 2.0
        elif order == 2:
            return 6 * x
        elif order == 3:
            return 6.0
        else:  # higher orders vanish
            return 0.0

    raise KeyError(f"Unknown function '{name}'")


# JAX-compatible versions for autodiff (only used when has_jax is True)
def _f_sin(x):
    """JAX-compatible sine function."""
    return jnp.sin(x)


def _f_cos(x):
    """JAX-compatible cosine function."""
    return jnp.cos(x)


def _f_exp(x):
    """JAX-compatible exponential function."""
    return jnp.exp(x)


def _f_poly3(x):
    """JAX-compatible cubic polynomial function."""
    return x**3 - 2.0 * x + 0.5


SCALAR_FUNCS = {
    "sin": _f_sin,
    "cos": _f_cos,
    "exp": _f_exp,
    "poly3": _f_poly3,
}

# Full grid for robust autodiff checks (autodiff is stable here).
X0_VALUES = [0.0, 0.3, -1.1, 2.5, -5.6, 3.3, -30.3]

# Subset where adaptive / finite are expected to behave "nicely".
X0_VALUES_STABLE = [0.0, 0.3, -1.1, 2.5, -5.6, 3.3]


def test_autodiff_method_is_registered():
    """Checks that the autodiff method is registered when JAX is present."""
    if not has_jax:
        pytest.skip("JAX not available; autodiff backend is not registered.")
    methods = set(available_methods())
    assert "autodiff" in methods


def test_autodiff_without_jax_raises():
    """In a no-JAX environment, trying to enable the backend fails cleanly."""
    if has_jax:
        pytest.skip("JAX is installed; this test is for no-JAX environments.")

    with pytest.raises(AutodiffUnavailable):
        register_jax_autodiff_backend()


pytestmark = pytest.mark.skipif(not has_jax, reason="autodiff backend requires JAX")

@pytest.mark.parametrize("name", list(SCALAR_FUNCS.keys()))
@pytest.mark.parametrize("x0", X0_VALUES)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_autodiff_matches_analytic_scalar(name, x0, order):
    """Tests that JAX autodiff matches analytic derivatives for scalars."""
    f = SCALAR_FUNCS[name]
    dk = DerivativeKit(function=f, x0=x0)

    d_auto = dk.differentiate(method="autodiff", order=order)
    exact = _analytic_derivative(name, order, x0)

    # autodiff should be our tightest backend
    assert _rel_err(d_auto, exact) < 1e-9


@pytest.mark.parametrize("name", list(SCALAR_FUNCS.keys()))
@pytest.mark.parametrize("x0", X0_VALUES_STABLE)
@pytest.mark.parametrize("order", [1, 2])
def test_adaptive_matches_analytic_scalar_reasonable(name, x0, order):
    """Tests that adaptive differentiation matches analytic derivatives for scalars."""
    f = SCALAR_FUNCS[name]
    dk = DerivativeKit(function=f, x0=x0)

    d_adapt = dk.differentiate(method="adaptive", order=order)
    exact = _analytic_derivative(name, order, x0)

    assert _rel_err(d_adapt, exact) < 1e-4


@pytest.mark.parametrize("name", list(SCALAR_FUNCS.keys()))
@pytest.mark.parametrize("x0", X0_VALUES_STABLE)
def test_finite_matches_analytic_scalar_first_derivative(name, x0):
    """Tests that finite-difference differentiation matches analytic first derivatives for scalars."""
    f = SCALAR_FUNCS[name]
    dk = DerivativeKit(function=f, x0=x0)

    d_fd = dk.differentiate(method="finite", order=1)
    exact = _analytic_derivative(name, 1, x0)

    assert _rel_err(d_fd, exact) < 1e-4


@pytest.mark.parametrize("name", ["sin", "cos", "exp", "poly3"])
@pytest.mark.parametrize("x0", [0.0, 0.3])
def test_autodiff_vs_adaptive_first_derivative(name, x0):
    """Tests that JAX autodiff and adaptive differentiation agree on first derivatives."""
    f = SCALAR_FUNCS[name]
    dk = DerivativeKit(function=f, x0=x0)

    d_auto = dk.differentiate(method="autodiff", order=1)
    d_adapt = dk.differentiate(method="adaptive", order=1)

    # Loose: just catch egregious mismatches.
    assert _rel_err(d_auto, d_adapt) < 1e-3


def quad_2d_xy(x: float, y: float) -> float:
    """A simple quadratic function in two dimensions with known gradient and Hessian."""
    return x**2 + y**2 + 0.5 * x * y


def quad_2d(theta: np.ndarray) -> float:
    """A simple quadratic function in two dimensions with known gradient and Hessian."""
    return quad_2d_xy(theta[0], theta[1])


def test_autodiff_gradient_matches_analytic():
    """Tests that autodiff_gradient matches analytic gradient for a simple 2D function."""
    x0 = np.array([0.4, -0.7], dtype=float)
    g = autodiff_gradient(quad_2d, x0)

    # For f(x, y) = x^2 + y^2 + 0.5xy:
    # ∂f/∂x = 2x + 0.5y
    # ∂f/∂y = 2y + 0.5x
    expected = np.array(
        [
            2.0 * x0[0] + 0.5 * x0[1],
            2.0 * x0[1] + 0.5 * x0[0],
        ],
        dtype=float,
    )

    assert g.shape == (2,)
    assert np.allclose(g, expected, rtol=1e-7, atol=1e-9)


def test_autodiff_hessian_matches_analytic():
    """Tests that autodiff_hessian matches analytic Hessian for a simple 2D function."""
    # f(x, y) = x^2 + y^2 + 0.5xy
    # Hessian =
    # [2.0, 0.5]
    # [0.5, 2.0]

    x0 = np.array([0.1, -0.3], dtype=float)

    hess = autodiff_hessian(quad_2d, x0)

    expected = np.array([[2.0, 0.5],
                         [0.5, 2.0]], dtype=float)

    assert hess.shape == (2, 2)
    assert np.allclose(hess, expected, rtol=1e-7, atol=1e-9)
