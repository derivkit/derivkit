"""Focused tests for the finite-difference backend."""

from functools import partial

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def quad(x, a=2.0, b=-3.0, c=1.5):
    """Quadratic function for testing."""
    return a * x**2 + b * x + c


def vecfunc(x):
    """Vector output function for testing."""
    return np.array([x**2, 2 * x])


def test_stencil_matches_analytic():
    """Tests that finite differences match the analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1, method="finite"
    )
    assert np.isclose(result, exact, rtol=1e-2)


def test_invalid_order_finite():
    """Tests that unsupported derivative order raises ValueError."""
    with pytest.raises(ValueError):
        # orders must be positive; 0 is invalid
        DerivativeKit(lambda x: x, 1.0).differentiate(
            order=0, method="finite", num_points=5
        )


def test_fd_second_derivative_quadratic_constant():
    """Tests that second derivative of a quadratic is constant: d²/dx² (ax²+bx+c) = 2a."""
    a, b, c = 3.0, -1.0, 2.0
    f = partial(quad, a=a, b=b, c=c)
    est = DerivativeKit(f, x0=0.3).differentiate(order=2, method="finite", num_points=5)
    assert np.isclose(est, 2 * a, rtol=1e-3, atol=1e-8)


def test_vector_output_returns_1d_array():
    """Tests that multi-component output returns 1D NumPy array of derivatives."""
    est = DerivativeKit(vecfunc, x0=0.5).differentiate(
        order=1, method="finite", num_points=5
    )
    assert isinstance(est, np.ndarray)
    assert est.shape == (2,)


def test_invalid_stencil_size_raises():
    """Tests that unsupported stencil size raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 0.0).differentiate(
            order=1, method="finite", num_points=4
        )  # not in [3,5,7,9]


def test_invalid_combo_stencil_order_raises():
    """Tests that unsupported (stencil size, order) combination raises ValueError."""
    # 3-point supports only order=1
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 0.0).differentiate(
            order=2, method="finite", num_points=3
        )


def test_scalar_returns_python_float():
    """Tests that scalar output returns Python float, not 1-element array."""
    val = DerivativeKit(lambda x: x**2, 1.0).differentiate(
        order=1, method="finite", num_points=5
    )
    assert isinstance(val, float)
