"""Unit tests for the gradient function in derivkit.forecasting.calculus."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.forecasting.calculus import build_gradient


def model_linear_1d(params):
    """A simple linear function in one dimension with known gradient."""
    x = np.asarray(params, float).reshape(-1)
    slope = 2.75
    intercept = -1.0
    return slope * x[0] + intercept


def grad_linear_1d(params):
    """Gradient of the simple linear function in one dimension."""
    _ = params
    return np.array([2.75], dtype=float)


def model_quad_2d(params):
    """A simple quadratic function in two dimensions with known gradient."""
    x = np.asarray(params, float).reshape(-1)
    matrix = np.array([[4.0, 1.0],
                       [1.0, 3.0]], dtype=float)
    offset = np.array([-1.0, 2.5], dtype=float)
    constant = 0.3
    return 0.5 * x @ matrix @ x + offset @ x + constant


def grad_quad_2d(params):
    """Gradient of the simple quadratic function in two dimensions."""
    x = np.asarray(params, float).reshape(-1)
    matrix = np.array([[4.0, 1.0],
                       [1.0, 3.0]], dtype=float)
    offset = np.array([-1.0, 2.5], dtype=float)
    return matrix @ x + offset


def model_smooth_2d(params):
    """A smooth nonlinear function in two dimensions with known gradient."""
    x = np.asarray(params, float).reshape(-1)
    return np.sin(x[0]) + np.exp(x[1]) + x[0] * x[1]


def grad_smooth_2d(params):
    """Gradient of the smooth nonlinear function in two dimensions."""
    x = np.asarray(params, float).reshape(-1)
    return np.array([np.cos(x[0]) + x[1], np.exp(x[1]) + x[0]], dtype=float)


def model_vector_output(params):
    """A model that returns a vector output instead of a scalar."""
    x = np.asarray(params, float).reshape(-1)
    return np.array([x.sum(), x.sum() ** 2], dtype=float)


def model_scalar_from_size(params):
    """A model that returns a scalar based on the size of the input."""
    x = np.asarray(params, float).reshape(-1)
    return float(x.size)


def model_nan(_params):
    """A model that returns NaN to test error handling."""
    return np.nan


def test_gradient_linear_one_dimension():
    """Test gradient computation for a simple linear function in one dimension."""
    point = [0.123]
    result = build_gradient(model_linear_1d, point)
    expected = grad_linear_1d(point)
    assert result.shape == (1,)
    assert_allclose(result, expected, rtol=2e-8, atol=0)


def test_gradient_quadratic_two_dimensions_workers_1():
    """Test gradient computation for a simple quadratic function in two dimensions."""
    point = np.array([0.2, -0.4], dtype=float)
    result = build_gradient(model_quad_2d, point, n_workers=1)
    expected = grad_quad_2d(point)
    assert result.shape == (2,)
    assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("worker_count", [1, 2, 4])
def test_gradient_is_consistent_across_worker_counts(worker_count):
    """Test that gradient results are consistent across different worker counts."""
    point = np.array([0.5, 0.25], dtype=float)
    result = build_gradient(model_quad_2d, point, n_workers=worker_count)
    expected = grad_quad_2d(point)
    assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_gradient_smooth_nonlinear_two_dimensions():
    """Test gradient computation for a smooth nonlinear function in two dimensions."""
    point = np.array([0.3, -0.5], dtype=float)
    result = build_gradient(model_smooth_2d, point)
    expected = grad_smooth_2d(point)
    assert result.shape == (2,)
    assert_allclose(result, expected, rtol=2e-6, atol=2e-8)


def test_gradient_rejects_non_scalar_output():
    """Test that gradient raises TypeError for non-scalar model outputs."""
    with pytest.raises(TypeError):
        build_gradient(model_vector_output, np.array([1.0, 2.0]))


def test_gradient_raises_for_empty_parameters():
    """Test that gradient raises ValueError for empty parameter array."""
    with pytest.raises(ValueError):
        build_gradient(model_scalar_from_size, np.array([]))


def test_gradient_raises_for_nonfinite_model_value():
    """Test that gradient raises FloatingPointError for NaN model output."""
    with pytest.raises(FloatingPointError):
        build_gradient(model_nan, np.array([1.0, 2.0]))


def sin_sum(x):
    """A test function: sum of sin(x) + 0.1 * x^2 over all elements of x."""
    return float(np.sum(np.sin(x) + 0.1 * x**2))


def test_build_gradient_parallel_equals_serial():
    """Test that parallel and serial gradient computations yield the same result."""
    t = np.array([0.1, -0.3, 0.7, 1.2])
    g1 = build_gradient(sin_sum, t, n_workers=1)
    g4 = build_gradient(sin_sum, t, n_workers=4)
    np.testing.assert_allclose(g4, g1, rtol=1e-8, atol=1e-10)
