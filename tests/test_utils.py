"""Lightweight tests for derivkit.utils."""

import tempfile
from collections.abc import Callable

import numpy as np
import pytest

from derivkit.utils import (
    central_difference_error_estimate,
    generate_test_function,
    get_partial_function,
    is_finite_and_differentiable,
    is_symmetric_grid,
    log_debug_message,
    normalize_derivative,
)


def test_log_debug_message_prints_when_debug_true(capsys):
    """Test that log_debug_message prints to stdout when debug is True."""
    log_debug_message("hello utils", debug=True)
    out = capsys.readouterr().out
    assert "hello utils" in out


def test_log_debug_message_writes_to_file():
    """Test that log_debug_message writes to a file when log_to_file is True."""
    with open(tempfile.NamedTemporaryFile().name, "w+") as f:
        log_debug_message("line1", log_to_file=True, log_file=f)
        for line in f:
          assert line == "line1"


def test_is_finite_and_differentiable_true_on_vector_function():
    """Test that is_finite_and_differentiable returns True for a well-behaved vector function."""
    def f(x: float):
        return np.array([x, x**2, np.sin(x)])

    assert is_finite_and_differentiable(f, 0.1, delta=1e-4)


def test_is_finite_and_differentiable_false_on_nan():
    """Test that is_finite_and_differentiable returns False when function returns NaN."""
    def f(_: float):
        return np.nan

    assert not is_finite_and_differentiable(f, 0.0)


def test_normalize_derivative_basic_and_zero_ref():
    """Test normalize_derivative with basic inputs and zero reference."""
    deriv = np.array([2.0, -2.0, 0.0])
    ref = np.array([1.0, -1.0, 0.0])

    out = normalize_derivative(deriv, ref)
    eps = 1e-12
    expected = (deriv - ref) / (np.abs(ref) + eps)

    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("order", "factor"),
    [
        (1, 1.0 / 6.0),
        (2, 1.0 / 12.0),
        (3, 1.0 / 20.0),
        (4, 1.0 / 30.0),
    ],
)
def test_central_difference_error_estimate(order: int, factor: float):
    """Test central_difference_error_estimate for orders 1-4."""
    h = 1e-2
    est = central_difference_error_estimate(h, order=order)
    assert np.isclose(est, factor * h * h)


def test_central_difference_error_estimate_invalid_order():
    """Test that central_difference_error_estimate raises ValueError for invalid order."""
    with pytest.raises(ValueError):
        _ = central_difference_error_estimate(0.1, order=5)


def test_is_symmetric_grid_true_odd():
    """Test is_symmetric_grid returns True for symmetric odd-length grid."""
    x_odd = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert is_symmetric_grid(x_odd) is True

def test_is_symmetric_grid_false_asymmetric():
    """Test is_symmetric_grid returns False for asymmetric grid."""
    y = np.array([-3.0, -1.0, 1.0, 3.001])
    assert is_symmetric_grid(y) is False


def test_is_symmetric_grid_false_when_asymmetric():
    """Test is_symmetric_grid returns False for nearly symmetric grid."""
    y = np.array([-2.0, -1.0, 0.0, 1.0, 2.001])
    assert is_symmetric_grid(y) is False


def test_generate_test_function_sin_tuple_and_values():
    """Test generate_test_function returns correct sin function and derivatives."""
    f, df, d2f = generate_test_function("sin")
    x = np.pi / 3.0
    assert np.isclose(f(x), np.sin(x))
    assert np.isclose(df(x), np.cos(x))
    assert np.isclose(d2f(x), -np.sin(x))


def test_generate_test_function_raises_on_unknown():
    """Test generate_test_function raises ValueError on unknown name."""
    with pytest.raises(ValueError):
        _ = generate_test_function("unknown")


def test_get_partial_function_varies_only_selected_and_keeps_shape():
    """Test get_partial_function varies only the selected variable and keeps output shape."""
    def full(params: list[float] | np.ndarray):
        a, b, c = params
        # return vector (length 2) to verify atleast_1d behavior downstream
        return np.array([a + 2.0 * b + 3.0 * c, a * b])

    fixed = [1.0, 2.0, 3.0]
    g: Callable[[float], np.ndarray] = get_partial_function(
        full_function=full, variable_index=1, fixed_values=fixed
    )

    # original fixed values must not be mutated (deepcopy inside)
    assert fixed == [1.0, 2.0, 3.0]

    out = g(10.0)  # vary b -> 10
    np.testing.assert_allclose(out, np.array([1.0 + 2.0 * 10.0 + 3.0 * 3.0, 1.0 * 10.0]))
    assert out.ndim == 1 and out.shape == (2,)
