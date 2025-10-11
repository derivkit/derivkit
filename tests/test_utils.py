"""Lightweight tests for derivkit.utils."""

import tempfile

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
    """Test central_difference_error_estimate raises ValueError for invalid order."""
    with pytest.raises(ValueError):
        central_difference_error_estimate(0.1, order=5)


def test_is_symmetric_grid_true_odd():
    """Test is_symmetric_grid returns True for symmetric odd-length grid."""
    x_odd = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert is_symmetric_grid(x_odd) is True


def test_is_symmetric_grid_false_asymmetric():
    """Test that is_symmetric_grid returns False for a deliberately asymmetric grid.

    The last point is offset by 1e-3, which is many orders of magnitude
    above the floating-point tolerance (~1e-12). This ensures the test
    fails due to real asymmetry, not rounding error.
    """
    y = np.array([-3.0, -1.0, 1.0, 3.001])
    assert is_symmetric_grid(y) is False


def test_generate_test_function_sin_tuple_and_values():
    """Test generate_test_function returns correct sin function and derivatives."""
    f, df, d2f = generate_test_function("sin")
    x = np.pi / 3.0
    assert np.isclose(f(x), np.sin(x))
    assert np.isclose(df(x), np.cos(x))
    assert np.isclose(d2f(x), -np.sin(x))


def test_generate_test_function_raises_on_unknown():
    """Test generate_test_function raises ValueError on unknown function name."""
    with pytest.raises(ValueError):
        generate_test_function("unknown")


def test_get_partial_function_handles_length_4_vector_and_preserves_length():
    """Verify get_partial_function preserves length and varies only the chosen index (4-vector)."""

    def full(params: np.ndarray):
        # Return length and a simple derived value to verify indices unchanged.
        return np.array([len(params), params[0] + params[-1]])

    fixed = np.array([1.0, 2.0, 3.0, 4.0])
    g = get_partial_function(
        full_function=full, variable_index=2, fixed_values=fixed
    )

    # Ensure original is not mutated
    assert fixed.tolist() == [1.0, 2.0, 3.0, 4.0]

    out = g(10.0)  # vary index 2 -> 10
    assert out.shape == (2,)
    assert out[0] == 4  # length preserved
    assert np.isclose(out[1], 1.0 + 4.0)  # sum of first and last unchanged
