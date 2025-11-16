"""Tests for derivkit.utils.sandbox."""

import numpy as np
import pytest

from derivkit.utils.sandbox import generate_test_function, get_partial_function


def test_generate_test_function_sin_tuple_and_values():
    """generate_test_function returns correct sin function and derivatives."""
    f, df, d2f = generate_test_function("sin")
    x = np.pi / 3.0
    assert np.isclose(f(x), np.sin(x))
    assert np.isclose(df(x), np.cos(x))
    assert np.isclose(d2f(x), -np.sin(x))


def test_generate_test_function_raises_on_unknown():
    """generate_test_function raises ValueError on unknown function name."""
    with pytest.raises(ValueError):
        generate_test_function("unknown")


def test_get_partial_function_handles_length_4_vector_and_preserves_length():
    """get_partial_function preserves length and varies only the chosen index (4-vector)."""

    def full(params: np.ndarray):
        # Return length and a simple derived value to verify indices unchanged.
        return np.array([len(params), params[0] + params[-1]])

    fixed = np.array([1.0, 2.0, 3.0, 4.0])
    g = get_partial_function(
        full_function=full,
        variable_index=2,
        fixed_values=fixed,
    )

    # Ensure original is not mutated
    assert fixed.tolist() == [1.0, 2.0, 3.0, 4.0]

    out = g(10.0)  # vary index 2 -> 10
    assert out.shape == (2,)
    assert out[0] == 4  # length preserved
    assert np.isclose(out[1], 1.0 + 4.0)  # sum of first and last unchanged
