"""Tests for derivkit.utils.validate."""

import numpy as np

from derivkit.utils.validate import is_finite_and_differentiable


def test_is_finite_and_differentiable_true_on_vector_function():
    """Returns True for a well-behaved vector function."""

    def f(x: float):
        return np.array([x, x**2, np.sin(x)])

    assert is_finite_and_differentiable(f, 0.1, delta=1e-4)


def test_is_finite_and_differentiable_false_on_nan():
    """Returns False when function returns NaN."""

    def f(_: float):
        return np.nan

    assert not is_finite_and_differentiable(f, 0.0)


