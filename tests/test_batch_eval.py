"""Unit tests for batch evaluation of functions."""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.batch_eval import eval_function_batch


def square_scalar(x: float) -> float:
    """Return x^2 as a scalar float."""
    return float(x**2)


def pair_linear_quadratic(x: float) -> np.ndarray:
    """Return a 2-vector [x, x^2]."""
    return np.array([x, x**2], dtype=float)


def bad_branching_shape(x: float) -> np.ndarray:
    """Return inconsistent shapes depending on x (used to trigger a ValueError)."""
    return (
        np.array([x, x**2], dtype=float)
        if x < 0.5
        else np.array([x], dtype=float)
    )


def test_scalar_output_shape_and_dtype():
    """Scalar outputs become (n_points, 1) float array."""
    xs = np.linspace(-1, 1, 5)
    Y = eval_function_batch(square_scalar, xs)
    assert Y.shape == (5, 1)
    assert Y.dtype == float


def test_vector_output_consistent_shape():
    """Vector outputs become (n_points, n_comp) float array."""
    xs = np.array([0.0, 0.5, 1.0], dtype=float)
    Y = eval_function_batch(pair_linear_quadratic, xs)
    assert Y.shape == (3, 2)


def test_mismatch_raises():
    """Inconsistent output shapes raise ValueError."""
    xs = np.array([0.0, 0.6], dtype=float)
    raised = False
    try:
        eval_function_batch(bad_branching_shape, xs)
    except ValueError:
        raised = True
    assert raised
