"""Unit tests for batch evaluation of functions."""

from __future__ import annotations

import numpy as np
import pytest

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
    y = eval_function_batch(square_scalar, xs)
    assert y.shape == (5, 1)
    assert y.dtype == float


def test_vector_output_consistent_shape():
    """Vector outputs become (n_points, n_comp) float array."""
    xs = np.array([0.0, 0.5, 1.0], dtype=float)
    y = eval_function_batch(pair_linear_quadratic, xs)
    assert y.shape == (3, 2)


def test_mismatch_raises():
    """Inconsistent output shapes raise ValueError."""
    xs = np.array([0.0, 0.6], dtype=float)
    raised = False
    try:
        eval_function_batch(bad_branching_shape, xs)
    except ValueError:
        raised = True
    assert raised


def test_parallel_consistency_scalar(extra_threads_ok):
    """Results should match regardless of n_workers for scalar outputs."""
    if not extra_threads_ok:
        pytest.skip("cannot spawn extra threads here")
    xs = np.linspace(-2.0, 2.0, 101)
    y1 = eval_function_batch(square_scalar, xs, n_workers=1)
    y4 = eval_function_batch(square_scalar, xs, n_workers=4)
    assert y1.shape == y4.shape
    assert np.allclose(y1, y4, atol=0.0, rtol=0.0)


def test_parallel_consistency_vector(extra_threads_ok):
    """Results should match regardless of n_workers for vector outputs."""
    if not extra_threads_ok:
        pytest.skip("cannot spawn extra threads here")
    xs = np.linspace(-1.0, 1.0, 51)
    y1 = eval_function_batch(pair_linear_quadratic, xs, n_workers=1)
    y3 = eval_function_batch(pair_linear_quadratic, xs, n_workers=3)
    assert y1.shape == y3.shape == (xs.size, 2)
    assert np.allclose(y1, y3, atol=0.0, rtol=0.0)
