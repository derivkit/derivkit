"""Tests for derivkit.utils.validate."""

import numpy as np
import pytest

from derivkit.utils.validate import (
    is_finite_and_differentiable,
    validate_covariance_matrix_shape,
    validate_symmetric_psd,
)


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


def test_validate_covariance_matrix_shape_accepts_scalar_1d_2d():
    """Tests that scalar, 1D, and square 2D arrays are accepted."""
    validate_covariance_matrix_shape(2.0)
    validate_covariance_matrix_shape([1.0, 2.0, 3.0])
    out = validate_covariance_matrix_shape(np.eye(3))
    assert out.shape == (3, 3)


def test_validate_covariance_matrix_shape_rejects_nonsquare_2d():
    """Tests that non-square 2D arrays are rejected."""
    with pytest.raises(ValueError, match="square"):
        validate_covariance_matrix_shape(np.zeros((2, 3)))


def test_validate_symmetric_psd_accepts_psd_matrix():
    """Tests that a symmetric positive semi-definite matrix is accepted."""
    a = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    out = validate_symmetric_psd(a)
    assert out.shape == (2, 2)


def test_validate_symmetric_psd_rejects_asymmetric_matrix():
    """Tests that an asymmetric matrix is rejected."""
    a = np.array([[1.0, 2.0],
                  [0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_symmetric_psd(a, sym_atol=1e-12)


def test_validate_symmetric_psd_rejects_indefinite_matrix():
    """Tests that a non-positive-symmetric-definite matrix is rejected."""
    a = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    with pytest.raises(ValueError, match="not positive semi-definite|min eigenvalue"):
        validate_symmetric_psd(a, psd_atol=1e-12)


def test_validate_symmetric_psd_accepts_nearly_symmetric_matrix():
    """Tests that a nearly symmetric matrix within tolerance is accepted."""
    a = np.array([[2.0, 0.5 + 1e-13],
                  [0.5, 1.0]])
    # asymmetry ~1e-13 < sym_atol
    out = validate_symmetric_psd(a, sym_atol=1e-12)
    assert out.shape == (2, 2)
