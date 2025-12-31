"""Unit tests for derivkit.utils.validate_laplace module."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.utils.validate_laplace import (
    validate_square_matrix_finite,
    validate_theta_1d_finite,
)


def test_validate_theta_1d_finite_accepts_1d_and_casts_float64():
    """Tests that validate_theta_1d_finite accepts valid input and casts to float64."""
    theta = [1, 2, 3]
    out = validate_theta_1d_finite(theta, name="theta")
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape == (3,)
    assert out.dtype == np.float64
    assert np.allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_validate_theta_1d_finite_rejects_non_1d():
    """Tests that validate_theta_1d_finite rejects non-1D input."""
    with pytest.raises(ValueError, match=r"theta must be 1D; got shape"):
        validate_theta_1d_finite([[1.0, 2.0]], name="theta")


def test_validate_theta_1d_finite_rejects_empty():
    """Tests that validate_theta_1d_finite rejects empty input."""
    with pytest.raises(ValueError, match=r"theta must be non-empty"):
        validate_theta_1d_finite([], name="theta")


def test_validate_theta_1d_finite_rejects_nonfinite():
    """Tests that validate_theta_1d_finite rejects non-finite values."""
    with pytest.raises(ValueError, match=r"theta contains non-finite values"):
        validate_theta_1d_finite([0.0, np.nan], name="theta")

    with pytest.raises(ValueError, match=r"theta contains non-finite values"):
        validate_theta_1d_finite([0.0, np.inf], name="theta")


def test_validate_theta_1d_finite_uses_custom_name_in_errors():
    """Tests that validate_theta_1d_finite uses custom name in error messages."""
    with pytest.raises(ValueError, match=r"params must be 1D; got shape"):
        validate_theta_1d_finite([[1.0, 2.0]], name="params")


def test_validate_square_matrix_finite_accepts_square_2d_and_casts_float64():
    """Tests that validate_square_matrix_finite accepts valid input and casts to float64."""
    m = np.array([[1, 2], [3, 4]], dtype=np.int64)
    out = validate_square_matrix_finite(m, name="matrix")
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape == (2, 2)
    assert out.dtype == np.float64
    assert np.allclose(out, m.astype(np.float64))


def test_validate_square_matrix_finite_rejects_non_2d():
    """Tests that validate_square_matrix_finite rejects non-2D input."""
    with pytest.raises(ValueError, match=r"matrix must be 2D; got ndim=1"):
        validate_square_matrix_finite([1.0, 2.0, 3.0], name="matrix")


def test_validate_square_matrix_finite_rejects_non_square():
    """Tests that validate_square_matrix_finite rejects non-square input."""
    with pytest.raises(ValueError, match=r"matrix must be square; got shape"):
        validate_square_matrix_finite([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="matrix")


def test_validate_square_matrix_finite_rejects_nonfinite():
    """Tests that validate_square_matrix_finite rejects non-finite values."""
    with pytest.raises(ValueError, match=r"matrix contains non-finite values"):
        validate_square_matrix_finite([[1.0, np.nan], [0.0, 1.0]], name="matrix")

    with pytest.raises(ValueError, match=r"matrix contains non-finite values"):
        validate_square_matrix_finite([[1.0, np.inf], [0.0, 1.0]], name="matrix")


def test_validate_square_matrix_finite_uses_custom_name_in_errors():
    """Tests that validate_square_matrix_finite uses custom name in error messages."""
    with pytest.raises(ValueError, match=r"H must be 2D; got ndim=1"):
        validate_square_matrix_finite([1.0, 2.0], name="H")
