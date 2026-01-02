"""Tests for derivkit.utils.validate."""

import numpy as np
import pytest

from derivkit.utils.validate import (
    ensure_finite,
    is_finite_and_differentiable,
    normalize_theta,
    validate_covariance_matrix_shape,
    validate_square_matrix_finite,
    validate_symmetric_psd,
    validate_theta_1d_finite,
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


def test_ensure_finite_accepts_finite_array():
    """Tests that an array with all finite values is accepted."""
    arr = np.array([0.0, 1.0, -3.5])
    ensure_finite(arr, msg="should not fail")


def test_ensure_finite_rejects_nan():
    """Tests that an array with NaN values is rejected."""
    arr = np.array([0.0, np.nan])
    with pytest.raises(FloatingPointError, match="nan"):
        ensure_finite(arr, msg="nan detected")


def test_ensure_finite_rejects_inf():
    """Tests that an array with Inf values is rejected."""
    arr = np.array([1.0, np.inf])
    with pytest.raises(FloatingPointError, match="inf"):
        ensure_finite(arr, msg="inf detected")


def test_normalize_theta_accepts_1d_array():
    """Tests that a 1D array is accepted and returned as a float ndarray."""
    theta = normalize_theta([1.0, 2.0, 3.0])
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (3,)
    assert theta.dtype == float


def test_normalize_theta_flattens_nd_array():
    """Tests that an N-D array is flattened to 1D."""
    theta = normalize_theta([[1.0, 2.0], [3.0, 4.0]])
    assert theta.shape == (4,)
    assert np.all(theta == np.array([1.0, 2.0, 3.0, 4.0]))


def test_normalize_theta_rejects_empty():
    """Tests that an empty array raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        normalize_theta([])


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
