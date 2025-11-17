"""Tests for derivkit.utils.linalg."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.linalg import (
    invert_covariance,
    normalize_covariance,
    solve_or_pinv,
)


def test_invert_covariance_scalar():
    """Scalar covariance should invert to 1 / scalar as a (1, 1) array."""
    cov = 2.0
    inv = invert_covariance(cov)

    assert inv.shape == (1, 1)
    assert_allclose(inv, np.array([[0.5]]))


def test_invert_covariance_diagonal_vector():
    """Tests that 1D diagonal covariance vector should become diagonal inverse matrix."""
    diag = np.array([1.0, 4.0, 0.25])
    inv = invert_covariance(diag)

    assert inv.shape == (3, 3)
    assert_allclose(inv, np.diag(1.0 / diag))


def test_invert_covariance_full_matrix():
    """Tests that 2D full covariance matrix is inverted correctly when full rank."""
    cov = np.array([[2.0, 0.0], [0.0, 5.0]])
    inv = invert_covariance(cov)

    expected = np.array([[0.5, 0.0], [0.0, 0.2]])
    assert inv.shape == (2, 2)
    assert_allclose(inv, expected)


def test_invert_covariance_singular_uses_pseudoinverse_and_warns():
    """Tests that ingular covariance should fall back to pseudoinverse with a warning."""
    cov = np.array([[1.0, 0.0], [0.0, 0.0]])  # rank-deficient

    with pytest.warns(RuntimeWarning) as record:
        inv = invert_covariance(cov, warn_prefix="test")

    # Expect at least one warning mentioning pseudoinverse
    assert any("pseudoinverse" in str(w.message) for w in record)

    expected = np.linalg.pinv(cov)
    assert_allclose(inv, expected)


def test_invert_covariance_warns_on_asymmetry():
    """Tests that non-symmetric covariance should trigger a symmetry warning."""
    cov = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.warns(RuntimeWarning) as record:
        _ = invert_covariance(cov, warn_prefix="linalg")

    assert any("not symmetric" in str(w.message) for w in record)


def test_normalize_covariance_scalar():
    """Tests that scalar covariance becomes k x k identity scaled by scalar."""
    cov = 2.5
    k = 3

    out = normalize_covariance(cov, n_parameters=k)

    assert out.shape == (k, k)
    assert_allclose(out, 2.5 * np.eye(k))


def test_normalize_covariance_vector_diag():
    """Tests that 1D covariance vector becomes diagonal covariance matrix."""
    diag = np.array([1.0, 4.0, 9.0])
    out = normalize_covariance(diag, n_parameters=3)

    assert out.shape == (3, 3)
    assert_allclose(out, np.diag(diag))


def test_normalize_covariance_vector_length_mismatch_raises():
    """Tests that length mismatch between cov vector and n_parameters should raise."""
    diag = np.array([1.0, 4.0])
    with pytest.raises(ValueError):
        normalize_covariance(diag, n_parameters=3)


def test_normalize_covariance_full_symmetric_ok():
    """Tests that symmetric full covariance with correct shape is returned unchanged."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    out = normalize_covariance(cov, n_parameters=2)

    assert out.shape == (2, 2)
    assert_allclose(out, cov)


def test_normalize_covariance_full_slightly_asymmetric_is_symmetrized():
    """Tests that small asymmetry below threshold should be symmetrized, not rejected."""
    # Almost symmetric matrix with tiny asymmetry
    eps = 1e-16
    cov = np.array([[1.0, eps], [0.0, 2.0]])
    out = normalize_covariance(cov, n_parameters=2, asym_atol=1e-12)

    expected = 0.5 * (cov + cov.T)
    assert_allclose(out, expected)


def test_normalize_covariance_full_too_asymmetric_raises():
    """Tests that large asymmetry should raise a ValueError."""
    cov = np.array([[1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        normalize_covariance(cov, n_parameters=2)


def test_normalize_covariance_nonfinite_scalar_raises():
    """Tests that non-finite scalar covariance should raise."""
    with pytest.raises(ValueError):
        normalize_covariance(np.nan, n_parameters=1)


def test_normalize_covariance_invalid_ndim_raises():
    """Tests that covariance with ndim not in {0,1,2} should raise."""
    cov = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        normalize_covariance(cov, n_parameters=2)


def test_solve_or_pinv_spd_cholesky_path():
    """Tests that SPD matrix with assume_symmetric=True uses Cholesky and solves correctly."""
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    vector = np.array([2.0, 3.0])

    x = solve_or_pinv(matrix, vector, assume_symmetric=True)

    assert_allclose(x, np.array([1.0, 1.0]))


def test_solve_or_pinv_general_solve_when_not_assuming_symmetric():
    """Tests that when assume_symmetric=False, use np.linalg.solve for full-rank matrices."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    vector = np.array([5.0, 6.0])

    x = solve_or_pinv(matrix, vector, assume_symmetric=False)

    expected = np.linalg.solve(matrix, vector)
    assert_allclose(x, expected)


def test_solve_or_pinv_rank_deficient_warns_and_uses_pinv():
    """Tests that rank-deficient matrix should warn and fall back to pseudoinverse."""
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])  # rank 1
    vector = np.array([1.0, 2.0])

    with pytest.warns(RuntimeWarning) as record:
        x = solve_or_pinv(matrix, vector, warn_context="test_solve")

    assert any("rank-deficient" in str(w.message) for w in record)

    expected = np.linalg.pinv(matrix) @ vector
    assert_allclose(x, expected)


def test_solve_or_pinv_cholesky_failure_warns_and_uses_pinv():
    """Tests that indefinite matrix with assume_symmetric=True should trigger fallback."""
    # Symmetric, full rank but indefinite, so Cholesky fails
    matrix = np.array([[0.0, -1.0], [-1.0, 0.0]])
    vector = np.array([1.0, 2.0])

    with pytest.warns(RuntimeWarning) as record:
        x = solve_or_pinv(matrix, vector, assume_symmetric=True, warn_context="chol_fail")

    assert any("not SPD or was singular" in str(w.message) for w in record)

    expected = np.linalg.pinv(matrix) @ vector
    assert_allclose(x, expected)


def test_solve_or_pinv_invalid_shapes_raise():
    """Tests that non-square matrix or mismatched vector shape should raise ValueError."""
    matrix = np.eye(3)
    bad_matrix = np.ones((2, 3))
    vector = np.ones(3)
    bad_vector = np.ones(4)

    with pytest.raises(ValueError):
        solve_or_pinv(bad_matrix, vector)

    with pytest.raises(ValueError):
        solve_or_pinv(matrix, bad_vector)
