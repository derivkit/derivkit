"""Tests for derivkit.utils.linalg."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.linalg import (
    as_1d_data_vector,
    invert_covariance,
    make_spd_by_jitter,
    normalize_covariance,
    solve_or_pinv,
    split_xy_covariance,
    symmetrize_matrix,
)


def _stack_xy_cov(cxx: np.ndarray, cxy: np.ndarray, cyy: np.ndarray) -> np.ndarray:
    """Builds a full covariance for the stacked data vector [x, y] from (Cxx, Cxy, Cyy)."""
    cxx = np.asarray(cxx, dtype=np.float64)
    cxy = np.asarray(cxy, dtype=np.float64)
    cyy = np.asarray(cyy, dtype=np.float64)
    return np.block([[cxx, cxy], [cxy.T, cyy]]).astype(np.float64, copy=False)


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


def test_invert_covariance_singular_uses_pseudoinverse_and_warns(caplog):
    """Tests that ingular covariance should fall back to pseudoinverse with a warning."""
    cov = np.array([[1.0, 0.0], [0.0, 0.0]])  # rank-deficient

    inv = invert_covariance(cov, warn_prefix="test")
    assert len(caplog.records) > 0

    for record in caplog.records:
        assert record.levelname == "WARNING"

    assert any("pseudoinverse" in str(w.message) for w in caplog.records)

    expected = np.linalg.pinv(cov)
    assert_allclose(inv, expected)


def test_invert_covariance_warns_on_asymmetry(caplog):
    """Tests that non-symmetric covariance should trigger a symmetry warning."""
    cov = np.array([[1.0, 2.0], [0.0, 1.0]])

    _ = invert_covariance(cov, warn_prefix="linalg")
    assert any("WARNING" == record.levelname for record in caplog.records)
    assert any("not symmetric" in str(record.message) for record in caplog.records)


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


def test_solve_or_pinv_rank_deficient_warns_and_uses_pinv(caplog):
    """Tests that rank-deficient matrix should warn and fall back to pseudoinverse."""
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])  # rank 1
    vector = np.array([1.0, 2.0])

    x = solve_or_pinv(matrix, vector, warn_context="test_solve")

    assert len(caplog.records) > 0

    for record in caplog.records:
        assert record.levelname == "WARNING"

    assert any("rank-deficient" in str(w.message) for w in caplog.records)

    expected = np.linalg.pinv(matrix) @ vector
    assert_allclose(x, expected)


def test_solve_or_pinv_cholesky_failure_warns_and_uses_pinv(caplog):
    """Tests that indefinite matrix with assume_symmetric=True should trigger fallback."""
    # Symmetric, full rank but indefinite, so Cholesky fails
    matrix = np.array([[0.0, -1.0], [-1.0, 0.0]])
    vector = np.array([1.0, 2.0])

    x = solve_or_pinv(matrix, vector, assume_symmetric=True, warn_context="chol_fail")

    for record in caplog.records:
        assert record.levelname == "WARNING"

    assert any("not SPD or was singular" in str(w.message) for w in caplog.records)

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


def test_invert_covariance_warns_once_on_asymmetry(caplog):
    """Tests that non-symmetric covariance should trigger a single symmetry warning."""
    cov = np.array([[1.0, 2.0], [0.0, 1.0]])

    inv = invert_covariance(cov, warn_prefix="linalg")
    assert inv.shape == (2, 2)
    assert np.all(np.isfinite(inv))
    assert len([x for x in caplog.records
        if "not symmetric" in x.message
        and "WARNING" == x.levelname
    ]) == 1


def test_symmetrize_matrix_symmetrizes_and_casts_float64():
    """Tests that symmetrize_matrix correctly symmetrizes a matrix and casts to float64."""
    a = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    out = symmetrize_matrix(a)

    assert out.dtype == np.float64
    assert out.shape == (2, 2)
    assert np.allclose(out, out.T)

    expected = 0.5 * (a + a.T)
    assert np.allclose(out, expected.astype(np.float64))


def test_symmetrize_matrix_raises_on_non_square_or_non_2d():
    """Tests that symmetrize_matrix raises ValueError on non-square or non-2D inputs."""
    with pytest.raises(ValueError, match=r"matrix must be square 2D"):
        symmetrize_matrix([1.0, 2.0, 3.0])  # ndim=1

    with pytest.raises(ValueError, match=r"matrix must be square 2D"):
        symmetrize_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3


def test_make_spd_by_jitter_returns_zero_jitter_when_already_spd():
    """Tests that make_spd_by_jitter returns zero jitter for an already SPD matrix."""
    h = np.array([[2.0, 0.2], [0.2, 1.5]], dtype=np.float64)

    h_spd, jitter = make_spd_by_jitter(h)

    assert jitter == pytest.approx(0.0)
    assert np.allclose(h_spd, 0.5 * (h + h.T))
    np.linalg.cholesky(h_spd)


def test_make_spd_by_jitter_adds_positive_jitter_for_indefinite_matrix():
    """Tests that make_spd_by_jitter adds positive jitter for an indefinite matrix."""
    h = np.array([[0.0, 0.0], [0.0, -1.0]], dtype=np.float64)

    h_spd, jitter = make_spd_by_jitter(h)

    assert jitter > 0.0
    assert np.allclose(h_spd, h_spd.T)

    np.linalg.cholesky(h_spd)


def test_make_spd_by_jitter_handles_nonfinite_or_zero_diag_mean_by_falling_back_to_1():
    """Tests that make_spd_by_jitter falls back to 1.0 when diag mean is non-finite or zero."""
    h = np.array([[0.0, 0.0], [0.0, -1.0]], dtype=np.float64)

    jitter_scale = 1e-6
    jitter_floor = 1e-8
    base_expected = jitter_scale * 1.0 + jitter_floor

    h_spd, jitter = make_spd_by_jitter(
        h, jitter_scale=jitter_scale, jitter_floor=jitter_floor, max_tries=12
    )

    assert jitter > 0.0
    assert jitter >= base_expected
    assert np.allclose(h_spd, h_spd.T)
    np.linalg.cholesky(h_spd)


def test_make_spd_by_jitter_raises_if_cannot_make_spd(monkeypatch):
    """Tests that make_spd_by_jitter raises LinAlgError if it cannot make the matrix SPD."""
    def always_fail_cholesky(_a):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "cholesky", always_fail_cholesky, raising=True)

    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    with pytest.raises(np.linalg.LinAlgError, match=r"could not be regularized"):
        make_spd_by_jitter(h, max_tries=3)


def test_as_1d_data_vector_scalar_returns_length_1_float64():
    """Tests that as_1d_data_vector converts a scalar into a length-1 float64 array."""
    out = as_1d_data_vector(3.5)
    assert out.shape == (1,)
    assert out.dtype == np.float64
    assert_allclose(out, np.array([3.5], dtype=np.float64))


def test_as_1d_data_vector_1d_returns_same_values():
    """Tests that as_1d_data_vector returns a 1D array unchanged in shape and values."""
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = as_1d_data_vector(y)
    assert out.shape == (3,)
    assert_allclose(out, y)


def test_as_1d_data_vector_flattens_higher_rank_c_order():
    """Tests that as_1d_data_vector flattens higher-rank arrays in row-major (C) order."""
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = as_1d_data_vector(y)
    assert out.shape == (4,)
    assert_allclose(out, y.ravel(order="C"))


def test_split_xy_covariance_splits_blocks_and_checks_cross_consistency():
    """Tests that split_xy_covariance returns (Cxx, Cxy, Cyy) with correct shapes and consistent cross-blocks."""
    cxx = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.float64)
    cxy = np.array([[0.1], [-0.2]], dtype=np.float64)
    cyy = np.array([[4.0]], dtype=np.float64)
    cov = _stack_xy_cov(cxx, cxy, cyy)

    out_cxx, out_cxy, out_cyy = split_xy_covariance(cov, nx=2)

    assert out_cxx.shape == (2, 2)
    assert out_cxy.shape == (2, 1)
    assert out_cyy.shape == (1, 1)
    assert_allclose(out_cxx, cxx)
    assert_allclose(out_cxy, cxy)
    assert_allclose(out_cyy, cyy)


def test_split_xy_covariance_raises_on_nonfinite_entries():
    """Tests that split_xy_covariance raises ValueError when cov contains non-finite values."""
    cov = np.eye(3, dtype=np.float64)
    cov[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite"):
        split_xy_covariance(cov, nx=1)


def test_split_xy_covariance_raises_on_asymmetric_covariance():
    """Tests that split_xy_covariance raises ValueError when cov is not symmetric within tolerance."""
    cov = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"symmetric"):
        split_xy_covariance(cov, nx=1)


def test_split_xy_covariance_raises_on_bad_nx_range():
    """Tests that split_xy_covariance raises ValueError when nx does not satisfy 0 < nx < n."""
    cov = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match=r"nx must satisfy"):
        split_xy_covariance(cov, nx=0)
    with pytest.raises(ValueError, match=r"nx must satisfy"):
        split_xy_covariance(cov, nx=3)


def test_split_xy_covariance_mapping_order_not_xy_raises():
    """Tests that split_xy_covariance raises ValueError when a mapping specifies an unsupported order."""
    cov = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match=r"Only order='xy'"):
        split_xy_covariance({"cov": cov, "order": "yx"}, nx=1)


def test_split_xy_covariance_mapping_missing_one_index_array_raises():
    """Tests that split_xy_covariance raises ValueError when only one of x_idx/y_idx is provided."""
    cov = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match=r"both 'x_idx' and 'y_idx'"):
        split_xy_covariance({"cov": cov, "x_idx": np.array([0], dtype=int)}, nx=1)


def test_split_xy_covariance_mapping_reorders_then_splits():
    """Tests that split_xy_covariance reorders a covariance via x_idx/y_idx before splitting into blocks."""
    cxx = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.float64)
    cxy = np.array([[0.1], [-0.2]], dtype=np.float64)
    cyy = np.array([[4.0]], dtype=np.float64)
    cov_xy = _stack_xy_cov(cxx, cxy, cyy)

    # Permute to [y, x0, x1] using idx [2,0,1] (since cov_xy is [x0,x1,y])
    idx = np.array([2, 0, 1], dtype=np.int64)
    cov_yx = cov_xy[np.ix_(idx, idx)]

    spec = {
        "cov": cov_yx,
        "x_idx": np.array([1, 2], dtype=np.int64),
        "y_idx": np.array([0], dtype=np.int64),
        "order": "xy",
    }

    out_cxx, out_cxy, out_cyy = split_xy_covariance(spec, nx=2)
    assert_allclose(out_cxx, cxx)
    assert_allclose(out_cxy, cxy)
    assert_allclose(out_cyy, cyy)


def test_split_xy_covariance_mapping_reorder_invalid_partition_raises():
    """Tests that split_xy_covariance raises ValueError when x_idx/y_idx do not partition indices exactly once."""
    cov = np.eye(3, dtype=np.float64)

    # overlap (duplicate 1) and missing 2 -> not a partition
    spec = {
        "cov": cov,
        "x_idx": np.array([0, 1], dtype=np.int64),
        "y_idx": np.array([1], dtype=np.int64),
    }

    with pytest.raises(ValueError, match=r"disjoint|cover all indices|partition"):
        split_xy_covariance(spec, nx=2)
