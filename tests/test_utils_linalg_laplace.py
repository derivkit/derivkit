"""Unit tests for derivkit.utils.linalg_laplace module."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.utils.linalg_laplace import make_spd_by_jitter, symmetrize_matrix


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
