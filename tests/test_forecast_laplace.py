"""Unit tests for derivkit.forecasting.laplace module."""

from __future__ import annotations

import numpy as np
import pytest

import derivkit.calculus_kit as calculus_kit_mod
import derivkit.forecasting.laplace as laplace_mod
from derivkit.forecasting.laplace import (
    laplace_approximation,
    laplace_covariance,
    laplace_hessian,
    negative_logposterior,
)

HESS_RAW_2X2 = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float64)
HESS_RAW_3X3 = np.eye(3, dtype=np.float64)
HESS_RAW_NDIM1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

HESS_NOT_SPD = np.array([[0.0, 0.0], [0.0, -1.0]], dtype=np.float64)
JITTER = 1.25
HESS_SPD = np.array([[JITTER, 0.0], [0.0, JITTER + 1.0]], dtype=np.float64)


def logpost_const(_theta: np.ndarray) -> float:
    """A log-posterior function that returns a constant value."""
    return 3.5


def logpost_nan(_theta: np.ndarray) -> float:
    """A log-posterior function that returns NaN."""
    return np.nan


def nlp_quadratic(theta: np.ndarray) -> float:
    """A negative log-posterior function that is quadratic in theta."""
    return float(theta @ theta)


def nlp_zero(_theta: np.ndarray) -> float:
    """A negative log-posterior function that returns zero."""
    return 0.0


def nlp_inf(_theta: np.ndarray) -> float:
    """A negative log-posterior function that returns infinity."""
    return np.inf


def fake_hessian_returns_h_raw_2x2(self, method=None, n_workers=1, **kwargs):
    """A fake Hessian function that returns a 2x2 raw Hessian matrix."""
    return HESS_RAW_2X2


def fake_hessian_returns_ndim1(self, method=None, n_workers=1, **kwargs):
    """A fake Hessian function that returns a 1-dimensional array."""
    return HESS_RAW_NDIM1


def fake_hessian_returns_3x3(self, method=None, n_workers=1, **kwargs):
    """A fake Hessian function that returns a 3x3 raw Hessian matrix."""
    return HESS_RAW_3X3


def fake_hessian_returns_not_spd(self, method=None, n_workers=1, **kwargs):
    """A fake Hessian function that returns a non-SPD Hessian matrix."""
    return HESS_NOT_SPD


def fake_make_spd_by_jitter(_h: np.ndarray):
    """A fake function that makes a matrix SPD by adding jitter."""
    return HESS_SPD, JITTER


def test_negative_logposterior_returns_negative_of_logposterior():
    """Tests that negative_logposterior returns the negative of the log-posterior."""
    theta = np.array([1.0, 2.0], dtype=np.float64)
    out = negative_logposterior(theta, logposterior=logpost_const)
    assert isinstance(out, float)
    assert out == pytest.approx(-3.5)


def test_negative_logposterior_raises_if_nonfinite():
    """Tests that negative_logposterior raises ValueError if log-posterior is non-finite."""
    with pytest.raises(ValueError, match="non-finite"):
        negative_logposterior([0.0, 1.0], logposterior=logpost_nan)


def test_laplace_covariance_inverts_hessian_and_symmetrizes():
    """Tests that laplace_covariance inverts and symmetrizes the Hessian matrix."""
    h = np.array([[2.0, 0.1], [0.0, 3.0]], dtype=np.float64)  # intentionally not symmetric
    cov = laplace_covariance(h)

    assert np.allclose(cov, cov.T)

    h_sym = 0.5 * (h + h.T)
    cov_expected = np.linalg.inv(h_sym)
    assert np.allclose(cov, cov_expected)


def test_laplace_hessian_symmetrizes_and_validates_shape(monkeypatch):
    """Tests that laplace_hessian symmetrizes the Hessian and validates its shape."""
    monkeypatch.setattr(
        calculus_kit_mod.CalculusKit, "hessian", fake_hessian_returns_h_raw_2x2, raising=True
    )

    h = laplace_hessian(neg_logposterior=nlp_quadratic, theta_map=[0.1, -0.2])

    assert np.allclose(h, h.T)
    assert np.allclose(h, 0.5 * (HESS_RAW_2X2 + HESS_RAW_2X2.T))


def test_laplace_hessian_raises_if_hessian_ndim_not_2(monkeypatch):
    """Tests that laplace_hessian raises TypeError if Hessian is not 2-dimensional."""
    monkeypatch.setattr(
        calculus_kit_mod.CalculusKit, "hessian", fake_hessian_returns_ndim1, raising=True
    )

    with pytest.raises(TypeError, match="requires a scalar negative log-posterior"):
        laplace_hessian(neg_logposterior=nlp_zero, theta_map=[0.0, 0.0])


def test_laplace_hessian_raises_if_shape_mismatch(monkeypatch):
    """Tests that laplace_hessian raises ValueError if Hessian shape mismatches theta_map."""
    monkeypatch.setattr(
        calculus_kit_mod.CalculusKit, "hessian", fake_hessian_returns_3x3, raising=True
    )

    with pytest.raises(ValueError, match=r"Hessian must have shape"):
        laplace_hessian(neg_logposterior=nlp_zero, theta_map=[0.0, 0.0])


def test_laplace_approximation_returns_expected_keys_and_jitter(monkeypatch):
    """Tests that laplace_approximation returns expected keys and jitter value."""
    monkeypatch.setattr(
        calculus_kit_mod.CalculusKit, "hessian", fake_hessian_returns_not_spd, raising=True
    )
    monkeypatch.setattr(
        laplace_mod, "make_spd_by_jitter", fake_make_spd_by_jitter, raising=True
    )

    theta_map = np.array([0.1, -0.2], dtype=np.float64)
    out = laplace_approximation(neg_logposterior=nlp_quadratic, theta_map=theta_map, ensure_spd=True)

    assert np.allclose(out["hessian"], HESS_SPD)
    assert out["jitter"] == pytest.approx(JITTER)


def test_laplace_approximation_raises_if_neg_logposterior_at_map_nonfinite():
    """Tests that laplace_approximation raises ValueError if neg_logposterior at theta_map is non-finite."""
    with pytest.raises(ValueError, match="non-finite value at theta_map"):
        laplace_approximation(neg_logposterior=nlp_inf, theta_map=[0.0, 0.0])
