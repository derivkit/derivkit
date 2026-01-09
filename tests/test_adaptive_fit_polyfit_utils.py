"""Unit tests for derivkit.adaptive.polyfit_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivatives.adaptive.polyfit_utils import fit_multi_power


def make_poly_data(coeffs: list[float], t: np.ndarray) -> np.ndarray:
    """Generate polynomial data from coefficients and t values."""
    y = np.zeros_like(t, dtype=float)
    for k, a in enumerate(coeffs):
        y += a * t**k
    return y


def test_scalar_exact_polynomial_fit_rrms_near_zero():
    """Test exact polynomial fit for scalar data with tiny rrms."""
    t = np.linspace(-1.0, 1.0, 25)
    coeffs_true = [1.5, -0.7, 2.0]  # a0 + a1 t + a2 t^2
    y = make_poly_data(coeffs_true, t)
    coeffs, rrms = fit_multi_power(t, y[:, None], deg=2)  # y must be 2D

    # exact fit → tiny rrms
    assert rrms.shape == (1,)
    assert rrms[0] <= 1e-12

    # model reproduces data
    vanderm = np.vander(t, N=3, increasing=True)  # (n, deg+1)
    y_hat = (vanderm @ coeffs)[:, 0]
    assert np.allclose(y_hat, y, rtol=0, atol=1e-12)

    # coefficients match (within tiny numerical error)
    assert np.allclose(coeffs[:, 0], np.array(coeffs_true), rtol=0, atol=1e-12)


def test_scalar_noisy_fit_rrms_between_0_and_1():
    """Test noisy polynomial fit for scalar data with reasonable rrms."""
    rng = np.random.default_rng(1)
    t = np.linspace(-2.0, 2.0, 51)
    y_true = make_poly_data([0.5, 1.0, -0.25], t)
    y = y_true + rng.normal(0, 1e-3, size=t.size)

    _, rrms = fit_multi_power(t, y[:, None], deg=2)
    assert rrms.shape == (1,)
    assert np.isfinite(rrms[0])
    assert 0 <= rrms[0] < 1.0  # small noise → relative RMS < 1


def test_scalar_validation_errors_and_types():
    """Test that invalid inputs raise appropriate errors."""
    t = np.linspace(0, 1, 5)
    y = np.sin(t)[:, None]

    # non-integer degree → np.vander will raise TypeError
    with pytest.raises(TypeError):
        fit_multi_power(t, y, deg=1.5)

    with pytest.raises(ValueError):
        fit_multi_power(t, y, deg=-1)  # negative degree

    with pytest.raises(ValueError):
        fit_multi_power(t[:3], y, deg=1)  # mismatched lengths

    # deg >= n should raise (underdetermined)
    with pytest.raises(ValueError):
        fit_multi_power(t, y, deg=len(t))  # deg == n
    with pytest.raises(ValueError):
        fit_multi_power(t, y, deg=len(t) + 1)  # deg > n


def test_multi_exact_polynomial_coefficients_match_power_basis():
    """Test exact polynomial fit for multiple components with known coefficients."""
    # Two components with different true polynomials in power basis
    t = np.linspace(-1.5, 1.2, 40)
    c0 = [1.0, -2.0, 0.5]  # degree 2
    c1 = [-0.25, 0.0, 1.5, -0.2]  # degree 3

    y0 = make_poly_data(c0, t)
    y1 = make_poly_data(c1, t)
    ys = np.column_stack([y0, y1])  # (n, 2)

    # Fit up to degree 3 to capture the highest-degree component
    coeffs, rrms = fit_multi_power(t, ys, deg=3)

    assert coeffs.shape == (4, 2)
    assert rrms.shape == (2,)
    assert np.all(rrms <= 1e-12)

    # Coefficients match truth (allow tiny numerical noise)
    assert np.allclose(coeffs[: len(c0), 0], np.array(c0), atol=1e-12, rtol=0)
    assert np.allclose(coeffs[: len(c1), 1], np.array(c1), atol=1e-12, rtol=0)
    # Higher unused degree for comp 0 should be ~0
    assert abs(coeffs[3, 0]) <= 1e-12


def test_multi_noisy_rrms_reasonable_and_shapes_ok():
    """Test noisy polynomial fit for multiple components with reasonable rrms."""
    rng = np.random.default_rng(2)
    t = np.linspace(-1.0, 1.0, 60)
    # 3 components with degrees 1,2,3
    c0 = [0.3, -0.7]
    c1 = [1.0, 0.5, -0.25]
    c2 = [0.1, -0.2, 0.3, -0.15]
    y_true = np.column_stack(
        [
            make_poly_data(c0, t),
            make_poly_data(c1, t),
            make_poly_data(c2, t),
        ]
    )
    y = y_true + rng.normal(0, 5e-4, size=y_true.shape)

    coeffs, rrms = fit_multi_power(t, y, deg=3)  # cap at cubic

    assert coeffs.shape == (4, 3)  # (deg+1, n_comp)
    assert rrms.shape == (3,)
    assert np.all(rrms < 1.0)  # noise small vs signal

    # Coefficients close to truth for degrees <= fit degree
    assert np.allclose(coeffs[0:2, 0], np.array(c0), rtol=0, atol=5e-3)
    assert np.allclose(coeffs[0:3, 1], np.array(c1), rtol=0, atol=5e-3)
    assert np.allclose(coeffs[0:4, 2], np.array(c2), rtol=0, atol=5e-3)


def test_multi_bad_shapes_raise():
    """Test that invalid shapes raise appropriate ValueErrors."""
    t = np.linspace(0, 1, 5)

    # Wrong Y shape (1D)
    with pytest.raises(ValueError):
        fit_multi_power(t, np.array([1, 2, 3, 4, 5], dtype=float), deg=1)

    # Mismatched first dimension
    with pytest.raises(ValueError):
        fit_multi_power(t, np.ones((4, 2), float), deg=1)
