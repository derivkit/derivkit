"""Unit tests for adaptive polynomial fitting derivative estimation."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative


def f_lin(x: float) -> float:
    """A simple linear function with slope 2 and intercept 1."""
    return 2.0 * x + 1.0


def f_quad(x: float) -> float:
    """A simple quadratic function with second derivative 6."""
    return 3.0 * x**2 + 2.0 * x + 1.0  # f'' = 6


def f_cubic(x: float) -> float:
    """A simple cubic function with third derivative 24."""
    return 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0  # f''' = 24


def f_quartic(x: float) -> float:
    """A simple quartic function with fourth derivative 120."""
    return 5.0 * x**4 + 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0  # f'''' = 120


def f_sin(x: float) -> float:
    """Sine function."""
    return np.sin(x)


def f_exp(x: float) -> float:
    """Exponential function."""
    return np.exp(x)


def f_logistic(x: float) -> float:
    """Logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-x))


def f_vec(x: float) -> np.ndarray:
    """Vector-valued function: [x^2, sin(x), exp(x)]."""
    return np.array([x**2, np.sin(x), np.exp(x)], dtype=float)


KW = dict(
    n_points=15,
    spacing="auto",
    direction="both",
    base_abs=5e-4,
)

KW_POLY0 = {**KW, "base_abs": 5e-3}

# ---------- correctness on simple truths ----------


@pytest.mark.parametrize("x0", [0.0, 0.5, -1.2])
def test_linear_first_derivative(x0):
    """Linear function should yield exact first derivative 2.0 everywhere."""
    d = AdaptiveFitDerivative(f_lin, x0)
    got = d.differentiate(order=1, **KW)
    assert np.isclose(got, 2.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_quadratic_second_derivative(x0):
    """Quadratic function should yield exact second derivative 6.0 everywhere."""
    d = AdaptiveFitDerivative(f_quad, x0)
    got = d.differentiate(order=2, **KW)
    assert np.isclose(got, 6.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_cubic_third_derivative(x0):
    """Cubic function should yield exact third derivative 24.0 everywhere."""
    d = AdaptiveFitDerivative(f_cubic, x0)
    kw = KW if x0 != 0.0 else KW_POLY0
    got = d.differentiate(order=3, **kw)
    assert np.isclose(got, 24.0, rtol=0, atol=1e-8)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_quartic_fourth_derivative(x0):
    """Quartic function should yield exact fourth derivative 120.0 everywhere."""
    d = AdaptiveFitDerivative(f_quartic, x0)
    kw = KW if x0 != 0.0 else KW_POLY0
    got = d.differentiate(order=4, **kw)
    assert np.isclose(got, 120.0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("x0", [0.0, 0.3, 1.0])
def test_sin_first_derivative(x0):
    """Sine function should yield first derivative cos(x0)."""
    d = AdaptiveFitDerivative(f_sin, x0)
    got = d.differentiate(order=1, **KW)
    want = np.cos(x0)
    assert np.isclose(got, want, rtol=5e-9, atol=1e-10)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_exp_second_derivative(x0):
    """Exponential function should yield second derivative exp(x0)."""
    d = AdaptiveFitDerivative(f_exp, x0)
    got = d.differentiate(order=2, **KW)
    want = np.exp(x0)
    assert np.isclose(got, want, rtol=5e-9, atol=1e-10)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_logistic_first_derivative(x0):
    """Logistic function should yield first derivative s*(1-s) where s=f(x0)."""
    d = AdaptiveFitDerivative(f_logistic, x0)
    got = d.differentiate(order=1, **KW)
    s = 1.0 / (1.0 + np.exp(-x0))
    want = s * (1.0 - s)
    assert np.isclose(got, want, rtol=5e-9, atol=1e-10)


@pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
def test_vector_second_derivative(x0):
    """Vector function should yield exact second derivatives [2, -sin(x0), exp(x0)]."""
    d = AdaptiveFitDerivative(f_vec, x0)
    got = d.differentiate(order=2, **KW)
    want = np.array([2.0, -np.sin(x0), np.exp(x0)], dtype=float)
    assert got.shape == (3,)
    assert np.allclose(got, want, rtol=5e-9, atol=1e-10)


def test_diagnostics_payload_and_center_omitted():
    """Diagnostics dict should contain expected keys and be center-free."""
    d = AdaptiveFitDerivative(f_quad, 0.3)
    val, diag = d.differentiate(order=2, diagnostics=True, **KW)
    assert np.isfinite(val)

    for key in ("x", "t", "u", "scale_s", "y", "degree"):
        assert key in diag

    t = np.asarray(diag["t"])
    # no exact 0 in the offsets
    assert not np.any(np.isclose(t, 0.0))
    # strictly center-free: smallest |t| must be > 0
    min_abs = float(np.min(np.abs(t)))
    assert min_abs > 0.0


def test_raises_when_too_few_points():
    """Should raise ValueError when too few points for the order."""
    d = AdaptiveFitDerivative(f_quad, 0.0)
    # need_min = max(5, m+2); for m=3, min is 5 → 4 should raise, 5 should pass
    with pytest.raises(ValueError):
        d.differentiate(
            order=3,
            n_points=4,
            spacing="auto",
            direction="both",
            base_abs=1e-3,
        )
    val = d.differentiate(
        order=3, n_points=5, spacing="auto", direction="both", base_abs=1e-3
    )
    assert np.isfinite(val)


@pytest.mark.parametrize("direction", ["both", "pos", "neg"])
def test_direction_modes_run(direction):
    """Test that all direction modes run without error."""
    d = AdaptiveFitDerivative(f_sin, 0.7)
    val = d.differentiate(
        order=1,
        n_points=17,
        spacing="auto",
        direction=direction,
        base_abs=1e-3,
    )
    assert np.isfinite(val)
