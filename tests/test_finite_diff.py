"""Focused tests for the finite-difference backend."""

from functools import partial

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def quad(x, a=2.0, b=-3.0, c=1.5):
    """Quadratic function for testing."""
    return a * x**2 + b * x + c


def vecfunc(x):
    """Vector output function for testing."""
    return np.array([x**2, 2 * x])


def test_stencil_matches_analytic():
    """Tests that finite differences match the analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1, method="finite"
    )
    assert np.isclose(result, exact, rtol=1e-2)


def test_invalid_order_finite():
    """Tests that unsupported derivative order raises ValueError."""
    with pytest.raises(ValueError):
        # orders must be positive; 0 is invalid
        DerivativeKit(lambda x: x, 1.0).differentiate(
            order=0, method="finite", num_points=5
        )


def test_fd_second_derivative_quadratic_constant():
    """Tests that second derivative of a quadratic is constant: d²/dx² (ax²+bx+c) = 2a."""
    a, b, c = 3.0, -1.0, 2.0
    f = partial(quad, a=a, b=b, c=c)
    est = DerivativeKit(f, x0=0.3).differentiate(order=2, method="finite", num_points=5)
    assert np.isclose(est, 2 * a, rtol=1e-3, atol=1e-8)


def test_vector_output_returns_1d_array():
    """Tests that multi-component output returns 1D NumPy array of derivatives."""
    est = DerivativeKit(vecfunc, x0=0.5).differentiate(
        order=1, method="finite", num_points=5
    )
    assert isinstance(est, np.ndarray)
    assert est.shape == (2,)


def test_invalid_stencil_size_raises():
    """Tests that unsupported stencil size raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 0.0).differentiate(
            order=1, method="finite", num_points=4
        )  # not in [3,5,7,9]


def test_invalid_combo_stencil_order_raises():
    """Tests that unsupported (stencil size, order) combination raises ValueError."""
    # 3-point supports only order=1
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 0.0).differentiate(
            order=2, method="finite", num_points=3
        )


def test_scalar_returns_python_float():
    """Tests that scalar output returns Python float, not 1-element array."""
    val = DerivativeKit(lambda x: x**2, 1.0).differentiate(
        order=1, method="finite", num_points=5
    )
    assert isinstance(val, float)


def test_finite_richardson_extrapolation_matches_analytic():
    """Tests that finite + Richardson extrapolation matches analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,  # fixed-level Richardson
    )

    assert np.isclose(est, exact, rtol=1e-4, atol=1e-8)


def test_finite_ridders_extrapolation_matches_analytic():
    """Tests that finite + Ridders extrapolation matches analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="ridders",
        levels=4,  # fixed-level Ridders
    )

    assert np.isclose(est, exact, rtol=1e-4, atol=1e-8)


def test_finite_gauss_richardson_extrapolation_matches_analytic():
    """Tests that finite + Gauss–Richardson extrapolation matches analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="gauss-richardson",
        levels=4,  # fixed-level GRE
    )

    assert np.isclose(est, exact, rtol=1e-3, atol=1e-8)


def test_finite_gre_alias_matches_analytic():
    """Tests that the 'gre' alias for Gauss–Richardson extrapolation works."""
    x0 = 0.3
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="gre",
        levels=4,
    )

    assert np.isclose(est, exact, rtol=1e-3, atol=1e-8)


def test_finite_richardson_adaptive_matches_analytic():
    """Tests that finite + Richardson extrapolation (adaptive) matches analytic derivative for sin(x)."""
    x0 = np.pi / 6
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",  # adaptive: no levels
    )

    assert np.isclose(est, exact, rtol=1e-4, atol=1e-8)


def test_finite_ridders_adaptive_matches_analytic():
    """Tests that finite + Ridders extrapolation (adaptive) matches analytic derivative for sin(x)."""
    x0 = 0.5
    exact = np.cos(x0)

    est = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="ridders",  # adaptive: no levels
    )

    assert np.isclose(est, exact, rtol=1e-4, atol=1e-8)


def test_finite_gauss_richardson_vector_output():
    """Tests that finite + Gauss–Richardson extrapolation works for vector-valued functions.

    We compare GRE against the baseline finite-difference estimate to ensure:
      * vector shapes are preserved, and
      * GRE stays numerically close to the non-extrapolated finite result.
    """
    x0 = 0.3
    dk = DerivativeKit(vecfunc, x0)

    # Baseline finite-difference derivative (no extrapolation)
    base = dk.differentiate(
        order=1,
        method="finite",
        num_points=5,
    )

    # Gauss–Richardson extrapolated derivative
    gre = dk.differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="gauss-richardson",
        levels=4,
    )

    assert isinstance(gre, np.ndarray)
    assert gre.shape == base.shape
    # GRE should be close to the baseline finite result
    assert np.allclose(gre, base, rtol=1e-3, atol=1e-6)


def test_finite_extrapolation_returns_error_scalar():
    """Tests that finite + Ridders extrapolation returns a sensible scalar error estimate."""
    x0 = 0.7
    exact = np.cos(x0)

    val, err = DerivativeKit(lambda x: np.sin(x), x0).differentiate(
        order=1,
        method="finite",
        stepsize=1e-2,
        num_points=5,
        extrapolation="ridders",
        levels=4,
        return_error=True,
    )

    assert np.isclose(val, exact, rtol=1e-4, atol=1e-8)
    assert isinstance(err, float)
    assert err >= 0.0
    # error should be small compared to the magnitude of the derivative
    assert err < 1e-2


def test_finite_extrapolation_invalid_scheme_raises():
    """Tests that an unknown extrapolation scheme raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x**2, 1.0).differentiate(
            order=1,
            method="finite",
            stepsize=1e-2,
            num_points=5,
            extrapolation="not-a-scheme",
        )
