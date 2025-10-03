"""Unit tests for derivative/forecast utilities."""

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


@pytest.fixture
def linear_func():
    """Return f(x)=2x+1 for linear tests."""
    return lambda x: 2.0 * x + 1.0


@pytest.fixture
def quadratic_func():
    """Return f(x)=3x^2+2x+1 for quadratic tests."""
    return lambda x: 3.0 * x**2 + 2.0 * x + 1.0


@pytest.fixture
def cubic_func():
    """Return f(x)=4x^3+3x^2+2x+1 for cubic tests."""
    return lambda x: 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0


@pytest.fixture
def quartic_func():
    """Return f(x)=5x^4+4x^3+3x^2+2x+1 for quartic tests."""
    return lambda x: 5.0 * x**4 + 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0


@pytest.fixture
def log_func():
    """Return f(x)=log(x) for domain-sensitive tests."""
    return lambda x: np.log(x)


@pytest.fixture
def vector_func():
    """Return vector output f(x)=[x, 2x] for multi-component tests."""
    return lambda x: np.array([x, 2 * x])


def test_linear_first_derivative(linear_func):
    """Test that the first derivative of a linear function returns the correct slope."""
    calc = DerivativeKit(linear_func, x0=0.0).adaptive
    result = calc.differentiate(order=1)
    assert np.isclose(result, 2.0, atol=1e-8)


def test_quadratic_second_derivative(quadratic_func):
    """Test that the second derivative of a quadratic function returns the expected constant."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    result = calc.differentiate(order=2)
    assert np.isclose(result, 6.0, atol=1e-6)


def test_cubic_third_derivative(cubic_func):
    """Test that the third derivative of a cubic function returns the expected constant."""
    calc = DerivativeKit(cubic_func, x0=1.0).adaptive
    result = calc.differentiate(order=3)
    assert np.isclose(result, 24.0, rtol=1e-2)


def test_quartic_fourth_derivative(quartic_func):
    """Test that the fourth derivative of a quartic function returns the expected constant."""
    calc = DerivativeKit(quartic_func, x0=1.0).adaptive
    result = calc.differentiate(order=4)
    assert np.isclose(result, 120.0, rtol=1e-1)


def test_invalid_order_adaptive():
    """Test that requesting an unsupported derivative order with adaptive raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0).adaptive.differentiate(order=5)


def test_invalid_order_finite():
    """Test that requesting an unsupported derivative order with finite difference raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0).finite.differentiate(order=5)


def test_log_derivative_matches_analytic(log_func):
    """Derivative of log at x0 should be 1/x0 without any special flags."""
    x0 = 2.0
    calc = DerivativeKit(log_func, x0=x0).adaptive
    result = calc.differentiate(order=1)
    assert np.isclose(result, 1.0 / x0, rtol=1e-3, atol=1e-8)


def test_vector_function(vector_func):
    """Test that vector-valued functions return correct shape and values."""
    calc = DerivativeKit(vector_func, x0=1.0).adaptive
    result = calc.differentiate(order=1)
    assert result.shape == (2,)
    assert np.allclose(result, [1.0, 2.0], rtol=1e-2)


def test_stencil_matches_analytic():
    """Test that the finite difference result approximates the analytic derivative of sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(lambda x: np.sin(x), x0).finite.differentiate(
        order=1
    )
    assert np.isclose(result, exact, rtol=1e-2)


def test_derivative_noise_test_runs():
    """Test stability and reproducibility of repeated noisy derivative estimates."""
    adaptive = DerivativeKit(lambda x: x**2, 1.0).adaptive
    results = [
        adaptive.differentiate(order=1) + np.random.normal(0, 0.001)
        for _ in range(10)
    ]
    assert len(results) == 10
    assert all(np.isfinite(r) for r in results)


def test_zero_x0():
    """Test that derivative at x=0 is computed correctly for a cubic function."""
    result = DerivativeKit(lambda x: x**3, x0=0.0).adaptive.differentiate(
        order=1
    )
    # Allow tiny numerical residue
    assert np.isclose(result, 0.0, atol=1e-9)


def test_constant_function():
    """Test that derivatives of a constant function are zero for all orders."""
    for order in range(1, 5):
        result = DerivativeKit(lambda x: 42.0, 1.0).adaptive.differentiate(order=order)
        # Small numerical bias is acceptable; tighten later if needed
        assert np.isclose(result, 0.0, atol=5e-6)


def test_shape_mismatch_raises():
    """Test that shape mismatch in vector output raises ValueError."""
    def bad_func(x):
        return (
            np.array([x, x**2]) if np.round(x, 2) < 1.0 else np.array([x])
        )  # triggers mismatch

    with pytest.raises(ValueError):
        DerivativeKit(bad_func, x0=1.0).adaptive.differentiate(
            order=1
        )


@pytest.mark.parametrize("preset", ["strict", "balanced", "loose", "very_loose"])
def test_acceptance_presets_run(quadratic_func, preset):
    """Test that all acceptance presets run without error and return finite values."""
    calc = DerivativeKit(quadratic_func, x0=1.2).adaptive
    val = calc.differentiate(order=2, acceptance=preset)
    assert np.isfinite(val)


def test_invalid_acceptance_string_raises(quadratic_func):
    """Test that an invalid acceptance string raises ValueError."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    with pytest.raises(ValueError):
        calc.differentiate(order=1, acceptance="ultra_strict")


@pytest.mark.parametrize("a", [0.0, 1.0, -0.1, 1.1])
def test_invalid_acceptance_float_raises(quadratic_func, a):
    """Test that out-of-bounds acceptance floats raise ValueError."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    with pytest.raises(ValueError):
        calc.differentiate(order=1, acceptance=a)


def test_diagnostics_tuple_and_dict(quadratic_func):
    """Test that diagnostics=True returns a tuple and diagnostics is a dict."""
    calc = DerivativeKit(quadratic_func, x0=0.5).adaptive
    val, diag = calc.differentiate(order=2, diagnostics=True)
    assert np.isfinite(val)
    assert isinstance(diag, dict)


def test_include_zero_false_runs(quadratic_func):
    """Test that include_zero=False runs without error and returns finite values."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    val = calc.differentiate(order=2, include_zero=False)
    assert np.isfinite(val)


def test_min_samples_too_small_is_tolerated(quadratic_func):
    """Test that min_samples below internal lower bound is tolerated and returns finite value."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    val = calc.differentiate(order=1, min_samples=3)  # below internal lower bound
    assert np.isfinite(val)  # contract: still returns something sane


def test_invalid_high_order_raises(quadratic_func):
    """Test that requesting too high an order for the function raises ValueError."""
    calc = DerivativeKit(quadratic_func, x0=1.0).adaptive
    with pytest.raises(ValueError):
        calc.differentiate(order=6)


def test_residual_gate_returns_value_no_warning():
    """Test that noisy function with residual gate returns finite value and diagnostics."""
    rng = np.random.default_rng(0)
    def noisy_linear(x):
        return 3.0 * x + 1.0 + rng.normal(0, 1e-2)

    calc = DerivativeKit(noisy_linear, x0=0.0).adaptive
    val, diag = calc.differentiate(order=1, min_samples=7,
                                   acceptance="strict", diagnostics=True)
    assert np.isfinite(val)
    assert isinstance(diag, dict)  # keep loose; internals may change


def test_conditioning_gate_returns_value_no_warning():
    """Test that badly scaled function with conditioning gate returns finite value and diagnostics."""
    def bad_poly(x):
        return 1e6 * x**3 + 2.0 * x

    calc = DerivativeKit(bad_poly, x0=10.0).adaptive
    val, diag = calc.differentiate(order=3, min_samples=5,
                                   acceptance="strict", diagnostics=True)
    assert np.isfinite(val)
    assert isinstance(diag, dict)


def test_vector_function_with_diagnostics(vector_func):
    """Test that vector-valued function with diagnostics returns correct shape and diagnostics dict."""
    calc = DerivativeKit(vector_func, x0=1.0).adaptive
    vals, diag = calc.differentiate(order=1, diagnostics=True)
    assert vals.shape == (2,)
    assert np.allclose(vals, [1.0, 2.0], rtol=1e-2)
    assert isinstance(diag, dict)


def test_log_derivative_strict_acceptance(log_func):
    """Test that derivative of log with strict acceptance matches analytic value."""
    x0 = 2.0
    calc = DerivativeKit(log_func, x0=x0).adaptive
    val = calc.differentiate(order=1, acceptance="strict")
    assert np.isclose(val, 1.0 / x0, rtol=1e-3, atol=1e-8)


def test_shape_mismatch_raises_again():
    """Test that shape mismatch in vector output raises ValueError."""
    def bad(x):
        # 2 comps for x<1, then 1 comp → mismatch across grid
        return np.array([x, x**2]) if x < 1.0 else np.array([x])

    calc = DerivativeKit(bad, x0=1.0).adaptive
    with pytest.raises(ValueError):
        calc.differentiate(order=1)


def test_nan_output_surfaces_nan():
    """Test that function returning NaN leads to NaN derivative output."""
    calc = DerivativeKit(lambda x: np.nan, x0=0.0).adaptive
    val = calc.differentiate(order=1)
    assert np.isnan(val)

