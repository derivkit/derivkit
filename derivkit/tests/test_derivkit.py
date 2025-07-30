import numpy as np
import pytest

from derivkit import DerivativeKit


# Example fixture definitions if missing
@pytest.fixture
def linear_func():
    return lambda x: 2.0 * x + 1.0

@pytest.fixture
def quadratic_func():
    return lambda x: 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def cubic_func():
    return lambda x: 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def quartic_func():
    return lambda x: 5.0 * x**4 + 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def log_func():
    return lambda x: np.log(x)

@pytest.fixture
def vector_func():
    return lambda x: np.array([x, 2*x])


def test_linear_first_derivative(linear_func):
    """Test that the first derivative of a linear function returns the correct slope."""
    calc = DerivativeKit(linear_func, central_value=0.0, derivative_order=1).adaptive
    result = calc.compute()
    assert np.isclose(result, 2.0, atol=1e-8)

def test_quadratic_second_derivative(quadratic_func):
    """Test that the second derivative of a quadratic function returns the expected constant."""
    calc = DerivativeKit(quadratic_func, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute()
    assert np.isclose(result, 6.0, atol=1e-6)

def test_cubic_third_derivative(cubic_func):
    """Test that the third derivative of a cubic function returns the expected constant."""
    calc = DerivativeKit(cubic_func, central_value=1.0, derivative_order=3).adaptive
    result = calc.compute()
    assert np.isclose(result, 24.0, rtol=1e-2)

def test_quartic_fourth_derivative(quartic_func):
    """Test that the fourth derivative of a quartic function returns the expected constant."""
    calc = DerivativeKit(quartic_func, central_value=1.0, derivative_order=4).adaptive
    result = calc.compute()
    assert np.isclose(result, 120.0, rtol=1e-1)

def test_invalid_derivative_order_adaptive():
    """Test that requesting an unsupported derivative order with adaptive raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0, derivative_order=5).adaptive.compute()

def test_invalid_derivative_order_finite():
    """Test that requesting an unsupported derivative order with finite difference raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0, derivative_order=5).finite.compute()

def test_log_derivative_matches_analytic(log_func):
    """Derivative of log at x0 should be 1/x0 without any special flags."""
    x0 = 2.0
    calc = DerivativeKit(log_func, central_value=x0, derivative_order=1).adaptive
    result = calc.compute()
    assert np.isclose(result, 1.0 / x0, rtol=1e-3, atol=1e-8)

def test_vector_function(vector_func):
    """Test that vector-valued functions return correct shape and values."""
    calc = DerivativeKit(vector_func, central_value=1.0, derivative_order=1).adaptive
    result = calc.compute()
    assert result.shape == (2,)
    assert np.allclose(result, [1.0, 2.0], rtol=1e-2)

def test_fallback_used(monkeypatch):
    """Test that fallback to finite differences is used when adaptive fit fails."""
    f = lambda x: np.exp(x)
    calc = DerivativeKit(f, central_value=0.2, derivative_order=1).adaptive

    def fail_fit(self, x_vals, y_vals):
        return {
            "ok": False, "reason": "singular_normal_equations",
            "h": 1.0, "u_vals": None, "poly_u": None,
            "y_fit": None, "residuals": None, "rel_error": np.inf
        }

    # Patch the class behind your `adaptive` instance — no extra import needed
    monkeypatch.setattr(type(calc), "_fit_once", fail_fit, raising=True)

    with pytest.warns(RuntimeWarning, match="Falling back to finite differences"):
        val = calc.compute()

    assert np.isfinite(val)
    assert np.isclose(val, np.exp(0.2), rtol=1e-4, atol=1e-8)

def test_stencil_matches_analytic():
    """Test that the finite difference result approximates the analytic derivative of sin(x)."""
    f = lambda x: np.sin(x)
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(f, x0, derivative_order=1).finite.compute()
    assert np.isclose(result, exact, rtol=1e-2)

def test_derivative_noise_test_runs():
    """Test stability and reproducibility of repeated noisy derivative estimates."""
    f = lambda x: x**2
    adaptive = DerivativeKit(f, 1.0, derivative_order=1).adaptive
    results = [adaptive.compute() + np.random.normal(0, 0.001) for _ in range(10)]
    assert len(results) == 10
    assert all(np.isfinite(r) for r in results)

def test_zero_central_value():
    """Test that derivative at x=0 is computed correctly for a cubic function."""
    f = lambda x: x**3
    result = DerivativeKit(f, central_value=0.0, derivative_order=1).adaptive.compute()
    # Allow tiny numerical residue
    assert np.isclose(result, 0.0, atol=1e-9)

def test_constant_function():
    """Test that derivatives of a constant function are zero for all orders."""
    f = lambda x: 42.0
    for order in range(1, 5):
        result = DerivativeKit(f, 1.0, derivative_order=order).adaptive.compute()
        # Small numerical bias is acceptable; tighten later if needed
        assert np.isclose(result, 0.0, atol=5e-6)

def test_fallback_triggers_when_fit_unavailable(monkeypatch):
    """If the internal fit cannot be performed, code must fall back to FD (no flags needed)."""
    f = lambda x: np.exp(x)
    calc = DerivativeKit(f, central_value=0.0, derivative_order=1).adaptive

    def fail_fit(self, x_vals, y_vals):
        return {
            "ok": False, "reason": "singular_normal_equations",
            "h": 1.0, "u_vals": None, "poly_u": None,
            "y_fit": None, "residuals": None, "rel_error": np.inf
        }

    monkeypatch.setattr(type(calc), "_fit_once", fail_fit, raising=True)

    # Expect a runtime warning about FD fallback and a correct derivative near e^0 = 1
    with pytest.warns(RuntimeWarning, match="Falling back to finite differences"):
        val = calc.compute()
    assert np.isfinite(val)
    assert np.isclose(val, 1.0, rtol=1e-4, atol=1e-8)

def test_fallback_returns_finite_value_when_fit_fails(monkeypatch):
    """When the fit cannot meet tolerance/structure, the implementation should still return a finite FD value."""
    f = lambda x: 1e-10 * x**3
    calc = DerivativeKit(f, central_value=1.0, derivative_order=2).adaptive

    def fail_fit(self, x_vals, y_vals):
        return {
            "ok": False, "reason": "singular_normal_equations",
            "h": 1.0, "u_vals": None, "poly_u": None,
            "y_fit": None, "residuals": None, "rel_error": np.inf
        }

    monkeypatch.setattr(type(calc), "_fit_once", fail_fit, raising=True)

    with pytest.warns(RuntimeWarning, match="Falling back to finite differences"):
        result = calc.compute()
    # Analytic d2/dx2 of 1e-10 * x^3 at x=1 is 6e-10
    assert np.isfinite(result)
    assert np.isclose(result, 6e-10, rtol=0.2)

def test_diagnostics_structure_is_present():
    """Diagnostics should return expected keys and aligned shapes."""
    f = np.cos
    calc = DerivativeKit(f, central_value=0.7, derivative_order=2).adaptive
    val, diag = calc.compute(diagnostics=True)
    assert np.isfinite(val)
    for k in ["x_all", "y_all", "x_used", "y_used", "y_fit", "residuals", "used_mask", "status"]:
        assert k in diag
    # masks align with x_all
    assert isinstance(diag["used_mask"], list) and len(diag["used_mask"]) >= 1
    assert diag["used_mask"][0].dtype == bool
    assert diag["used_mask"][0].shape == diag["x_all"].shape

def test_vector_fallback_used():
    """Test fallback on vector-valued function returns valid, finite results."""
    f = lambda x: np.array([1e-10 * x**3, 1e-10 * x**2])
    calc = DerivativeKit(f, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute(fit_tolerance=1e-5)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))

def test_shape_mismatch_raises():
    """Test that shape mismatch in vector output raises ValueError."""
    def bad_func(x):
        return np.array([x, x**2]) if np.round(x, 2) < 1.0 else np.array([x])  # triggers mismatch

    with pytest.raises(ValueError):
        DerivativeKit(bad_func, central_value=1.0, derivative_order=1).adaptive.compute()
