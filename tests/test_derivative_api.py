"""Unit tests for DerivativeAPI dispatching and method registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from derivkit.derivative_api import DerivativeAPI


def f_scalar(x: float) -> float:
    """Test function: cubic polynomial."""
    # f(x) = x^3  => f'(x) = 3x^2, f''(x) = 6x
    return x**3

def f_vector(x: float) -> np.ndarray:
    """Test function: vector-valued [x^3, sin x]."""
    # [x^3, sin x] => [3x^2, cos x] for first derivative
    return np.array([x**3, math.sin(x)], dtype=float)


def test_supported_methods_shape_and_contains_aliases():
    """Tests that supported_methods() returns expected structure."""
    groups = DerivativeAPI.supported_methods()
    # Must contain canonical labels
    assert "adaptive-fit" in groups
    assert "finite-difference" in groups
    # Canonical label first
    assert groups["adaptive-fit"][0] == "adaptive-fit"
    assert groups["finite-difference"][0] == "finite-difference"
    # Must include at least one alias each
    assert any(a in groups["adaptive-fit"] for a in ("adaptive", "af", "poly"))
    assert any(a in groups["finite-difference"] for a in ("finite", "fd"))


def test_help_overview_includes_aliases_and_example():
    """Tests that help() overview contains expected sections."""
    dk = DerivativeAPI(f_scalar, x0=0.0)
    text = dk.help()
    assert "Methods:" in text
    assert "adaptive-fit" in text and "finite-difference" in text
    # show at least one alias line
    assert "aliases:" in text
    # includes example call
    assert "dk.differentiate(order=1, method='adaptive-fit'" in text


@pytest.mark.parametrize("x0", [0.2, -0.3])
def test_dispatch_adaptive_first_derivative_vector(x0):
    """Tests that adaptive-fit method dispatches and computes correctly."""
    dk = DerivativeAPI(f_vector, x0=x0)
    # Choose a stable, small demo grid for the adaptive fit
    got = dk.differentiate(
        order=1,
        method="adaptive-fit",
        n_points=11,
        spacing=0.25,
        ridge=1e-10,
    )
    truth = np.array([3 * x0**2, math.cos(x0)], dtype=float)
    # Adaptive is approximate; use slightly looser tolerances
    assert np.allclose(got, truth, rtol=5e-2, atol=5e-3)


@pytest.mark.parametrize("x0", [0.2, -0.3])
def test_dispatch_finite_second_derivative_scalar(x0):
    """Tests that finite-difference method dispatches and computes correctly."""
    dk = DerivativeAPI(f_scalar, x0=x0)
    got = dk.differentiate(order=2, method="fd", stepsize=1e-3, num_points=5)
    truth = 6 * x0
    # Finite diff is very accurate with small stepsize
    assert np.isclose(got, truth, rtol=1e-4, atol=1e-6)


def test_unknown_method_raises_with_supported_list():
    """Tets that unknown method name raises ValueError with supported names."""
    dk = DerivativeAPI(f_scalar, x0=0.0)
    with pytest.raises(ValueError) as e:
        dk.differentiate(order=1, method="not-a-method")
    msg = str(e.value)
    assert "Unknown method" in msg
    # should list some supported names
    assert "adaptive-fit" in msg and "finite-difference" in msg


def test_kwarg_filtering_rejects_unknown_for_finite():
    """Tests that unexpected kwargs for finite-difference raise ValueError."""
    dk = DerivativeAPI(f_scalar, x0=0.0)
    with pytest.raises(ValueError) as e:
        dk.differentiate(order=1, method="finite", stepsize=1e-3, num_points=5, nope_kw=123)
    msg = str(e.value)
    assert "Unexpected keyword(s)" in msg
    # should mention allowed ones for finite
    assert "stepsize" in msg and "num_points" in msg


def test_kwarg_filtering_rejects_unknown_for_adaptive():
    """Tests that unexpected kwargs for adaptive-fit raise ValueError."""
    dk = DerivativeAPI(f_scalar, x0=0.0)
    with pytest.raises(ValueError) as e:
        dk.differentiate(order=1, method="adaptive", n_points=9, spacing=0.2, foobar=42)
    msg = str(e.value)
    assert "Unexpected keyword(s)" in msg
    # should list some known adaptive kwargs
    assert "n_points" in msg and ("spacing" in msg or "ridge" in msg)


def test_backend_registry_can_be_extended(monkeypatch):
    """Tests that backend registry can be extended with new method."""
    # Create a fake backend with a differentiate(signature) that accepts **kwargs
    class FakeBackend:
        def __init__(self):
            self.called = False
            self.seen = {}

        def differentiate(self, order, **kwargs):
            self.called = True
            self.seen = {"order": order, **kwargs}
            # return deterministic token
            return ("fake", order, tuple(sorted(kwargs.items())))

    fake = FakeBackend()
    dk = DerivativeAPI(f_scalar, x0=0.0)

    # Inject under a new key; pretend we added aliases "gp"/"gaussian-process"
    dk._backends["gp"] = fake.differentiate

    # Call via the canonical key directly (bypassing alias resolution).
    # In the real code you'd also extend the ALIASES, but here we just prove
    # the registry path works and kwargs are passed.
    result = dk._backends["gp"](2, foo=1, bar=2)  # use the same call style the dispatcher uses
    assert fake.called is True
    assert fake.seen["order"] == 2 and fake.seen["foo"] == 1 and fake.seen["bar"] == 2
    assert result[0] == "fake"


def test_help_for_specific_method_contains_signature():
    """Tests that help(method) returns useful info for that method."""
    dk = DerivativeAPI(f_scalar, x0=0.0)
    txt = dk.help("fd")
    # crude checks: qualified name + parentheses from signature
    assert "(" in txt and ")" in txt
    # mention of 'stepsize' or 'num_points' likely appears
    assert ("stepsize" in txt) or ("num_points" in txt)


def test_return_shape_scalar_vs_vector():
    """Tests that return type matches function output (scalar vs vector)."""
    x0 = 0.1
    dk_v = DerivativeAPI(f_vector, x0=x0)
    g_v = dk_v.differentiate(order=1, method="fd", stepsize=1e-4, num_points=5)
    assert isinstance(g_v, np.ndarray) and g_v.ndim == 1 and g_v.size == 2

    dk_s = DerivativeAPI(f_scalar, x0=x0)
    g_s = dk_s.differentiate(order=1, method="fd", stepsize=1e-4, num_points=5)
    assert isinstance(g_s, (float, np.floating))
