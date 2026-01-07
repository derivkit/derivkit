"""Unit tests for DerivativeKit method-dispatch API."""

from functools import partial

import numpy as np
import pytest

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.derivative_kit import DerivativeKit, _resolve
from derivkit.finite.finite_difference import FiniteDifferenceDerivative
from derivkit.fornberg import FornbergDerivative
from derivkit.local_polynomial_derivative.local_polynomial_derivative import (
    LocalPolynomialDerivative,
)


def quad(x, a=2.0, b=-3.0, c=1.5):
    """A quadratic function: f(x) = a*x^2 + b*x + c."""
    return a * x**2 + b * x + c


_calls_dispatch = {"adaptive": {}, "finite": {}}
_invoked_dispatch = {"adaptive": {}, "finite": {}}


def _reset_dispatch_state() -> None:
    """Resets global dispatch tracking dicts."""
    _calls_dispatch["adaptive"] = {}
    _calls_dispatch["finite"] = {}
    _invoked_dispatch["adaptive"] = {}
    _invoked_dispatch["finite"] = {}


class _FakeAdaptiveEngine:
    """Fake adaptive engine that records constructor args and differentiate kwargs."""

    def __init__(self, function, x0, **kwargs):
        """Records ctor arguments."""
        _calls_dispatch["adaptive"] = {
            "function": function,
            "x0": x0,
            "kwargs": dict(kwargs),
        }

    def differentiate(self, **kwargs):
        """Records differentiate kwargs."""
        _invoked_dispatch["adaptive"] = dict(kwargs)
        return 0.0


class _FakeFiniteEngine:
    """Fake finite-difference engine that records constructor args and differentiate kwargs."""

    def __init__(self, function, x0, **kwargs):
        """Records ctor arguments."""
        _calls_dispatch["finite"] = {
            "function": function,
            "x0": x0,
            "kwargs": dict(kwargs),
        }

    def differentiate(self, **kwargs):
        """Records differentiate kwargs."""
        _invoked_dispatch["finite"] = dict(kwargs)
        return 0.0


def _fake_resolve_dispatch(name: str):
    """Resolver for dispatch tests: returns the appropriate fake engine class."""
    key = (name or "").lower()
    if key in {"adaptive", "ad"}:
        return _FakeAdaptiveEngine
    if key in {"finite", "fd"}:
        return _FakeFiniteEngine
    raise ValueError(f"Unknown method: {name}")


def _setup_fakes(monkeypatch):
    """Patches DerivativeKit resolver to use dispatch fakes and reset their state."""
    _reset_dispatch_state()
    monkeypatch.setattr(
        "derivkit.derivative_kit._resolve",
        _fake_resolve_dispatch,
        raising=True,
    )
    return _calls_dispatch, _invoked_dispatch


class _FakeArrayX0Engine:
    """Fake engine that records scalar x0 values for array-x0 tests."""
    instances = []

    def __init__(self, function, x0, **kwargs):
        """Records the scalar x0 value."""
        self.function = function
        self.x0 = x0
        type(self).instances.append(self)

    def differentiate(self, **kwargs):
        """Returns a fixed 2-vector based on x0."""
        return np.array([self.x0, 2 * self.x0])


def _fake_resolve_array_x0(name: str):
    """Resolver that always returns the array-x0 fake engine."""
    return _FakeArrayX0Engine


class _FakeGridEngine:
    """Fake engine for 2D x0 grids that returns a fixed 3-vector per point."""

    def __init__(self, function, x0, **kwargs):
        """Records the x0 grid."""
        self.x0 = x0

    def differentiate(self, **kwargs):
        """Returns a fixed 3-vector based on x0."""
        return np.array([self.x0, self.x0 + 1.0, self.x0 + 2.0])


def _fake_resolve_grid(name: str):
    """Resolver that always returns the grid fake engine."""
    return _FakeGridEngine


def test_adaptive_dispatch(monkeypatch):
    """Tests that DerivativeKit dispatches to AdaptiveFitDerivative (fake) and forwards kwargs."""
    calls, invoked = _setup_fakes(monkeypatch)
    f = partial(quad, a=2.0, b=-3.0, c=1.5)
    x0 = 0.5

    dk = DerivativeKit(f, x0)
    out = dk.differentiate(order=1, method="adaptive", n_workers=3, tol=1e-6)

    assert out == 0.0

    # constructor was called for adaptive, not for finite
    assert calls["adaptive"]  # non-empty dict
    adaptive_call = calls["adaptive"]
    f_called = adaptive_call["function"]
    x_called = adaptive_call["x0"]
    ctor_kwargs = adaptive_call["kwargs"]

    assert f_called is f
    assert np.isclose(x_called, x0)
    assert isinstance(ctor_kwargs, dict)

    # differentiate was called for adaptive with the right kwargs
    assert invoked["adaptive"]
    kwargs_ad = invoked["adaptive"]
    assert kwargs_ad["order"] == 1
    assert kwargs_ad["n_workers"] == 3
    assert kwargs_ad["tol"] == 1e-6

    # finite should not have been used at all
    assert calls["finite"] == {}
    assert invoked["finite"] == {}


def test_finite_dispatch(monkeypatch):
    """Tests that DerivativeKit dispatches to FiniteDifferenceDerivative (fake) and forwards kwargs."""
    calls, invoked = _setup_fakes(monkeypatch)
    f = partial(quad, a=2.0, b=-3.0, c=1.5)
    x0 = 0.5

    dk = DerivativeKit(f, x0)
    out = dk.differentiate(order=2, method="finite", step=1e-3)

    assert out == 0.0

    # constructor was called for finite, not for adaptive
    assert calls["finite"]
    finite_call = calls["finite"]
    f_called = finite_call["function"]
    x_called = finite_call["x0"]
    ctor_kwargs = finite_call["kwargs"]

    assert f_called is f
    assert np.isclose(x_called, x0)
    assert isinstance(ctor_kwargs, dict)

    # differentiate was called for finite with the right kwargs
    assert invoked["finite"]
    kwargs_fd = invoked["finite"]
    assert kwargs_fd["order"] == 2
    assert kwargs_fd["step"] == 1e-3

    # adaptive should not have been used at all
    assert calls["adaptive"] == {}
    assert invoked["adaptive"] == {}


def test_default_method_is_adaptive():
    """Tests that method=None defaults to adaptive behavior."""
    f = partial(quad, a=1.0, b=0.0, c=0.0)
    dk = DerivativeKit(f, 0.0)

    # method=None should behave exactly like method="adaptive"
    y_none = dk.differentiate(order=1, method=None)
    y_adpt = dk.differentiate(order=1, method="adaptive")

    # robust to scalars/arrays
    if hasattr(y_none, "__array__") or hasattr(y_adpt, "__array__"):
        np.testing.assert_allclose(y_none, y_adpt)
    else:
        assert y_none == y_adpt


def test_array_x0_loops_and_stacks(monkeypatch):
    """Tests that array x0 values are handled correctly by looping and stacking results."""
    x0_vals = np.array([-0.5, 0.0, 0.5])

    # reset class-local state so tests stay isolated
    _FakeArrayX0Engine.instances = []

    monkeypatch.setattr(
        "derivkit.derivative_kit._resolve",
        _fake_resolve_array_x0,
        raising=True,
    )

    dk = DerivativeKit(quad, x0=x0_vals)
    out = dk.differentiate(order=1)

    # Engine must have been called once per x0, with scalar x0 each time
    seen_x0 = np.array([engine.x0 for engine in _FakeArrayX0Engine.instances])
    np.testing.assert_allclose(seen_x0, x0_vals)

    # Output should be stacked with a leading axis over x0
    assert out.shape == (len(x0_vals), 2)
    expected = np.stack([np.array([x, 2 * x]) for x in x0_vals], axis=0)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "method, extra_kwargs",
    [
        # finite: all extrapolation schemes
        ("finite", {}),
        ("finite", {"extrapolation": "richardson"}),
        ("finite", {"extrapolation": "ridders"}),
        ("finite", {"extrapolation": "gauss-richardson"}),
        ("adaptive", {}),
        ("local_polynomial", {}),
    ],
)
def test_array_x0_with_real_engines_and_configs(method, extra_kwargs):
    """Tests that array x0 values work with real engines and configs."""
    f = np.sin
    x0 = np.array([-0.3, 0.0, 0.7])
    dk = DerivativeKit(f, x0)

    out = dk.differentiate(order=1, method=method, **extra_kwargs)
    expected = np.cos(x0)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def test_array_x0_preserves_shape_for_2d_grid(monkeypatch):
    """Tests that 2D array x0 values preserve shape in output."""
    x0_grid = np.array([[0.0, 0.5], [1.0, 1.5]])

    monkeypatch.setattr(
        "derivkit.derivative_kit._resolve",
        _fake_resolve_grid,
        raising=True,
    )

    dk = DerivativeKit(quad, x0=x0_grid)
    out = dk.differentiate(order=1)

    assert out.shape == (2, 2, 3)

    np.testing.assert_allclose(out[0, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(out[1, 1], np.array([1.5, 2.5, 3.5]))


def vec_fun_2d(x):
    """Array-valued function: f(x) = [sin(x), cos(x)]."""
    return np.array([np.sin(x), np.cos(x)])


@pytest.mark.parametrize(
    "method, extra_kwargs",
    [
        ("finite", {}),
        ("finite", {"extrapolation": "richardson"}),
        ("finite", {"extrapolation": "ridders"}),
        ("finite", {"extrapolation": "gauss-richardson"}),
        ("adaptive", {}),
        ("local_polynomial", {}),
    ],
)
def test_array_valued_function_with_array_x0_all_methods(method, extra_kwargs):
    """Tests that an array-valued function is differentiated correctly for array x0 across methods."""
    x0 = np.array([-0.3, 0.0, 0.7])

    dk = DerivativeKit(vec_fun_2d, x0)
    out = dk.differentiate(order=1, method=method, **extra_kwargs)

    expected = np.stack(
        [np.array([np.cos(x), -np.sin(x)]) for x in x0],
        axis=0,
    )

    assert out.shape == expected.shape == (len(x0), 2)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def test_array_valued_function_with_scalar_x0():
    """Tests that an array-valued function works for scalar x0."""
    x0 = 0.3
    dk = DerivativeKit(vec_fun_2d, x0)
    out = dk.differentiate(order=1, method="finite")

    expected = np.array([np.cos(x0), -np.sin(x0)])
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def test_tabulated_mode_dispatches_with_finite_engine(monkeypatch):
    """Tests that tabulated mode dispatches correctly and passes a callable to the engine."""
    calls, invoked = _setup_fakes(monkeypatch)

    x_tab = np.linspace(-1.0, 1.0, 11)
    y_tab = 3.0 * x_tab + 1.0

    # Use tabulated mode: no function, just tab_x/tab_y
    dk = DerivativeKit(x0=0.0, tab_x=x_tab, tab_y=y_tab)
    out = dk.differentiate(method="finite", order=1)

    assert out == 0.0  # fake engine return value

    # Check that the finite fake engine was used and got a callable function
    assert calls["finite"]
    finite_call = calls["finite"]
    f_called = finite_call["function"]
    x_called = finite_call["x0"]

    assert callable(f_called)
    assert np.isclose(x_called, 0.0)

    # Adaptive fake should not have been touched
    assert calls["adaptive"] == {}
    assert invoked["adaptive"] == {}


@pytest.mark.parametrize("method", ["finite", "adaptive", "local_polynomial"])
def test_tabulated_mode_linear_function_with_real_engines(method):
    """Tests that tabulated mode gives the correct derivative for a linear function."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y_tab = 3.0 * x_tab + 1.0  # true derivative is 3 everywhere

    x0 = np.array([-0.5, 0.0, 0.7])
    dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)

    d1 = dk.differentiate(method=method, order=1)
    expected = 3.0 * np.ones_like(x0, dtype=float)

    np.testing.assert_allclose(d1, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "alias, expected_class",
    [
        ("adaptive-fit", AdaptiveFitDerivative),
        ("adapt", AdaptiveFitDerivative),
        ("findiff", FiniteDifferenceDerivative),
        ("finite_diff", FiniteDifferenceDerivative),
        ("localpoly", LocalPolynomialDerivative),
        ("local-poly", LocalPolynomialDerivative),
        ("fb", FornbergDerivative),
        ("forn", FornbergDerivative),
    ],
)
def test_method_aliases_resolve_to_expected_engine(alias, expected_class):
    """Tests that method aliases resolve to the correct engine class."""
    Engine = _resolve(alias)
    assert Engine is expected_class
