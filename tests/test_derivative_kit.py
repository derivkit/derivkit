"""Tests for DerivativeKit method-dispatch API."""

from functools import partial
import pytest

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def quad(x, a=2.0, b=-3.0, c=1.5):
    """A quadratic function: f(x) = a*x^2 + b*x + c."""
    return a * x**2 + b * x + c


def _setup_fakes(monkeypatch):
    """Sets up fake derivative classes and patches DerivativeKit to use them."""
    calls = {"adaptive": None, "finite": None}
    invoked = {"adaptive": None, "finite": None}

    class FakeAdaptive:
        def __init__(self, function, x0, **kwargs):
            calls["adaptive"] = (function, x0, kwargs)

        def differentiate(self, **kwargs):
            invoked["adaptive"] = kwargs
            return 0.0

    class FakeFinite:
        def __init__(self, function, x0, **kwargs):
            calls["finite"] = (function, x0, kwargs)

        def differentiate(self, **kwargs):
            invoked["finite"] = kwargs
            return 0.0

    # Patch the resolver used by DerivativeKit so it returns our fakes,
    # regardless of how the internal mapping/dict is implemented.
    def fake_resolve(name: str):
        key = (name or "").lower()
        if key in {"adaptive", "ad"}:
            return FakeAdaptive
        if key in {"finite", "fd"}:
            return FakeFinite
        raise ValueError(f"Unknown method: {name}")

    monkeypatch.setattr("derivkit.derivative_kit._resolve", fake_resolve, raising=True)
    return calls, invoked


def test_adaptive_dispatch(monkeypatch):
    """Tests that DerivativeKit dispatches to AdaptiveFitDerivative (fake) and forwards kwargs."""
    calls, invoked = _setup_fakes(monkeypatch)
    f = partial(quad, a=2.0, b=-3.0, c=1.5)
    x0 = 0.5

    dk = DerivativeKit(f, x0)
    out = dk.differentiate(order=1, method="adaptive", n_workers=3, tol=1e-6)

    assert out == 0.0

    assert calls["adaptive"] is not None
    f_called, x_called, ctor_kwargs = calls["adaptive"]
    assert f_called is f and np.isclose(x_called, x0)
    assert isinstance(ctor_kwargs, dict)

    assert invoked["adaptive"] is not None
    kwargs_ad = invoked["adaptive"]
    assert kwargs_ad.get("order") == 1
    assert kwargs_ad.get("n_workers") == 3
    assert kwargs_ad.get("tol") == 1e-6

    assert calls["finite"] is None
    assert invoked["finite"] is None


def test_finite_dispatch(monkeypatch):
    """Tests that DerivativeKit dispatches to FiniteDifferenceDerivative (fake) and forwards kwargs."""
    calls, invoked = _setup_fakes(monkeypatch)
    f = partial(quad, a=2.0, b=-3.0, c=1.5)
    x0 = 0.5

    dk = DerivativeKit(f, x0)
    out = dk.differentiate(order=2, method="finite", step=1e-3)

    assert out == 0.0

    assert calls["finite"] is not None
    f_called, x_called, ctor_kwargs = calls["finite"]
    assert f_called is f and np.isclose(x_called, x0)
    assert isinstance(ctor_kwargs, dict)

    assert invoked["finite"] is not None
    kwargs_fd = invoked["finite"]
    assert kwargs_fd.get("order") == 2
    assert kwargs_fd.get("step") == 1e-3

    assert calls["adaptive"] is None
    assert invoked["adaptive"] is None


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
    seen_x0 = []

    class FakeEngine:
        def __init__(self, function, x0, **kwargs):
            # record scalar x0 passed to each engine instance
            seen_x0.append(x0)
            self.function = function
            self.x0 = x0

        def differentiate(self, **kwargs):
            # return a simple vector that depends only on x0
            return np.array([self.x0, 2 * self.x0])

    def fake_resolve(name: str):
        # ignore method name; always use FakeEngine
        return FakeEngine

    monkeypatch.setattr("derivkit.derivative_kit._resolve", fake_resolve, raising=True)

    dk = DerivativeKit(quad, x0=x0_vals)
    out = dk.differentiate(order=1)

    # Engine must have been called once per x0, with scalar x0 each time
    np.testing.assert_allclose(np.array(seen_x0), x0_vals)

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

    class FakeEngine:
        def __init__(self, function, x0, **kwargs):
            self.x0 = x0

        def differentiate(self, **kwargs):
            # pretend output is a length-3 vector per point
            return np.array([self.x0, self.x0 + 1.0, self.x0 + 2.0])

    def fake_resolve(name: str):
        return FakeEngine

    monkeypatch.setattr("derivkit.derivative_kit._resolve", fake_resolve, raising=True)

    dk = DerivativeKit(quad, x0=x0_grid)
    out = dk.differentiate(order=1)

    # shape: x0.shape + (3,)
    assert out.shape == (2, 2, 3)

    # spot-check a couple of entries
    np.testing.assert_allclose(out[0, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(out[1, 1], np.array([1.5, 2.5, 3.5]))
