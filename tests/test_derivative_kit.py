"""Tests for DerivativeKit method-dispatch API."""

from functools import partial

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
