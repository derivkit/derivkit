"""Tests for DerivativeKit wrapper."""

from functools import partial

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def quad(x, a=2.0, b=-3.0, c=1.5):
    """Quadratic function for testing."""
    return a * x**2 + b * x + c


def test_constructor_wires_adaptive_and_finite(monkeypatch):
    """Ensure Adaptive and Finite receive the same (function, x0) and are stored.

    Checks:
      * Both constructors are called with identical (f, x0).
      * Call arguments are recorded correctly.
      * Instances are attached as `adaptive` and `finite`.
    """
    calls = {"adaptive": None, "finite": None}

    class FakeAdaptive:
        def __init__(self, function, x0, **kwargs):
            calls["adaptive"] = (function, x0, kwargs)

        def differentiate(self, **kwargs):  # not used here
            return 0.0

    class FakeFinite:
        def __init__(self, function, x0, **kwargs):
            calls["finite"] = (function, x0, kwargs)

    # Patch module-local symbols used by DerivativeKit
    monkeypatch.setattr(
        "derivkit.derivative_kit.AdaptiveFitDerivative",
        FakeAdaptive,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.derivative_kit.FiniteDifferenceDerivative",
        FakeFinite,
        raising=True,
    )

    a, b, c = 2.0, -3.0, 1.5
    f = partial(quad, a=a, b=b, c=c)

    dk = DerivativeKit(f, 0.5)

    assert calls["adaptive"][0] is f and np.isclose(calls["adaptive"][1], 0.5)
    assert calls["finite"][0] is f and np.isclose(calls["finite"][1], 0.5)
    assert isinstance(dk.adaptive, FakeAdaptive)
    assert isinstance(dk.finite, FakeFinite)
