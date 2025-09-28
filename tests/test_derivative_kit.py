"""Tests for DerivativeKit wrapper."""

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def test_constructor_wires_adaptive_and_finite(monkeypatch):
    """DerivativeKit constructs Adaptive/Finite with identical (function, x0) and stores them."""
    calls = {"adaptive": None, "finite": None}

    class FakeAdaptive:
        def __init__(self, function, x0, **kwargs):  # noqa: D401
            calls["adaptive"] = (function, x0, kwargs)

        def differentiate(self, **kwargs):  # not used here
            return 0.0

    class FakeFinite:
        def __init__(self, function, x0, **kwargs):  # noqa: D401
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

    def f(x):
        return x

    dk = DerivativeKit(f, 0.5)

    assert calls["adaptive"][0] is f and np.isclose(calls["adaptive"][1], 0.5)
    assert calls["finite"][0] is f and np.isclose(calls["finite"][1], 0.5)
    assert isinstance(dk.adaptive, FakeAdaptive)
    assert isinstance(dk.finite, FakeFinite)


def test_get_used_points_plumbs_diagnostics(monkeypatch):
    """get_used_points forwards kwargs and parses diagnostics.

    Forwards (order, n_workers), requests diagnostics=True, and returns parsed
    (x_all, y_all, x_used, y_used, used_mask).
    """
    captured = {"called": False, "kwargs": None}

    class FakeAdaptive:
        def __init__(self, *args, **kwargs):
            pass

        def differentiate(self, *, order, diagnostics, n_workers, **_):
            captured["called"] = True
            captured["kwargs"] = {
                "order": order,
                "diagnostics": diagnostics,
                "n_workers": n_workers,
            }

            # Diagnostics payload matching DerivativeKit expectations
            M, C = 5, 1
            diag = {
                "x_all": np.linspace(0.0, 1.0, M),
                "y_all": np.linspace(10.0, 20.0, M).reshape(M, C),
                "x_used": [np.array([0.0, 0.25, 0.5])],
                "y_used": [np.array([10.0, 12.5, 15.0])],
                "used_mask": [np.array([True, False, True, True, False])],
            }
            return 0.0, diag

    class FakeFinite:
        def __init__(self, *args, **kwargs):
            pass

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

    dk = DerivativeKit(lambda x: x, 0.0)
    x_all, y_all, x_used, y_used, used_mask = dk.get_used_points(
        order=2, n_workers=3
    )

    # Ensure diagnostics=True and kwargs forwarded
    assert captured["called"] is True
    assert captured["kwargs"] == {
        "order": 2,
        "diagnostics": True,
        "n_workers": 3,
    }

    # Shapes & basic value checks
    assert x_all.shape == (5,)
    assert y_all.shape == (5,)
    assert x_used.shape == (3,)
    assert y_used.shape == (3,)
    assert used_mask.shape == (5,)
    np.testing.assert_allclose(x_used, np.array([0.0, 0.25, 0.5]))
    assert used_mask.dtype == bool
