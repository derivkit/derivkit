"""Tests for ForecastKit class."""
import numpy as np

from derivkit.forecast_kit import ForecastKit


def test_forecastkit_delegates(monkeypatch):
    """Test that ForecastKit delegates to LikelihoodExpansion correctly."""
    calls = {"ctor": None, "fisher": None, "dali": None}

    class FakeLX:
        def __init__(self, function, theta0, cov):
            # capture constructor args once
            calls["ctor"] = (function, np.asarray(theta0), np.asarray(cov))

        def get_forecast_tensors(self, *, forecast_order, n_workers=1):
            if forecast_order == 1:
                calls["fisher"] = n_workers
                return np.full((2, 2), 42.0)  # sentinel Fisher
            if forecast_order == 2:
                calls["dali"] = n_workers
                G = np.zeros((2, 2, 2))
                H = np.ones((2, 2, 2, 2))
                return G, H
            raise AssertionError("Unexpected forecast_order")

    # Patch the class that ForecastKit uses internally (module-local import)
    monkeypatch.setattr("derivkit.forecast_kit.LikelihoodExpansion", FakeLX, raising=True)

    # inputs
    def model(theta):
        return np.asarray(theta)  # any callable

    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    # constructor wiring captured
    ctor_fn, ctor_theta0, ctor_cov = calls["ctor"]
    assert ctor_fn is model
    np.testing.assert_allclose(ctor_theta0, theta0)
    np.testing.assert_allclose(ctor_cov, cov)

    # fisher() -> forecast_order=1, forwards n_workers
    F = fk.fisher(n_workers=3)
    assert F.shape == (2, 2)
    assert np.all(F == 42.0)
    assert calls["fisher"] == 3

    # dali() -> forecast_order=2, forwards n_workers
    G, H = fk.dali(n_workers=4)
    assert G.shape == (2, 2, 2)
    assert H.shape == (2, 2, 2, 2)
    assert calls["dali"] == 4


def test_default_n_workers_forwarded(monkeypatch):
    """Test that default n_workers=1 is forwarded to LikelihoodExpansion."""
    seen = {"fisher": None, "dali": None}

    class FakeLX:
        def __init__(self, *a, **k):
            pass

        def get_forecast_tensors(self, *, forecast_order, n_workers=1):
            if forecast_order == 1:
                seen["fisher"] = n_workers
                return np.zeros((1, 1))
            elif forecast_order == 2:
                seen["dali"] = n_workers
                return np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1))
            raise AssertionError("Unexpected forecast_order")

    monkeypatch.setattr("derivkit.forecast_kit.LikelihoodExpansion", FakeLX, raising=True)

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))
    fk.fisher()  # no n_workers arg
    fk.dali()    # no n_workers arg
    assert seen["fisher"] == 1 and seen["dali"] == 1


def test_return_types_match_lx(monkeypatch):
    """Test that return types from ForecastKit match those from LikelihoodExpansion."""
    class FakeLX:
        def __init__(self, *a, **k):
            pass

        def get_forecast_tensors(self, *, forecast_order, n_workers=1):
            if forecast_order == 1:
                return np.array([[123.0]])
            elif forecast_order == 2:
                return np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))
            raise AssertionError("Unexpected forecast_order")

    monkeypatch.setattr("derivkit.forecast_kit.LikelihoodExpansion", FakeLX, raising=True)

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))
    F = fk.fisher()
    assert isinstance(F, np.ndarray)
    G, H = fk.dali()
    assert isinstance(G, np.ndarray) and isinstance(H, np.ndarray)
