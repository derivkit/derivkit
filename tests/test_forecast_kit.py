"""Tests for ForecastKit class."""

import numpy as np

from derivkit.forecast_kit import ForecastKit


def test_forecastkit_delegates(monkeypatch):
    """Tests that ForecastKit delegates to fisher/dali helpers correctly."""
    calls = {
        "fisher": None,
        "dali": None,
    }

    def fake_build_fisher_matrix(function, theta0, cov, *, method=None, n_workers=1, **dk_kwargs):
        """Returns a mock Fisher matrix."""
        calls["fisher"] = {
            "function": function,
            "theta0": np.asarray(theta0),
            "cov": np.asarray(cov),
            "method": method,
            "n_workers": n_workers,
            "dk_kwargs": dk_kwargs,
        }
        return np.full((2, 2), 42.0)  # sentinel Fisher

    def fake_build_dali(function, theta0, cov, *, method=None, n_workers=1, **dk_kwargs):
        """Returns mock DALI tensors."""
        calls["dali"] = {
            "function": function,
            "theta0": np.asarray(theta0),
            "cov": np.asarray(cov),
            "method": method,
            "n_workers": n_workers,
            "dk_kwargs": dk_kwargs,
        }
        g_tensor = np.zeros((2, 2, 2))
        h_tensor = np.ones((2, 2, 2, 2))
        return g_tensor, h_tensor

    # Patch the helpers that ForecastKit uses internally
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix", fake_build_fisher_matrix, raising=True
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali", fake_build_dali, raising=True
    )

    # inputs
    def model(theta):
        return np.asarray(theta)  # any callable

    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    # The fisher() method defaults to forecast_order=1 and forwards n_workers.
    # The Fisher computation delegates to the helper function and forwards n_workers.
    fish = fk.fisher(n_workers=3)
    assert fish.shape == (2, 2)
    assert np.all(fish == 42.0)

    assert calls["fisher"] is not None
    np.testing.assert_allclose(calls["fisher"]["theta0"], theta0)
    np.testing.assert_allclose(calls["fisher"]["cov"], cov)
    assert calls["fisher"]["function"] is model
    assert calls["fisher"]["n_workers"] == 3

    # The DALI computation delegates to the helper function and forwards n_workers.
    g_tensor, h_tensor = fk.dali(n_workers=4)
    assert g_tensor.shape == (2, 2, 2)
    assert h_tensor.shape == (2, 2, 2, 2)

    assert calls["dali"] is not None
    np.testing.assert_allclose(calls["dali"]["theta0"], theta0)
    np.testing.assert_allclose(calls["dali"]["cov"], cov)
    assert calls["dali"]["function"] is model
    assert calls["dali"]["n_workers"] == 4


def test_default_n_workers_forwarded(monkeypatch):
    """Tests that default n_workers=1 is forwarded to fisher/dali helpers."""
    n_workers_seen = {"fisher": None, "dali": None}

    def fake_build_fisher_matrix(*args, n_workers=1, **kwargs):
        n_workers_seen["fisher"] = n_workers
        return np.zeros((1, 1))

    def fake_build_dali(*args, n_workers=1, **kwargs):
        n_workers_seen["dali"] = n_workers
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1))

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix", fake_build_fisher_matrix, raising=True
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali", fake_build_dali, raising=True
    )

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))
    fk.fisher()  # no n_workers arg
    fk.dali()    # no n_workers arg

    assert n_workers_seen["fisher"] == 1
    assert n_workers_seen["dali"] == 1


def test_return_types_match_helpers(monkeypatch):
    """Tests that return types from ForecastKit match those from helper functions."""

    def fake_build_fisher_matrix(*args, **kwargs):
        """Returns mock Fisher matrix."""
        return np.array([[123.0]])

    def fake_build_dali(*args, **kwargs):
        """Returns mock DALI tensors."""
        return np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix",
        fake_build_fisher_matrix,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali",
        fake_build_dali,
        raising=True,
    )

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))

    fish = fk.fisher()
    assert isinstance(fish, np.ndarray)

    g_tensor, h_tensor = fk.dali()
    assert isinstance(g_tensor, np.ndarray)
    assert isinstance(h_tensor, np.ndarray)
