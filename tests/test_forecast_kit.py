"""Tests for ForecastKit class."""

import numpy as np
import pytest

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
        return np.full((2, 2), 42.0)

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
    fk.fisher()
    fk.dali()

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


def test_init_resolves_covariance_callable_caches_cov0():
    """Tests that ForecastKit accepts cov=cov_fn and caches cov0."""

    def cov_fn(_theta):
        """Returns a mock covariance matrix."""
        return np.eye(3)

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=cov_fn)

    assert fk.cov_fn is cov_fn
    assert fk.cov0.shape == (3, 3)
    np.testing.assert_allclose(fk.cov0, np.eye(3))
    assert fk.n_observables == 3


def test_init_resolves_covariance_tuple_keeps_cov0_and_cov_fn():
    """Tests that ForecastKit accepts cov=(cov0, cov_fn) and caches both."""
    cov0 = 2.0 * np.eye(4)

    def cov_fn(theta):
        """Returns a mock covariance matrix."""
        _ = np.asarray(theta)
        return np.eye(4)

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=(cov0, cov_fn))

    np.testing.assert_allclose(fk.cov0, cov0)
    assert fk.cov_fn is cov_fn
    assert fk.n_observables == 4


def test_fisher_raises_if_function_is_none():
    """Tests that ForecastKit.fisher requires a mean model."""
    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(1))
    with pytest.raises(ValueError, match=r"ForecastKit\.fisher: function must be provided\."):
        fk.fisher()


def test_dali_raises_if_function_is_none():
    """Tests that ForecastKit.dali requires a mean model."""
    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(1))
    with pytest.raises(ValueError, match=r"ForecastKit\.dali: function must be provided\."):
        fk.dali()


def test_fisher_bias_delegates(monkeypatch):
    """Tests that ForecastKit.fisher_bias delegates to build_fisher_bias."""
    seen = {}

    def fake_build_fisher_bias(
        *,
        function,
        theta0,
        cov,
        fisher_matrix,
        delta_nu,
        method=None,
        n_workers=1,
        rcond=1e-12,
        **dk_kwargs,
    ):
        """Mock build_fisher_bias that records inputs and returns fixed outputs."""
        seen["function"] = function
        seen["theta0"] = np.asarray(theta0)
        seen["cov"] = np.asarray(cov)
        seen["fisher_matrix"] = np.asarray(fisher_matrix)
        seen["delta_nu"] = np.asarray(delta_nu)
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["rcond"] = rcond
        seen["dk_kwargs"] = dk_kwargs
        return np.array([0.1, 0.2]), np.array([-0.01, 0.03])

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_bias",
        fake_build_fisher_bias,
        raising=True,
    )

    def model(theta):
        """Mock mean model function."""
        return np.asarray(theta)

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=model, theta0=theta0, cov=np.eye(3))

    fisher_matrix = np.eye(2)
    delta_nu = np.arange(3.0)

    bias_vec, dtheta = fk.fisher_bias(
        fisher_matrix=fisher_matrix,
        delta_nu=delta_nu,
        method="finite",
        n_workers=7,
        rcond=1e-9,
        step=1e-4,
    )

    np.testing.assert_allclose(bias_vec, np.array([0.1, 0.2]))
    np.testing.assert_allclose(dtheta, np.array([-0.01, 0.03]))

    assert seen["function"] is model
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["cov"], np.eye(3))
    np.testing.assert_allclose(seen["fisher_matrix"], fisher_matrix)
    np.testing.assert_allclose(seen["delta_nu"], delta_nu)
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 7
    assert seen["rcond"] == 1e-9
    assert seen["dk_kwargs"]["step"] == 1e-4


def test_delta_nu_delegates(monkeypatch):
    """Tests that ForecastKit.delta_nu delegates to build_delta_nu."""
    seen = {}

    def fake_build_delta_nu(*, cov, data_with, data_without):
        seen["cov"] = np.asarray(cov)
        seen["with"] = np.asarray(data_with)
        seen["without"] = np.asarray(data_without)
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_delta_nu",
        fake_build_delta_nu,
        raising=True,
    )

    fk = ForecastKit(function=lambda t: np.asarray(t), theta0=np.array([0.0]), cov=np.eye(3))
    out = fk.delta_nu(data_unbiased=np.zeros(3), data_biased=np.ones(3))

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(seen["cov"], np.eye(3))
    np.testing.assert_allclose(seen["with"], np.ones(3))
    np.testing.assert_allclose(seen["without"], np.zeros(3))


def test_generalized_fisher_raises_without_cov_fn_when_needed():
    """Tests that ForecastKit.generalized_fisher raises ValueError if cov_fn is needed but missing."""
    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(2))

    with pytest.raises(ValueError):
        fk.generalized_gaussian_fisher(term="cov")
    with pytest.raises(ValueError):
        fk.generalized_gaussian_fisher(term="both")


def test_generalized_fisher_delegates_with_cov_fn(monkeypatch):
    """Tests that ForecastKit.generalized_fisher delegates to build_generalized_fisher_matrix."""
    seen = {}

    def fake_build_generalized_fisher_matrix(
        *,
        theta0,
        cov,
        function,
        term="both",
        method=None,
        n_workers=1,
        rcond=1e-12,
        symmetrize_dcov=True,
        **dk_kwargs,
    ):
        """Mock build_generalized_fisher_matrix that records inputs and returns fixed output."""
        seen["theta0"] = np.asarray(theta0)
        seen["cov"] = cov
        seen["function"] = function
        seen["term"] = term
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["rcond"] = rcond
        seen["symmetrize_dcov"] = symmetrize_dcov
        seen["dk_kwargs"] = dk_kwargs
        return np.full((2, 2), 9.0)

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_generalized_gaussian_fisher_matrix",
        fake_build_generalized_fisher_matrix,
        raising=True,
    )

    cov0 = 2.0 * np.eye(3)

    def cov_fn(theta):
        """Returns a mock covariance matrix."""
        _ = np.asarray(theta)
        return np.eye(3)

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=(cov0, cov_fn))
    out = fk.generalized_gaussian_fisher(
        term="cov",
        method="finite",
        n_workers=5,
        rcond=1e-8,
        symmetrize_dcov=False,
        step=1e-4,
    )

    np.testing.assert_allclose(out, np.full((2, 2), 9.0))
    assert isinstance(seen["cov"], tuple) and len(seen["cov"]) == 2
    np.testing.assert_allclose(seen["cov"][0], cov0)
    assert seen["cov"][1] is cov_fn
    assert seen["function"] is None
    assert seen["term"] == "cov"
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 5
    assert seen["rcond"] == 1e-8
    assert seen["symmetrize_dcov"] is False
    assert seen["dk_kwargs"]["step"] == 1e-4
