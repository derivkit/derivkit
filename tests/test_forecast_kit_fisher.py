"""Tests for Fisher matrix construction in forecast_kit/fisher.py."""

import numpy as np
import pytest

from derivkit.forecasting.fisher import build_fisher_matrix


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    _ = theta
    return np.zeros(2, dtype=float)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = theta
    return np.zeros(3, dtype=float)


@pytest.fixture
def forecasting_mocks(monkeypatch):
    """Provides fake get_forecast_tensors with per-test state."""
    class Mocks:
        def __init__(self):
            self.d1 = None
            self.invcov = None
            self.call_info = None

        def set_state(self, d1, invcov):
            self.d1 = np.asarray(d1, dtype=float)
            self.invcov = np.asarray(invcov, dtype=float)

    mocks = Mocks()

    def fake_get_forecast_tensors(
        *,
        function,
        theta0,
        cov,
        forecast_order,
        method,
        n_workers,
        **dk_kwargs,
    ):
        """Takes the place of get_forecast_tensors, returns d1 @ invcov @ d1.T and records call."""
        mocks.call_info = {
            "function": function,
            "theta0": np.asarray(theta0, dtype=float),
            "cov": np.asarray(cov, dtype=float),
            "forecast_order": forecast_order,
            "method": method,
            "n_workers": n_workers,
            "dk_kwargs": dk_kwargs,
        }
        d1 = mocks.d1
        invcov = mocks.invcov
        return d1 @ invcov @ d1.T

    monkeypatch.setattr(
        "derivkit.forecasting.fisher.get_forecast_tensors",
        fake_get_forecast_tensors,
        raising=True,
    )

    return mocks


def test_get_forecast_tensors_order1_matches_matrix_product(forecasting_mocks):
    """Tests that the Fisher matrix equals the derivative matrix @ invcov."""
    # derivative order 1
    d1 = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
        ]
    )
    # these are dummy values; only invcov matters for the test
    cov = np.diag([2.0, 0.5])
    invcov = np.linalg.inv(cov)
    theta0 = np.array([0.0, 0.0])

    forecasting_mocks.set_state(d1=d1, invcov=invcov)

    fisher = build_fisher_matrix(
        function=two_obs_model,
        theta0=theta0,
        cov=cov,
        method="adaptive",
        n_workers=1,
    )

    expected = d1 @ invcov @ d1.T

    assert fisher.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(fisher, expected)


def test_get_forecast_tensors_order1_returns_symmetric_fisher(forecasting_mocks):
    """Tests that the Fisher matrix returned for order=1 is symmetric."""
    rng = np.random.default_rng(123)

    d1 = rng.normal(size=(3, 3))
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3)
    invcov = np.linalg.inv(cov)

    forecasting_mocks.set_state(d1=d1, invcov=invcov)

    fisher = build_fisher_matrix(
        function=three_obs_model,
        theta0=np.zeros(3),
        cov=cov,
    )

    assert fisher.shape == (3, 3)
    np.testing.assert_allclose(fisher, fisher.T)


def test_get_forecast_tensors_order1_builds_fisher(forecasting_mocks):
    """Tests that get_forecast_tensors order=1 builds Fisher matrix correctly."""
    theta0 = np.array([0.1, -0.2])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])

    invcov = np.array(
        [
            [50.0 / 49.0, -5.0 / 49.0],
            [-5.0 / 49.0, 25.0 / 49.0],
        ]
    )

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.5], [-0.3, 2.0]]),
        invcov=invcov,
    )

    fisher = build_fisher_matrix(
        function=two_obs_model,
        theta0=theta0,
        cov=cov,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    d1 = forecasting_mocks.d1
    expected = d1 @ invcov @ d1.T
    assert fisher.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(fisher, expected)

    info = forecasting_mocks.call_info
    dk_kwargs = info["dk_kwargs"]

    assert info["forecast_order"] == 1
    assert info["method"] == "adaptive"
    assert info["n_workers"] == 4
    assert dk_kwargs["step_size"] == 1e-3


def test_get_forecast_tensors_checks_model_output_length():
    """Tests that model output length is checked against covariance shape."""
    cov = np.eye(2)
    theta0 = np.array([0.0])

    # three_obs_model returns 3 observables but cov is 2x2 (n_observables=2),
    # so get_forecast_tensors should complain about the inconsistency.
    with pytest.raises(ValueError):
        build_fisher_matrix(function=three_obs_model, theta0=theta0, cov=cov)


def test_get_forecast_tensors_order1_default_n_workers(forecasting_mocks):
    """Tests that default n_workers=1 is used in _get_derivatives."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    fisher = build_fisher_matrix(
        function=two_obs_model,
        theta0=theta0,
        cov=cov,
    )

    d1 = forecasting_mocks.d1
    invcov = forecasting_mocks.invcov
    expected = d1 @ invcov @ d1.T
    np.testing.assert_allclose(fisher, expected)

    info = forecasting_mocks.call_info
    dk_kwargs = info["dk_kwargs"]

    assert info["forecast_order"] == 1
    assert info["n_workers"] == 1
    assert dk_kwargs == {}


@pytest.mark.parametrize("method", ["adaptive", "finite"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_get_forecast_tensors_order1_forwards_derivative_kwargs(
    forecasting_mocks,
    method,
    extrapolation,
    stencil,
):
    """Tests that derivative method and its kwargs are forwarded to the internal derivative routine."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    fisher = build_fisher_matrix(
        function=two_obs_model,
        theta0=theta0,
        cov=cov,
        method=method,
        n_workers=2,
        extrapolation=extrapolation,
        stencil=stencil,
    )

    d1 = forecasting_mocks.d1
    invcov = forecasting_mocks.invcov
    expected = d1 @ invcov @ d1.T
    np.testing.assert_allclose(fisher, expected)

    info = forecasting_mocks.call_info
    dk_kwargs = info["dk_kwargs"]

    assert info["forecast_order"] == 1
    assert info["method"] == method
    assert info["n_workers"] == 2
    assert dk_kwargs["extrapolation"] == extrapolation
    assert dk_kwargs["stencil"] == stencil


def test_get_forecast_tensors_order1_forwards_local_polyfit_kwargs(forecasting_mocks):
    """Test that local_polyfit method and its kwargs are forwarded to the internal derivative routine."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    fisher = build_fisher_matrix(
        function=two_obs_model,
        theta0=theta0,
        cov=cov,
        method="local_polyfit",
        n_workers=3,
        degree=5,
        window=4,
        trim_fraction=0.2,
    )

    d1 = forecasting_mocks.d1
    invcov = forecasting_mocks.invcov
    expected = d1 @ invcov @ d1.T
    np.testing.assert_allclose(fisher, expected)

    info = forecasting_mocks.call_info
    dk_kwargs = info["dk_kwargs"]

    assert info["forecast_order"] == 1
    assert info["method"] == "local_polyfit"
    assert info["n_workers"] == 3
    assert dk_kwargs["degree"] == 5
    assert dk_kwargs["window"] == 4
    assert dk_kwargs["trim_fraction"] == 0.2
