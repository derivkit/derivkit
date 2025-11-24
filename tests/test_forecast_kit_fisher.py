"""Tests for Fisher matrix construction in LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    return np.zeros(2, dtype=float)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    return np.zeros(3, dtype=float)


@pytest.fixture
def forecasting_mocks(monkeypatch):
    """Provides fake derivative + covariance inversion with per-test state."""
    class Mocks:
        def __init__(self):
            """Initializes empty state for the mocks."""
            self.d1 = None
            self.invcov = None
            self.deriv_call_info = None
            self.invcov_call_info = None

        def set_state(self, d1, invcov):
            """Sets the derivative and inverse covariance matrices for this test."""
            self.d1 = np.asarray(d1, dtype=float)
            self.invcov = np.asarray(invcov, dtype=float)
            self.deriv_call_info = None
            self.invcov_call_info = None

    mocks = Mocks()

    def fake_get_derivatives(*args, **kwargs):
        """Takes the place of _get_derivatives, returns pre-set matrix and records call."""
        mocks.deriv_call_info = {
            "args": args,
            "kwargs": kwargs,
        }
        return mocks.d1

    def fake_invert_covariance(cov, warn_prefix=None):
        """Mimics invert_covariance, returns pre-set matrix and records call."""
        cov_arr = np.asarray(cov, dtype=float)
        mocks.invcov_call_info = {
            "cov": cov_arr,
            "warn_prefix": warn_prefix,
        }
        return mocks.invcov

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.invert_covariance",
        fake_invert_covariance,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.LikelihoodExpansion._get_derivatives",
        fake_get_derivatives,
        raising=True,
    )

    return mocks


def test_get_forecast_tensors_order1_matches_matrix_product(forecasting_mocks):
    """Tests that the Fisher from order=1 equals D1 @ invcov @ D1.T."""
    d1 = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
        ]
    )
    # these are dummy values; only invcov matters for the test
    cov = np.diag([2.0, 0.5])
    invcov = np.linalg.inv(cov)

    forecasting_mocks.set_state(d1=d1, invcov=invcov)

    theta0 = np.array([0.0, 0.0])
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
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

    lx = LikelihoodExpansion(
        function=three_obs_model,
        theta0=np.zeros(3),
        cov=cov,
    )

    fisher = lx.get_forecast_tensors(forecast_order=1)

    assert fisher.shape == (3, 3)
    np.testing.assert_allclose(fisher, fisher.T)


def test_get_forecast_tensors_order1_builds_fisher(forecasting_mocks):
    """Tests that get_forecast_tensors order=1 builds Fisher matrix correctly."""
    theta0 = np.array([0.1, -0.2])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    invcov = np.linalg.inv(cov)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.5], [-0.3, 2.0]]),
        invcov=invcov,
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    d1 = forecasting_mocks.d1
    expected = d1 @ invcov @ d1.T
    assert fisher.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(fisher, expected)

    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["method"] == "adaptive"
    assert kwargs["n_workers"] == 4
    assert kwargs["step_size"] == 1e-3

    np.testing.assert_allclose(forecasting_mocks.invcov_call_info["cov"], cov)
    assert forecasting_mocks.invcov_call_info["warn_prefix"] == "LikelihoodExpansion"


def test_get_forecast_tensors_invalid_order_raises():
    """Tests that invalid forecast_order raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    with pytest.raises(ValueError):
        lx.get_forecast_tensors(forecast_order=3)


def test_get_forecast_tensors_checks_model_output_length():
    """Tests that model output length is checked against covariance shape."""
    cov = np.eye(2)
    theta0 = np.array([0.0])

    # three_obs_model returns 3 observables but cov is 2x2 (n_observables=2),
    # so get_forecast_tensors should complain about the inconsistency.
    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    with pytest.raises(ValueError):
        lx.get_forecast_tensors(forecast_order=1)


def test_get_forecast_tensors_order1_default_n_workers(forecasting_mocks):
    """Tests that default n_workers=1 is used in _get_derivatives."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)
    fisher = lx.get_forecast_tensors(forecast_order=1)

    d1 = forecasting_mocks.d1
    invcov = forecasting_mocks.invcov
    expected = d1 @ invcov @ d1.T
    np.testing.assert_allclose(fisher, expected)

    # n_workers should default to 1 inside _get_derivatives
    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["n_workers"] == 1


def test_normalize_workers_various_inputs():
    """Tests that _normalize_workers handles various n_workers inputs correctly."""
    lx = LikelihoodExpansion(function=two_obs_model, theta0=np.zeros(1), cov=np.eye(2))

    assert lx._normalize_workers(1) == 1
    assert lx._normalize_workers(4) == 4
    assert lx._normalize_workers(0) == 1
    assert lx._normalize_workers(-3) == 1
    assert lx._normalize_workers(None) == 1
    assert lx._normalize_workers(2.7) == 2


@pytest.mark.parametrize("method", ["adaptive", "finite"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_get_forecast_tensors_order1_forwards_derivative_kwargs(
    forecasting_mocks,
    method,
    extrapolation,
    stencil,
):
    """Tests that derivative method and its kwargs are forwarded to he internal derivative routine."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method=method,
        n_workers=2,
        extrapolation=extrapolation,
        stencil=stencil,
    )

    d1 = forecasting_mocks.d1
    invcov = forecasting_mocks.invcov
    expected = d1 @ invcov @ d1.T
    np.testing.assert_allclose(fisher, expected)

    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["method"] == method
    assert kwargs["n_workers"] == 2
    assert kwargs["extrapolation"] == extrapolation
    assert kwargs["stencil"] == stencil


def test_get_forecast_tensors_order1_forwards_local_polyfit_kwargs(forecasting_mocks):
    """Test that local_polyfit method and its kwargs are forwarded to the internal derivative routine."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.0], [0.0, 1.0]]),
        invcov=np.eye(2),
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
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

    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["method"] == "local_polyfit"
    assert kwargs["n_workers"] == 3
    assert kwargs["degree"] == 5
    assert kwargs["window"] == 4
    assert kwargs["trim_fraction"] == 0.2
