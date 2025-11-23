"""Tests for Fisher matrix construction in LikelihoodExpansion."""

import pytest

import numpy as np

from derivkit.forecasting.expansions import LikelihoodExpansion


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    return np.zeros(2, dtype=float)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    return np.zeros(3, dtype=float)


def wrong_length_model(theta):
    """Model that returns a wrong-length observable vector (for shape check)."""
    return np.zeros(3, dtype=float)


@pytest.fixture
def forecasting_mocks(monkeypatch):
    """Provides fake derivative + covariance inversion with per-test state."""

    class ForecastingMocks:
        """Holds state and fakes for monkeypatching."""
        def __init__(self):
            """Initializes with empty state."""
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

        def fake_get_derivatives(self, *args, **kwargs):
            """Takes the place of _get_derivatives, returns pre-set matrix and records call."""
            self.deriv_call_info = {
                "args": args,
                "kwargs": kwargs,
            }
            return self.d1

        def fake_invert_covariance(self, cov, warn_prefix=None):
            """Mimics invert_covariance, returns pre-set matrix and records call."""
            cov_arr = np.asarray(cov, dtype=float)
            self.invcov_call_info = {
                "cov": cov_arr,
                "warn_prefix": warn_prefix,
            }
            return self.invcov

    mocks = ForecastingMocks()

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.invert_covariance",
        mocks.fake_invert_covariance,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.LikelihoodExpansion._get_derivatives",
        mocks.fake_get_derivatives,
        raising=True,
    )

    return mocks


def test_build_fisher_matches_matrix_product():
    """Tests that _build_fisher computes Fisher matrix as expected."""
    d1 = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
        ]
    )
    cov = np.diag([2.0, 1.0, 0.5])
    invcov = np.linalg.inv(cov)

    theta0 = np.array([0.0, 0.0])
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=np.eye(3))

    fisher = lx._build_fisher(d1, invcov)
    expected = d1 @ invcov @ d1.T

    assert fisher.shape == (2, 2)
    np.testing.assert_allclose(fisher, expected)


def test_build_fisher_is_symmetric():
    """Tests that _build_fisher returns a symmetric Fisher matrix."""
    rng = np.random.default_rng(123)
    d1 = rng.normal(size=(3, 4))
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(
        function=three_obs_model,
        theta0=np.zeros(3),
        cov=np.eye(4),
    )

    fisher = lx._build_fisher(d1, invcov)

    np.testing.assert_allclose(fisher, fisher.T)


def test_get_forecast_tensors_order1_builds_fisher(forecasting_mocks):
    """Tests that get_forecast_tensors order=1 builds Fisher matrix correctly."""
    theta0 = np.array([0.1, -0.2])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])

    forecasting_mocks.set_state(
        d1=np.array([[1.0, 0.5], [-0.3, 2.0]]),
        invcov=np.array([[10.0, 0.0], [0.0, 0.5]]),
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    expected = lx._build_fisher(forecasting_mocks.d1, forecasting_mocks.invcov)
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

    # wrong_length_model returns 3 observables, so this should fail
    lx = LikelihoodExpansion(function=wrong_length_model, theta0=theta0, cov=cov)

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

    expected = lx._build_fisher(forecasting_mocks.d1, forecasting_mocks.invcov)
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
    """Tests that derivative method and its kwargs are forwarded to _get_derivatives."""
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

    expected = lx._build_fisher(forecasting_mocks.d1, forecasting_mocks.invcov)
    np.testing.assert_allclose(fisher, expected)

    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["method"] == method
    assert kwargs["n_workers"] == 2
    assert kwargs["extrapolation"] == extrapolation
    assert kwargs["stencil"] == stencil


def test_get_forecast_tensors_order1_forwards_local_polyfit_kwargs(forecasting_mocks):
    """Test that local_polyfit method and its kwargs are forwarded to _get_derivatives."""
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

    expected = lx._build_fisher(forecasting_mocks.d1, forecasting_mocks.invcov)
    np.testing.assert_allclose(fisher, expected)

    kwargs = forecasting_mocks.deriv_call_info["kwargs"]
    assert kwargs["order"] == 1
    assert kwargs["method"] == "local_polyfit"
    assert kwargs["n_workers"] == 3
    assert kwargs["degree"] == 5
    assert kwargs["window"] == 4
    assert kwargs["trim_fraction"] == 0.2
