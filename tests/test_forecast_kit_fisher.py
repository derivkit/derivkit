"""Tests for Fisher matrix construction in LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion

# Globals used by fake helpers
D1_GLOBAL = None
DERIV_CALL_INFO: dict = {}
INVCOV_GLOBAL = None
INVCOV_CALL_INFO: dict = {}


def fake_get_derivatives(*args, **kwargs):
    """Takes the place of _get_derivatives and returns a pre-set matrix, recording arguments."""
    global DERIV_CALL_INFO
    DERIV_CALL_INFO = {
        "args": args,
        "kwargs": kwargs,
    }
    return D1_GLOBAL


def fake_invert_covariance(cov, warn_prefix=None):
    """Mimics invert_covariance, returning a pre-set matrix and recording arguments."""
    global INVCOV_CALL_INFO
    cov_arr = np.asarray(cov, dtype=float)
    INVCOV_CALL_INFO = {
        "cov": cov_arr,
        "warn_prefix": warn_prefix,
    }
    return INVCOV_GLOBAL


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    _ = np.atleast_1d(theta)
    return np.zeros(2, dtype=float)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)
    return np.zeros(3, dtype=float)


def wrong_length_model(theta):
    """Model that returns a wrong-length observable vector (for shape check)."""
    _ = np.atleast_1d(theta)
    return np.zeros(3, dtype=float)


def test_build_fisher_matches_matrix_product():
    """Tests that _build_fisher computes F = d1 @ invcov @ d1.T correctly."""
    # P = 2 parameters, N = 3 observables
    d1 = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
        ]
    )
    cov = np.diag([2.0, 1.0, 0.5])
    invcov = np.linalg.inv(cov)

    # n_observables is inferred from cov, so use shape (3, 3)
    theta0 = np.array([0.0, 0.0])
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=np.eye(3))

    fisher_einsum = lx._build_fisher(d1, invcov)
    fisher_manual = d1 @ invcov @ d1.T

    assert fisher_einsum.shape == (2, 2)
    np.testing.assert_allclose(fisher_einsum, fisher_manual)


def test_get_forecast_tensors_order1_builds_fisher(monkeypatch):
    """Tests that get_forecast_tensors(order=1) builds the Fisher matrix correctly."""
    # P = 2 parameters, N = 2 observables
    theta0 = np.array([0.1, -0.2])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])

    # Set globals used by the fakes
    global D1_GLOBAL, INVCOV_GLOBAL, DERIV_CALL_INFO, INVCOV_CALL_INFO
    D1_GLOBAL = np.array(
        [
            [1.0, 0.5],
            [-0.3, 2.0],
        ]
    )
    INVCOV_GLOBAL = np.array([[10.0, 0.0], [0.0, 0.5]])
    DERIV_CALL_INFO = {}
    INVCOV_CALL_INFO = {}

    # Patch internals: invert_covariance and _get_derivatives
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

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    # Shape and value check
    expected = lx._build_fisher(D1_GLOBAL, INVCOV_GLOBAL)
    assert fisher.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(fisher, expected)

    # Check that _get_derivatives was called as expected
    assert DERIV_CALL_INFO["kwargs"]["order"] == 1
    assert DERIV_CALL_INFO["kwargs"]["method"] == "adaptive"
    assert DERIV_CALL_INFO["kwargs"]["n_workers"] == 4
    assert DERIV_CALL_INFO["kwargs"]["step_size"] == 1e-3

    # Check that invert_covariance saw the stored covariance
    np.testing.assert_allclose(INVCOV_CALL_INFO["cov"], cov)
    assert INVCOV_CALL_INFO["warn_prefix"] == "LikelihoodExpansion"


def test_build_fisher_is_symmetric():
    """Tests that _build_fisher returns a symmetric Fisher matrix."""
    rng = np.random.default_rng(123)
    # P = 3 parameters, N = 4 observables
    d1 = rng.normal(size=(3, 4))
    # Make a symmetric positive-definite covariance
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + np.eye(4)
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(
        function=three_obs_model,
        theta0=np.zeros(3),
        cov=np.eye(4),
    )

    fisher = lx._build_fisher(d1, invcov)

    # Symmetry check
    np.testing.assert_allclose(fisher, fisher.T)


def test_get_forecast_tensors_invalid_order_raises():
    """Tests that get_forecast_tensors raises ValueError for unsupported forecast_order."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    with pytest.raises(ValueError):
        lx.get_forecast_tensors(forecast_order=3)


def test_get_forecast_tensors_checks_model_output_length():
    """Tests that get_forecast_tensors raises ValueError if model output length mismatches cov shape."""
    cov = np.eye(2)
    theta0 = np.array([0.0])

    # wrong_length_model returns 3 observables, so this should fail
    lx = LikelihoodExpansion(function=wrong_length_model, theta0=theta0, cov=cov)

    with pytest.raises(ValueError):
        lx.get_forecast_tensors(forecast_order=1)


def test_get_forecast_tensors_order1_default_n_workers(monkeypatch):
    """Tests that get_forecast_tensors(order=1) defaults n_workers to 1 in _get_derivatives."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    global D1_GLOBAL, INVCOV_GLOBAL, DERIV_CALL_INFO, INVCOV_CALL_INFO
    D1_GLOBAL = np.array([[1.0, 0.0], [0.0, 1.0]])
    INVCOV_GLOBAL = np.eye(2)
    DERIV_CALL_INFO = {}
    INVCOV_CALL_INFO = {}

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

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)
    fisher = lx.get_forecast_tensors(forecast_order=1)

    expected = lx._build_fisher(D1_GLOBAL, INVCOV_GLOBAL)
    np.testing.assert_allclose(fisher, expected)

    # n_workers should default to 1 inside _get_derivatives
    assert DERIV_CALL_INFO["kwargs"]["order"] == 1
    assert DERIV_CALL_INFO["kwargs"]["n_workers"] == 1


def test_normalize_workers_various_inputs():
    """Tests that _normalize_workers handles various n_workers inputs correctly."""
    lx = LikelihoodExpansion(function=two_obs_model, theta0=np.zeros(1), cov=np.eye(2))

    assert lx._normalize_workers(1) == 1
    assert lx._normalize_workers(4) == 4
    assert lx._normalize_workers(0) == 1
    assert lx._normalize_workers(-3) == 1
    assert lx._normalize_workers(None) == 1
    assert lx._normalize_workers(2.7) == 2


@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
def test_get_forecast_tensors_order1_forwards_all_extrapolations(monkeypatch, extrapolation):
    """Tests that all extrapolation options are forwarded to _get_derivatives."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    global D1_GLOBAL, INVCOV_GLOBAL, DERIV_CALL_INFO, INVCOV_CALL_INFO
    D1_GLOBAL = np.array([[1.0, 0.0], [0.0, 1.0]])
    INVCOV_GLOBAL = np.eye(2)
    DERIV_CALL_INFO = {}
    INVCOV_CALL_INFO = {}

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

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method="finite",
        n_workers=2,
        extrapolation=extrapolation,
        stencil="7-point",
    )

    expected = lx._build_fisher(D1_GLOBAL, INVCOV_GLOBAL)
    np.testing.assert_allclose(fisher, expected)

    # Check forwarding
    assert DERIV_CALL_INFO["kwargs"]["order"] == 1
    assert DERIV_CALL_INFO["kwargs"]["method"] == "finite"
    assert DERIV_CALL_INFO["kwargs"]["n_workers"] == 2
    assert DERIV_CALL_INFO["kwargs"]["extrapolation"] == extrapolation
    assert DERIV_CALL_INFO["kwargs"]["stencil"] == "7-point"


def test_get_forecast_tensors_order1_forwards_local_polyfit_kwargs(monkeypatch):
    """Tests that local polyfit method and its kwargs are forwarded to _get_derivatives."""
    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    global D1_GLOBAL, INVCOV_GLOBAL, DERIV_CALL_INFO, INVCOV_CALL_INFO
    D1_GLOBAL = np.array([[1.0, 0.0], [0.0, 1.0]])
    INVCOV_GLOBAL = np.eye(2)
    DERIV_CALL_INFO = {}
    INVCOV_CALL_INFO = {}

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

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = lx.get_forecast_tensors(
        forecast_order=1,
        method="local_polyfit",
        n_workers=3,
        degree=5,
        window=4,
        trim_fraction=0.2,
    )

    expected = lx._build_fisher(D1_GLOBAL, INVCOV_GLOBAL)
    np.testing.assert_allclose(fisher, expected)

    # Check forwarding of method + polyfit-specific kwargs
    assert DERIV_CALL_INFO["kwargs"]["order"] == 1
    assert DERIV_CALL_INFO["kwargs"]["method"] == "local_polyfit"
    assert DERIV_CALL_INFO["kwargs"]["n_workers"] == 3
    assert DERIV_CALL_INFO["kwargs"]["degree"] == 5
    assert DERIV_CALL_INFO["kwargs"]["window"] == 4
    assert DERIV_CALL_INFO["kwargs"]["trim_fraction"] == 0.2
