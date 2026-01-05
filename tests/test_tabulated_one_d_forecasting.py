"""Unit tests for ForecastKit with 1D tabulated models."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit
from derivkit.tabulated_model.one_d import Tabulated1DModel


def make_tabulated_linear_obs(a_vec: np.ndarray) -> Tabulated1DModel:
    """Creates a tabulated linear observable model: f_i(x) = a_i * x."""
    x_tab = np.linspace(-1.0, 1.0, 51)
    y_tab = np.outer(x_tab, a_vec)
    return Tabulated1DModel(x_tab, y_tab)


# Parameters for the linear tabulated model
_LINEAR_A_VEC = np.array([1.0, 2.0, -1.5])
_LINEAR_B_VEC = np.array([0.5, -0.3, 1.1])
_LINEAR_C_VEC = np.array([0.1, 0.2, -0.4])
_LINEAR_TAB = make_tabulated_linear_obs(_LINEAR_A_VEC)


def linear_tabulated_model(theta: np.ndarray) -> np.ndarray:
    """A tabulated linear observable model."""
    theta = np.asarray(theta, dtype=float)
    t0, t1 = theta
    f0 = np.asarray(_LINEAR_TAB(t0), dtype=float)
    return f0 + _LINEAR_B_VEC * t1 + _LINEAR_C_VEC


def make_linear_tabulated_model():
    """A tabulated linear observable model setup for testing."""
    theta0 = np.array([0.2, -0.5])
    return linear_tabulated_model, theta0, _LINEAR_A_VEC, _LINEAR_B_VEC, _LINEAR_C_VEC


def make_tabulated_identity() -> Tabulated1DModel:
    """A tabulated identity model: f(x) = x."""
    x_tab = np.linspace(-1.0, 1.0, 101)
    y_tab = x_tab.copy()
    return Tabulated1DModel(x_tab, y_tab)


_QUAD_A = 2.0
_QUAD_TAB_ID = make_tabulated_identity()


def quadratic_tabulated_model(theta: np.ndarray) -> np.ndarray:
    """A single-parameter quadratic observable via tabulated model."""
    theta = np.asarray(theta, dtype=float)
    t = float(theta[0])
    x = float(_QUAD_TAB_ID(t))
    f = 0.5 * _QUAD_A * x**2
    return np.array([f], dtype=float)


def make_quadratic_tabulated_setup():
    """Returns (model, theta0, cov) for the quadratic DALI test."""
    theta0 = np.array([0.3])
    sigma = 0.1
    cov = sigma**2 * np.eye(1)
    return quadratic_tabulated_model, theta0, cov


def make_tabulated_cubic() -> Tabulated1DModel:
    """Returns a cubic mononomial of unit weight.

    Specifically it returns the value of :math: f(x) = x^3.
    """
    x_tab = np.linspace(-1.0, 1.0, 101)
    y_tab = x_tab**3
    return Tabulated1DModel(x_tab, y_tab)


_CUBIC_TAB = make_tabulated_cubic()


def cubic_tabulated_model(theta: np.ndarray) -> np.ndarray:
    """Returns a single-parameter cubic observable via tabulated model."""
    theta = np.asarray(theta, dtype=float)
    t = float(theta[0])
    return np.array([float(_CUBIC_TAB(t))], dtype=float)


def make_cubic_tabulated_setup():
    """Returns (model, theta0, cov) for the cubic DALI test."""
    theta0 = np.array([0.2])
    cov = 0.01 * np.eye(1)
    return cubic_tabulated_model, theta0, cov


def test_fisher_from_tabulated_linear_model():
    """Tests that Fisher matrix from tabulated linear model is correct."""
    model, theta0, a_vec, b_vec, c_vec = make_linear_tabulated_model()

    sigma = 0.1
    n_obs = a_vec.size
    cov = sigma**2 * np.eye(n_obs)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)
    fisher = fk.fisher(method="adaptive")

    assert fisher.shape == (2, 2)
    assert np.allclose(fisher, fisher.T)

    jac = np.column_stack([a_vec, b_vec])
    inv_cov = np.eye(n_obs) / sigma**2
    fisher_expected = jac.T @ inv_cov @ jac

    np.testing.assert_allclose(
        fisher, fisher_expected, rtol=1e-4, atol=1e-6
    )


def test_fisher_bias_from_tabulated_linear_model():
    """Tests that Fisher bias from tabulated linear model is correct."""
    model, theta0, a_vec, b_vec, c_vec = make_linear_tabulated_model()
    sigma = 0.1
    n_obs = a_vec.size
    cov = sigma**2 * np.eye(n_obs)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    data_ref = model(theta0)
    delta_data = np.zeros_like(data_ref)
    delta_data[0] = 2.5
    data_biased = data_ref + delta_data


    fisher = fk.fisher(method="adaptive")
    delta_nu = fk.delta_nu(data_biased=data_biased, data_unbiased=data_ref)
    bias_vec, dtheta = fk.fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        method="adaptive",
    )

    assert bias_vec.shape == theta0.shape
    assert np.any(np.abs(bias_vec) > 0.0)


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
def test_dali_from_tabulated_quadratic_model(method: str):
    """Tests that DALI from tabulated quadratic model is well-defined."""
    model, theta0, cov = make_quadratic_tabulated_setup()
    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    g, h = fk.dali(method=method)

    assert g.shape == (1, 1, 1)
    assert h.shape == (1, 1, 1, 1)

    assert np.all(np.isfinite(g))
    assert np.all(np.isfinite(h))

    assert np.any(np.abs(g) > 0.0) or np.any(np.abs(h) > 0.0)


@pytest.mark.parametrize("method", ["adaptive"])
def test_dali_nontrivial_symmetry_from_cubic_model(method: str):
    """Tests that DALI from tabulated cubic model is non-trivial."""
    model, theta0, cov = make_cubic_tabulated_setup()

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)
    g, h = fk.dali(method=method)

    assert g.shape == (1, 1, 1)
    assert h.shape == (1, 1, 1, 1)

    assert not np.allclose(g, 0.0)
    assert not np.allclose(h, 0.0)