"""Unit tests for derivkit.forecasting.fisher_xy module."""

from __future__ import annotations

import numpy as np
import pytest

import derivkit.forecasting.fisher_xy as mod


def mu_xy_scalar(x: np.ndarray, theta: np.ndarray) -> float:
    """Scalar mean function: ``mu(x, theta) = theta[0] + 2 * x[0]``."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    return float(theta[0] + 2.0 * x[0])


def mu_xy_vector(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vector mean function: ``mu(x, theta) = [theta[0] + x[0], theta[1] - 3 * x[0]]``."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    return np.array([theta[0] + x[0], theta[1] - 3.0 * x[0]], dtype=np.float64)


def mu_xy_matrix(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Matrix mean function: ``mu(x, theta) = [[theta[0] + x[0], theta[1]], [x[0], theta[0] - theta[1]]]``."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    return np.array(
        [[theta[0] + x[0], theta[1]], [x[0], theta[0] - theta[1]]],
        dtype=np.float64,
    )


class FakeCalculusKit:
    """Fake CalculusKit that returns fixed Jacobian."""

    def __init__(self, function, x0):
        """Initialises the class."""
        self.function = function
        self.x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))

    def jacobian(self, method=None, n_workers=1, **dk_kwargs):
        """Returns fixed Jacobian."""
        _ = method, n_workers, dk_kwargs
        return np.array([[1.0], [-3.0]], dtype=np.float64)


def fake_build_gaussian_fisher_matrix(**kwargs):
    """Fake build_gaussian_fisher_matrix that returns fixed Fisher matrix."""
    _ = kwargs
    return np.array([[123.0]], dtype=np.float64)


def _stack_xy_cov(cxx: np.ndarray, cxy: np.ndarray, cyy: np.ndarray) -> np.ndarray:
    """Builds full covariance for stacked vector [x, y]."""
    cxx = np.asarray(cxx, dtype=np.float64)
    cxy = np.asarray(cxy, dtype=np.float64)
    cyy = np.asarray(cyy, dtype=np.float64)
    top = np.hstack([cxx, cxy])
    bot = np.hstack([cxy.T, cyy])
    return np.vstack([top, bot])


def test_mu_xy_given_theta_scalar_returns_1d():
    """Tests that mu_xy_given_theta wraps a scalar model output into a 1D data vector."""
    x = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0], dtype=np.float64)
    out = mod.mu_xy_given_theta(x, theta=theta, mu_xy=mu_xy_scalar)
    assert out.shape == (1,)
    assert np.allclose(out, np.array([5.0], dtype=np.float64))


def test_mu_xy_given_theta_matrix_flattens_c_order():
    """Tests that mu_xy_given_theta flattens matrix model outputs in C order into a 1D vector."""
    x = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0, 10.0], dtype=np.float64)
    out = mod.mu_xy_given_theta(x, theta=theta, mu_xy=mu_xy_matrix)
    expected = mu_xy_matrix(x, theta).ravel(order="C")
    assert out.ndim == 1
    assert np.allclose(out, expected)


def test_mu_xy_given_x0_vector_returns_1d():
    """Tests that mu_xy_given_x0 evaluates the model at fixed x0 and returns a 1D vector."""
    x0 = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0, 10.0], dtype=np.float64)
    out = mod.mu_xy_given_x0(theta, x0=x0, mu_xy=mu_xy_vector)
    assert out.shape == (2,)
    assert np.allclose(out, np.array([3.0, 4.0], dtype=np.float64))


def test_build_mu_theta_from_mu_xy_returns_callable_mean():
    """Tests that build_mu_theta_from_mu_xy returns a callable mu(theta) that matches mu_xy(x0, theta)."""
    x0 = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0, 10.0], dtype=np.float64)
    mu_theta = mod.build_mu_theta_from_mu_xy(mu_xy_vector, x0=x0)
    out = mu_theta(theta)
    assert out.shape == (2,)
    assert np.allclose(out, mu_xy_vector(x0, theta))


def test_build_t_matrix_uses_calculus_kit_and_shapes_to_2d(monkeypatch):
    """Tests that build_t_matrix uses CalculusKit.jacobian and returns a 2D (ny, nx) sensitivity matrix."""
    monkeypatch.setattr(mod, "CalculusKit", FakeCalculusKit)

    x0 = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0, 10.0], dtype=np.float64)

    t = mod.build_t_matrix(mu_xy_vector, x0=x0, theta=theta, method=None, n_workers=1)

    assert t.shape == (2, 1)
    assert np.allclose(t[:, 0], np.array([1.0, -3.0], dtype=np.float64))


def test_build_effective_covariance_r_matches_formula():
    """Tests that build_effective_covariance_r reproduces the Heavens et al. effective covariance formula."""
    x0 = np.array([2.0], dtype=np.float64)

    cxx = np.array([[4.0]], dtype=np.float64)
    cyy = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    cxy = np.array([[0.2, -0.1]], dtype=np.float64)
    cov = _stack_xy_cov(cxx, cxy, cyy)

    t = np.array([[1.0], [-3.0]], dtype=np.float64)

    r = mod.build_effective_covariance_r(cov=cov, x0=x0, t=t)

    expected = cyy - (cxy.T @ t.T) - (t @ cxy) + (t @ cxx @ t.T)
    assert r.shape == (2, 2)
    assert np.allclose(r, expected)


def test_build_effective_covariance_r_raises_on_bad_t_shape():
    """Tests that build_effective_covariance_r raises ValueError when t does not have shape (ny, nx)."""
    x0 = np.array([2.0], dtype=np.float64)

    cxx = np.array([[4.0]], dtype=np.float64)
    cyy = np.eye(2, dtype=np.float64)
    cxy = np.zeros((1, 2), dtype=np.float64)
    cov = _stack_xy_cov(cxx, cxy, cyy)

    # ny=2, nx=1, so t must be (2,1); give (1,2) to trigger the guard.
    t_bad = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"t must have shape"):
        mod.build_effective_covariance_r(cov=cov, x0=x0, t=t_bad)


def test_build_effective_covariance_r_raises_when_cov_too_small_for_nx():
    """Tests that build_effective_covariance_r raises ValueError when cov cannot be split for the given nx."""
    # nx inferred from x0.size = 2, but cov is only 1x1 -> cannot split into Cxx/Cxy/Cyy
    x0 = np.array([1.0, 2.0], dtype=np.float64)
    cov = np.eye(1, dtype=np.float64)
    t = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        mod.build_effective_covariance_r(cov=cov, x0=x0, t=t)


def test_effective_covariance_r_theta_recomputes_t_and_matches_build(monkeypatch):
    """Tests that effective_covariance_r_theta recomputes t(theta) and matches build_effective_covariance_r."""
    monkeypatch.setattr(mod, "CalculusKit", FakeCalculusKit)

    x0 = np.array([2.0], dtype=np.float64)
    theta = np.array([1.0, 10.0], dtype=np.float64)

    cxx = np.array([[4.0]], dtype=np.float64)
    cyy = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    cxy = np.array([[0.2, -0.1]], dtype=np.float64)
    cov = _stack_xy_cov(cxx, cxy, cyy)

    r = mod.effective_covariance_r_theta(
        theta,
        mu_xy=mu_xy_vector,
        x0=x0,
        cov=cov,
        method=None,
        n_workers=1,
        dk_kwargs={},
    )

    t = np.array([[1.0], [-3.0]], dtype=np.float64)
    expected = mod.build_effective_covariance_r(cov=cov, x0=x0, t=t)

    assert np.allclose(r, expected)


def test_build_xy_gaussian_fisher_matrix_wires_cov_and_function(monkeypatch):
    """Tests that build_xy_gaussian_fisher_matrix wires mu(theta) and R(theta) into build_gaussian_fisher_matrix."""
    monkeypatch.setattr(mod, "CalculusKit", FakeCalculusKit)
    monkeypatch.setattr(mod, "build_gaussian_fisher_matrix", fake_build_gaussian_fisher_matrix)

    theta0 = np.array([1.0, 10.0], dtype=np.float64)
    x0 = np.array([2.0], dtype=np.float64)

    cxx = np.array([[4.0]], dtype=np.float64)
    cyy = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    cxy = np.array([[0.2, -0.1]], dtype=np.float64)
    cov = _stack_xy_cov(cxx, cxy, cyy)

    out = mod.build_xy_gaussian_fisher_matrix(
        theta0=theta0,
        x0=x0,
        mu_xy=mu_xy_vector,
        cov=cov,
        method=None,
        n_workers=1,
    )

    assert out.shape == (1, 1)
    assert np.allclose(out, np.array([[123.0]], dtype=np.float64))
