"""Unit tests for derivkit.forecasting.fisher_general.build_gaussian_fisher_matrix."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.forecasting.fisher_gaussian_general import (
    build_gaussian_fisher_matrix as gg_fisher,
)


def make_spd_matrix(n: int) -> np.ndarray:
    """Return a symmetric positive definite matrix with no randomness.

    Constructed as ``A.T @ A + 1e-1 * I`` where ``A`` is built from
    ``np.arange(1, n*n + 1).reshape(n, n)``.
    """
    base = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    mat = base.T @ base
    mat += 1e-1 * np.eye(n)
    return mat


def mean_linear_model(theta: np.ndarray, mean_matrix: np.ndarray) -> np.ndarray:
    """Linear mean model mu(theta) = mean_matrix @ theta."""
    return mean_matrix @ theta


def covariance_linear_model(
    theta: np.ndarray,
    cov_base: np.ndarray,
    cov_slope_0: np.ndarray,
    cov_slope_1: np.ndarray,
) -> np.ndarray:
    """Linear covariance model C(theta) = cov_base + theta0*cov_slope_0 + theta1*cov_slope_1."""
    return cov_base + float(theta[0]) * cov_slope_0 + float(theta[1]) * cov_slope_1


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_invalid_term_raises(method: str) -> None:
    """Tests that invalid term raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    cov0 = make_spd_matrix(2)

    with pytest.raises(ValueError, match="term must be one of"):
        gg_fisher(theta0=theta0, cov=cov0, function=None, term="nope", method=method)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_requires_function_for_mean_or_both(method: str) -> None:
    """Tests that missing function raises ValueError for mean or both terms."""
    theta0 = np.array([0.0, 0.0])
    cov0 = make_spd_matrix(2)

    with pytest.raises(ValueError, match="function must be provided"):
        gg_fisher(theta0=theta0, cov=cov0, function=None, term="mean", method=method)

    with pytest.raises(ValueError, match="function must be provided"):
        gg_fisher(theta0=theta0, cov=cov0, function=None, term="both", method=method)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_requires_cov_callable_for_cov_term(method: str) -> None:
    """Tests that missing cov callable raises ValueError for cov term."""
    theta0 = np.array([0.0, 0.0])
    cov0 = make_spd_matrix(2)

    with pytest.raises(ValueError, match="requires a parameter-dependent covariance callable"):
        gg_fisher(theta0=theta0, cov=cov0, function=None, term="cov", method=method)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_mean_term_matches_analytic_linear_mean(method: str) -> None:
    """Tests that generalized Fisher from linear mean model matches analytic result."""
    mean_matrix = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
            [3.0, 0.0],
        ],
        dtype=float,
    )
    theta0 = np.array([0.1, -0.2])
    cov0 = make_spd_matrix(3)

    cov_inv = np.linalg.inv(cov0)
    expected = mean_matrix.T @ cov_inv @ mean_matrix
    expected = 0.5 * (expected + expected.T)

    got = gg_fisher(
        theta0=theta0,
        cov=cov0,
        function=lambda t: mean_linear_model(t, mean_matrix),
        term="mean",
        method=method,
        n_workers=1,
    )

    if method == "finite":
        rtol, atol = 5e-5, 1e-8
    else:
        rtol, atol = 5e-3, 1e-6

    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_cov_only_matches_analytic_linear_cov(method: str) -> None:
    """Tests that generalized Fisher from linear covariance model matches analytic result."""
    theta0 = np.array([0.2, -0.1])

    cov_base = make_spd_matrix(2)
    cov_slope_0 = np.array([[0.3, 0.1], [0.1, 0.2]], dtype=float)
    cov_slope_1 = np.array([[0.1, -0.05], [-0.05, 0.4]], dtype=float)

    cov_at_theta0 = covariance_linear_model(theta0, cov_base, cov_slope_0, cov_slope_1)
    cov_inv = np.linalg.inv(cov_at_theta0)

    slopes = [cov_slope_0, cov_slope_1]
    expected = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            expected[i, j] = 0.5 * np.trace(cov_inv @ slopes[i] @ cov_inv @ slopes[j])
    expected = 0.5 * (expected + expected.T)

    got = gg_fisher(
        theta0=theta0,
        cov=(cov_at_theta0, lambda t: covariance_linear_model(t, cov_base, cov_slope_0, cov_slope_1)),
        function=None,
        term="cov",
        method=method,
        n_workers=1,
    )

    if method == "finite":
        rtol, atol = 5e-5, 1e-8
    else:
        rtol, atol = 5e-3, 1e-6

    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_both_equals_mean_plus_cov(method: str) -> None:
    """Tests that generalized Fisher from both terms equals sum of individual terms."""
    mean_matrix = np.array([[1.0, 0.2], [-0.3, 2.0]], dtype=float)
    theta0 = np.array([0.05, -0.02])

    cov_base = make_spd_matrix(2)
    cov_slope_0 = np.array([[0.2, 0.05], [0.05, 0.1]], dtype=float)
    cov_slope_1 = np.array([[0.1, -0.02], [-0.02, 0.3]], dtype=float)

    cov_at_theta0 = covariance_linear_model(theta0, cov_base, cov_slope_0, cov_slope_1)

    fisher_mean = gg_fisher(
        theta0=theta0,
        cov=cov_at_theta0,
        function=lambda t: mean_linear_model(t, mean_matrix),
        term="mean",
        method=method,
        n_workers=1,
    )
    fisher_cov = gg_fisher(
        theta0=theta0,
        cov=(cov_at_theta0, lambda t: covariance_linear_model(t, cov_base, cov_slope_0, cov_slope_1)),
        function=None,
        term="cov",
        method=method,
        n_workers=1,
    )
    fisher_both = gg_fisher(
        theta0=theta0,
        cov=(cov_at_theta0, lambda t: covariance_linear_model(t, cov_base, cov_slope_0, cov_slope_1)),
        function=lambda t: mean_linear_model(t, mean_matrix),
        term="both",
        method=method,
        n_workers=1,
    )

    np.testing.assert_allclose(fisher_both, fisher_mean + fisher_cov, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_scalar_mu_requires_nobs_1(method: str) -> None:
    """Tests that scalar mean function raises ValueError if n_observables != 1."""
    theta0 = np.array([0.0, 0.0])
    cov0 = make_spd_matrix(2)

    def mean_scalar(theta: np.ndarray) -> float:
        return float(theta[0] + theta[1])

    with pytest.raises(ValueError, match="returned a scalar, but cov implies n_observables=2"):
        gg_fisher(theta0=theta0, cov=cov0, function=mean_scalar, term="mean", method=method)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_output_is_symmetric(method: str) -> None:
    """Tests that generalized Fisher matrix output is symmetric."""
    theta0 = np.array([0.1, -0.2])
    cov0 = make_spd_matrix(2)

    mean_matrix = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=float)

    fisher = gg_fisher(
        theta0=theta0,
        cov=cov0,
        function=lambda t: mean_linear_model(t, mean_matrix),
        term="mean",
        method=method,
        n_workers=1,
    )

    np.testing.assert_allclose(fisher, fisher.T, rtol=0.0, atol=0.0)
