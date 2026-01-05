"""Unit tests for derivkit.forecasting.fisher_general.build_gaussian_fisher_matrix."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.fisher_gaussian import (
    build_gaussian_fisher_matrix as gg_fisher,
)


def make_spd_matrix(n: int) -> np.ndarray:
    """Return a symmetric positive definite matrix with no randomness."""
    base = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    mat = base.T @ base
    mat += 1e-1 * np.eye(n)
    return mat


def mean_linear_model(theta: np.ndarray, mean_matrix: np.ndarray) -> np.ndarray:
    """Return mu(theta) = mean_matrix @ theta."""
    return mean_matrix @ theta


def covariance_linear_model(
    theta: np.ndarray,
    cov_base: np.ndarray,
    cov_slope_0: np.ndarray,
    cov_slope_1: np.ndarray,
) -> np.ndarray:
    """Return c(theta) = cov_base + theta0*cov_slope_0 + theta1*cov_slope_1."""
    return cov_base + float(theta[0]) * cov_slope_0 + float(theta[1]) * cov_slope_1


def analytic_fisher_mean_only(mean_matrix: np.ndarray, cov0: np.ndarray) -> np.ndarray:
    """Return analytic mean-only Fisher for linear mean and fixed covariance."""
    cov_inv = np.linalg.inv(cov0)
    fisher = mean_matrix.T @ cov_inv @ mean_matrix
    return 0.5 * (fisher + fisher.T)


def analytic_fisher_cov_only(
    cov0: np.ndarray,
    cov_slope_0: np.ndarray,
    cov_slope_1: np.ndarray,
    *,
    symmetrize: bool,
) -> np.ndarray:
    """Return analytic cov-only Fisher for linear covariance model."""
    cov_inv = np.linalg.inv(cov0)
    slope_0 = cov_slope_0
    slope_1 = cov_slope_1
    if symmetrize:
        slope_0 = 0.5 * (slope_0 + slope_0.T)
        slope_1 = 0.5 * (slope_1 + slope_1.T)

    fisher = np.zeros((2, 2), dtype=float)
    slopes = [slope_0, slope_1]
    for i in range(2):
        for j in range(2):
            fisher[i, j] = 0.5 * np.trace(cov_inv @ slopes[i] @ cov_inv @ slopes[j])
    return 0.5 * (fisher + fisher.T)


def method_tolerances(method: str) -> tuple[float, float]:
    """Return numeric tolerances for the given derivative method."""
    if method == "finite":
        return 5e-5, 1e-8
    return 5e-3, 1e-6


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_mean_term_matches_analytic_linear_mean(method: str) -> None:
    """Test mean-only fisher matches analytic for linear mean."""
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

    expected = analytic_fisher_mean_only(mean_matrix, cov0)

    mean_fn = partial(mean_linear_model, mean_matrix=mean_matrix)
    got = gg_fisher(
        theta0=theta0,
        cov=cov0,
        function=mean_fn,
        method=method,
        n_workers=1,
    )

    rtol, atol = method_tolerances(method)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_mean_plus_cov_matches_analytic(method: str) -> None:
    """Test mean+cov fisher matches analytic for linear mean and cov."""
    theta0 = np.array([0.2, -0.1])

    mean_matrix = np.array([[1.0, 0.2], [-0.3, 2.0]], dtype=float)

    cov_base = make_spd_matrix(2)
    cov_slope_0 = np.array([[0.3, 0.1], [0.1, 0.2]], dtype=float)
    cov_slope_1 = np.array([[0.1, -0.05], [-0.05, 0.4]], dtype=float)

    cov0 = covariance_linear_model(theta0, cov_base, cov_slope_0, cov_slope_1)

    expected_mean = analytic_fisher_mean_only(mean_matrix, cov0)
    expected_cov = analytic_fisher_cov_only(
        cov0,
        cov_slope_0,
        cov_slope_1,
        symmetrize=True,
    )
    expected = expected_mean + expected_cov

    mean_fn = partial(mean_linear_model, mean_matrix=mean_matrix)
    cov_fn = partial(
        covariance_linear_model,
        cov_base=cov_base,
        cov_slope_0=cov_slope_0,
        cov_slope_1=cov_slope_1,
    )

    got = gg_fisher(
        theta0=theta0,
        cov=cov_fn,
        function=mean_fn,
        method=method,
        n_workers=1,
    )

    rtol, atol = method_tolerances(method)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_mean_plus_cov_matches_analytic_no_symmetrize(method: str) -> None:
    """Test mean+cov fisher matches analytic without dcov symmetrization."""
    theta0 = np.array([0.2, -0.1])

    mean_matrix = np.array([[1.0, 0.2], [-0.3, 2.0]], dtype=float)

    cov_base = make_spd_matrix(2)
    cov_slope_0 = np.array([[0.3, 0.1], [0.0, 0.2]], dtype=float)
    cov_slope_1 = np.array([[0.1, -0.05], [0.02, 0.4]], dtype=float)

    cov0 = covariance_linear_model(theta0, cov_base, cov_slope_0, cov_slope_1)

    expected_mean = analytic_fisher_mean_only(mean_matrix, cov0)
    expected_cov = analytic_fisher_cov_only(
        cov0,
        cov_slope_0,
        cov_slope_1,
        symmetrize=False,
    )
    expected = expected_mean + expected_cov

    mean_fn = partial(mean_linear_model, mean_matrix=mean_matrix)
    cov_fn = partial(
        covariance_linear_model,
        cov_base=cov_base,
        cov_slope_0=cov_slope_0,
        cov_slope_1=cov_slope_1,
    )

    got = gg_fisher(
        theta0=theta0,
        cov=cov_fn,
        function=mean_fn,
        method=method,
        n_workers=1,
        symmetrize_dcov=False,
    )

    rtol, atol = method_tolerances(method)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_scalar_mu_requires_nobs_1(method: str) -> None:
    """Test scalar mean raises when n_observables is not one."""
    theta0 = np.array([0.0, 0.0])
    cov0 = make_spd_matrix(2)

    def mean_scalar(theta: np.ndarray) -> float:
        return float(theta[0] + theta[1])

    with pytest.raises(ValueError, match="returned a scalar"):
        gg_fisher(
            theta0=theta0,
            cov=cov0,
            function=mean_scalar,
            method=method,
            n_workers=1,
        )


@pytest.mark.parametrize("method", ["finite", "adaptive", "localpolynomial"])
def test_build_generalized_fisher_output_is_symmetric(method: str) -> None:
    """Test fisher output is symmetric."""
    theta0 = np.array([0.1, -0.2])
    cov0 = make_spd_matrix(2)

    mean_matrix = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=float)

    mean_fn = partial(mean_linear_model, mean_matrix=mean_matrix)
    fisher = gg_fisher(
        theta0=theta0,
        cov=cov0,
        function=mean_fn,
        method=method,
        n_workers=1,
    )

    np.testing.assert_allclose(fisher, fisher.T, rtol=0.0, atol=0.0)
