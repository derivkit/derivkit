"""Unit tests for derivkit.forecasting.fisher_gaussian.build_gaussian_fisher_matrix."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.forecasting.fisher_gaussian import build_gaussian_fisher_matrix


def _analytic_fisher_mean(theta0: np.ndarray, cov0: np.ndarray) -> np.ndarray:
    """Returns a mean-derivative-only Fisher matrix."""
    _ = theta0
    return np.linalg.inv(cov0)


def _analytic_fisher_cov(theta0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns a covariance matrix and its derivatives for the toy model."""
    t0, t1 = float(theta0[0]), float(theta0[1])
    c0 = np.array([[np.exp(t0), t1], [t1, 2.0]], dtype=float)
    dc0 = np.array([[np.exp(t0), 0.0], [0.0, 0.0]], dtype=float)
    dc1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    return c0, np.stack([dc0, dc1], axis=0)  # (p=2, 2, 2)


def _analytic_fisher_total(theta0: np.ndarray) -> np.ndarray:
    """Returns a full Fisher matrix for the toy model (mean and cov term)."""
    cov0, dcov = _analytic_fisher_cov(theta0)
    cinv = np.linalg.inv(cov0)

    f_mean = _analytic_fisher_mean(theta0, cov0)

    # Cov term: 1/2 Tr[C^{-1} C_i C^{-1} C_j]
    p = theta0.size
    f_cov = np.zeros((p, p), dtype=float)
    for i in range(p):
        ai = cinv @ dcov[i]
        for j in range(p):
            aj = cinv @ dcov[j]
            f_cov[i, j] = 0.5 * np.trace(ai @ aj)

    f = f_mean + f_cov
    return 0.5 * (f + f.T)


def test_build_gaussian_fisher_matrix_callable_cov_matches_analytic():
    """Tests that callable covariance matches the analytic Fisher result."""

    def mu_fn(theta: np.ndarray) -> np.ndarray:
        """Toy mean function: mu(theta) = [theta0, theta1]."""
        return np.array([theta[0], theta[1]], dtype=float)

    def cov_fn(theta: np.ndarray) -> np.ndarray:
        """Toy covariance function with parameter dependence."""
        t0, t1 = float(theta[0]), float(theta[1])
        return np.array([[np.exp(t0), t1], [t1, 2.0]], dtype=float)

    theta0 = np.array([0.1, 0.2], dtype=float)

    fisher = build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=cov_fn,
        function=mu_fn,
        symmetrize_dcov=True,
        n_workers=1,
    )

    expected = _analytic_fisher_total(theta0)

    assert fisher.shape == (2, 2)
    assert np.allclose(fisher, fisher.T, rtol=0.0, atol=1e-13)
    assert np.allclose(fisher, expected, rtol=5e-6, atol=5e-8)


def test_build_gaussian_fisher_matrix_fixed_cov_skips_covariance_term():
    """Tests that fixed covariance skips the covariance-derivative term."""
    theta0 = np.array([0.1, 0.2], dtype=float)

    # Use the same cov(theta0) but pass it as a fixed matrix (array input)
    cov0, _dcov = _analytic_fisher_cov(theta0)

    def mu_fn(theta: np.ndarray) -> np.ndarray:
        return np.array([theta[0], theta[1]], dtype=float)

    fisher = build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=cov0,
        function=mu_fn,
        method="finite",
        n_workers=1,
    )

    expected = np.linalg.inv(cov0)

    assert fisher.shape == (2, 2)
    assert np.allclose(fisher, fisher.T, rtol=0.0, atol=1e-13)
    assert np.allclose(fisher, expected, rtol=5e-12, atol=5e-12)


def test_build_gaussian_fisher_matrix_raises_on_scalar_mean_with_multi_obs():
    """Tests that a scalar mean raises when the covariance implies multiple observables."""
    theta0 = np.array([0.0, 0.0], dtype=float)
    cov0 = np.eye(2, dtype=float)

    def scalar_mu(_theta: np.ndarray) -> float:
        return 1.0

    with pytest.raises(ValueError, match=r"returned a scalar"):
        _ = build_gaussian_fisher_matrix(
            theta0=theta0,
            cov=cov0,
            function=scalar_mu,
            method="finite",
            n_workers=1,
        )


def test_build_gaussian_fisher_matrix_callable_cov_constant_equals_fixed_cov():
    """Tests that a constant callable covariance matches the fixed covariance result."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    cov0, _dcov = _analytic_fisher_cov(theta0)

    def mu_fn(theta: np.ndarray) -> np.ndarray:
        """Toy mean function: mu(theta) = [theta0, theta1]."""
        return np.array([theta[0], theta[1]], dtype=float)

    def cov_const(_theta: np.ndarray) -> np.ndarray:
        """Constant covariance function."""
        return cov0

    fisher_fixed = build_gaussian_fisher_matrix(
        theta0=theta0, cov=cov0, function=mu_fn, method="finite", n_workers=1
    )
    fisher_call = build_gaussian_fisher_matrix(
        theta0=theta0, cov=cov_const, function=mu_fn, method="finite", n_workers=1
    )

    assert fisher_call.shape == fisher_fixed.shape == (2, 2)
    assert np.allclose(fisher_call, fisher_fixed, rtol=5e-12, atol=5e-12)


def test_build_gaussian_fisher_matrix_symmetrize_dcov_changes_result_for_nonsymmetric_cov_fn():
    """Tests that symmetrize_dcov changes the result for a non-symmetric covariance function."""
    theta0 = np.array([0.1, 0.2], dtype=float)

    def mu_fn(theta: np.ndarray) -> np.ndarray:
        """Toy mean function: mu(theta) = [theta0, theta1]."""
        return np.array([theta[0], theta[1]], dtype=float)

    def cov_nonsym(theta: np.ndarray) -> np.ndarray:
        """Toy non-symmetric covariance function."""
        t0, t1 = float(theta[0]), float(theta[1])
        base = np.array([[np.exp(t0), t1], [t1, 2.0]], dtype=float)
        skew = np.array([[0.0, 0.3 * t0], [0.0, 0.0]], dtype=float)
        return base + skew

    fisher_sym = build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=cov_nonsym,
        function=mu_fn,
        method="finite",
        n_workers=1,
        symmetrize_dcov=True,
    )
    fisher_raw = build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=cov_nonsym,
        function=mu_fn,
        n_workers=1,
        symmetrize_dcov=False,
    )

    assert np.allclose(fisher_sym, fisher_sym.T, rtol=0.0, atol=1e-13)
    assert np.allclose(fisher_raw, fisher_raw.T, rtol=0.0, atol=1e-13)

    diff = float(np.max(np.abs(fisher_sym - fisher_raw)))
    assert diff > 1e-10


def test_build_gaussian_fisher_matrix_pinv_fallback_warns_for_rank_deficient_cov(
    caplog: pytest.LogCaptureFixture,
):
    """Tests that a warning is logged when a rank-deficient covariance uses the pseudoinverse."""
    theta0 = np.array([0.1, 0.2], dtype=float)

    def mu_fn(theta: np.ndarray) -> np.ndarray:
        return np.array([theta[0], theta[1]], dtype=float)

    cov_singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)  # rank 1

    def cov_fn(_theta: np.ndarray) -> np.ndarray:
        # Callable so the covariance-derivative branch executes; always singular at theta0.
        return cov_singular

    caplog.clear()
    _ = build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=cov_fn,
        function=mu_fn,
        method="finite",
        n_workers=1,
    )

    joined = " ".join(rec.message for rec in caplog.records).lower()
    assert ("pseudoinverse" in joined) or ("pinv" in joined) or ("rank-deficient" in joined)


def test_build_gaussian_fisher_matrix_raises_on_vector_mean_wrong_length():
    """Tests that build_gaussian_fisher_matrix raises on wrong-length mean vector."""
    theta0 = np.array([0.0, 0.0], dtype=float)
    cov0 = np.eye(2, dtype=float)

    def bad_mu(_theta: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match=r"must return shape"):
        _ = build_gaussian_fisher_matrix(
            theta0=theta0,
            cov=cov0,
            function=bad_mu,
            method="finite",
            n_workers=1,
        )
