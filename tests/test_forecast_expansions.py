"""Unit tests for the derivkit.forecasting.expansions module."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import (
    build_delta_chi2_dali,
    build_delta_chi2_fisher,
    build_logposterior_dali,
    build_logposterior_fisher,
    build_submatrix_dali,
    build_submatrix_fisher,
)


def _spd_fisher(p: int, seed: int = 0) -> np.ndarray:
    """Returns a random symmetric positive definite Fisher matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(p, p))
    f = a.T @ a
    f += 1e-2 * np.eye(p)
    return f


def _toy_tensors(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns deterministic G and H DALI tensors for testing."""
    # Deterministic but nontrivial tensors
    g = np.zeros((p, p, p), dtype=float)
    h = np.zeros((p, p, p, p), dtype=float)

    for i in range(p):
        g[i, i, i] = 0.3 * (i + 1)
        for j in range(p):
            h[i, i, j, j] = 0.2 * (i + 1) * (j + 1)
    return g, h


def _manual_g3(g: np.ndarray, d: np.ndarray) -> float:
    """Returns G:d^3 tensor computed via einsum for testing."""
    return float(np.einsum("ijk,i,j,k->", g, d, d, d))


def _manual_h4(h: np.ndarray, d: np.ndarray) -> float:
    """Returns H:d^4 tensor computed via einsum for testing."""
    return float(np.einsum("ijkl,i,j,k,l->", h, d, d, d, d))


def test_submatrix_fisher_extracts_correct_block():
    """Tests that submatrix_fisher extracts the correct sub-block."""
    f = np.arange(25, dtype=float).reshape(5, 5)
    idx = [0, 2, 4]
    sub = build_submatrix_fisher(f, idx)
    assert sub.shape == (3, 3)
    assert np.allclose(sub, f[np.ix_(idx, idx)])


def test_submatrix_fisher_raises_on_non_square():
    """Tests that submatrix_fisher raises on non-square input."""
    f = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError, match="square 2D"):
        build_submatrix_fisher(f, [0])


def test_submatrix_dali_extracts_all_tensors():
    """Tests that submatrix_dali extracts correct sub-blocks for all tensors."""
    p = 5
    theta0 = np.linspace(0.0, 1.0, p)
    f = _spd_fisher(p, seed=1)
    g, h = _toy_tensors(p)

    idx = [1, 3, 4]
    t0s, fs, gs, hs = build_submatrix_dali(theta0, f, g, h, idx)

    assert t0s.shape == (len(idx),)
    assert fs.shape == (len(idx), len(idx))
    assert gs.shape == (len(idx), len(idx), len(idx))
    assert hs.shape == (len(idx), len(idx), len(idx), len(idx))

    assert np.allclose(t0s, theta0[idx])
    assert np.allclose(fs, f[np.ix_(idx, idx)])
    assert np.allclose(gs, g[np.ix_(idx, idx, idx)])
    assert np.allclose(hs, h[np.ix_(idx, idx, idx, idx)])


def test_submatrix_dali_h_none_propagates():
    """Tests that submatrix_dali propagates None for H DALI tensor tensor."""
    p = 4
    theta0 = np.zeros(p)
    f = _spd_fisher(p, seed=2)
    g, _ = _toy_tensors(p)

    _, _, _, hs = build_submatrix_dali(theta0, f, g, None, [0, 2])
    assert hs is None


def test_delta_chi2_fisher_matches_manual():
    """Tests that delta_chi2_fisher matches manual quadratic form calculation."""
    p = 4
    theta0 = np.array([0.2, -0.1, 0.3, 0.0])
    theta = np.array([0.25, -0.05, 0.10, 0.2])
    f = _spd_fisher(p, seed=3)

    d = theta - theta0
    expected = float(d @ f @ d)
    assert build_delta_chi2_fisher(theta, theta0, f) == pytest.approx(expected)


def test_logposterior_fisher_no_prior_is_minus_half_chi2():
    """Tests that logposterior_fisher without prior is -0.5 * delta_chi2_fisher."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, -0.2, 0.05])
    f = _spd_fisher(p, seed=4)

    chi2 = build_delta_chi2_fisher(theta, theta0, f)
    lp = build_logposterior_fisher(theta, theta0, f)
    assert lp == pytest.approx(-0.5 * chi2)


def test_logposterior_fisher_with_logprior_adds_term():
    """Tests that logposterior_fisher with logprior adds the prior term correctly."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.array([0.2, -0.1])
    f = _spd_fisher(p, seed=5)

    def logprior(th: np.ndarray) -> float:
        _ = th
        return -1.23

    chi2 = build_delta_chi2_fisher(theta, theta0, f)
    lp = build_logposterior_fisher(theta, theta0, f, logprior=logprior)
    assert lp == pytest.approx(-1.23 - 0.5 * chi2)


def test_logposterior_fisher_prior_returns_minus_inf_short_circuits():
    """Tests that logposterior_fisher returns -inf if logprior is -inf."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.array([0.2, -0.1])
    f = _spd_fisher(p, seed=6)

    def logprior(_: np.ndarray) -> float:
        return -np.inf

    lp = build_logposterior_fisher(theta, theta0, f, logprior=logprior)
    assert lp == -np.inf


def test_logposterior_fisher_rejects_prior_spec_and_logprior_together():
    """Tests that logposterior_fisher raises if both prior spec and logprior are given."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2])
    f = _spd_fisher(p, seed=7)

    with pytest.raises(ValueError, match="either `logprior` or"):
        build_logposterior_fisher(
            theta,
            theta0,
            f,
            logprior=lambda th: 0.0,
            prior_bounds=[(0.0, 1.0), (0.0, 1.0)],
        )


def test_delta_chi2_dali_reduces_to_fisher_when_g_h_zero():
    """Tests that delta_chi2_dali reduces to delta_chi2_fisher when G and H tensors are zero."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.2, -0.1, 0.05])
    f = _spd_fisher(p, seed=8)
    g = np.zeros((p, p, p), dtype=float)
    h = np.zeros((p, p, p, p), dtype=float)

    chi2_f = build_delta_chi2_fisher(theta, theta0, f)
    chi2_d = build_delta_chi2_dali(theta, theta0, f, g, h, convention="delta_chi2")
    assert chi2_d == pytest.approx(chi2_f)


def test_delta_chi2_dali_convention_delta_chi2_matches_formula():
    """Tests that delta_chi2_dali with 'delta_chi2' convention matches manual calculation."""
    p = 3
    theta0 = np.array([0.0, 0.1, -0.2])
    theta = np.array([0.2, -0.1, 0.05])
    f = _spd_fisher(p, seed=9)
    g, h = _toy_tensors(p)

    d = theta - theta0
    quad = float(d @ f @ d)
    g3 = _manual_g3(g, d)
    h4 = _manual_h4(h, d)

    expected = quad + (1.0 / 3.0) * g3 + (1.0 / 12.0) * h4
    got = build_delta_chi2_dali(theta, theta0, f, g, h, convention="delta_chi2")
    assert got == pytest.approx(expected)


def test_delta_chi2_dali_convention_matplotlib_matches_formula():
    """Tests that delta_chi2_dali with 'matplotlib_loglike' convention matches manual calculation."""
    p = 3
    theta0 = np.array([0.0, 0.1, -0.2])
    theta = np.array([0.2, -0.1, 0.05])
    f = _spd_fisher(p, seed=10)
    g, h = _toy_tensors(p)

    d = theta - theta0
    quad = float(d @ f @ d)
    g3 = _manual_g3(g, d)
    h4 = _manual_h4(h, d)

    expected = quad + g3 + 0.25 * h4
    got = build_delta_chi2_dali(theta, theta0, f, g, h, convention="matplotlib_loglike")
    assert got == pytest.approx(expected)


def test_delta_chi2_dali_unknown_convention_raises():
    """Tests that delta_chi2_dali raises on unknown convention."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.ones(p)
    f = _spd_fisher(p, seed=11)
    g = np.zeros((p, p, p))
    with pytest.raises(ValueError, match="Unknown convention"):
        build_delta_chi2_dali(theta, theta0, f, g, None, convention="nope")


def test_logposterior_dali_no_prior_is_minus_half_delta_chi2():
    """Tests that logposterior_dali without prior is -0.5 * delta_chi2_dali."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    f = _spd_fisher(p, seed=12)
    g, h = _toy_tensors(p)

    chi2 = build_delta_chi2_dali(theta, theta0, f, g, h, convention="delta_chi2")
    lp = build_logposterior_dali(theta, theta0, f, g, h, convention="delta_chi2")
    assert lp == pytest.approx(-0.5 * chi2)


def test_logposterior_dali_prior_short_circuit_to_minus_inf():
    """Tests that logposterior_dali returns -inf if logprior is -inf."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    f = _spd_fisher(p, seed=13)
    g, h = _toy_tensors(p)

    def logprior(_: np.ndarray) -> float:
        return -np.inf

    lp = build_logposterior_dali(theta, theta0, f, g, h, logprior=logprior)
    assert lp == -np.inf


def test_logposterior_dali_rejects_prior_spec_and_logprior_together():
    """Tests that logposterior_dali raises if both prior spec and logprior are given."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    f = _spd_fisher(p, seed=14)
    g, h = _toy_tensors(p)

    with pytest.raises(ValueError, match="either `logprior` or"):
        build_logposterior_dali(
            theta,
            theta0,
            f,
            g,
            h,
            logprior=lambda th: 0.0,
            prior_bounds=[(0.0, 1.0)] * p,
        )


def test_logposterior_fisher_prior_bounds_enforced_via_build_prior():
    """Tests that logposterior_fisher enforces prior bounds correctly."""
    p = 2
    theta0 = np.zeros(p)
    f = _spd_fisher(p, seed=15)

    bounds = [(-0.5, 0.5), (-0.5, 0.5)]
    theta_inside = np.array([0.1, -0.2])
    theta_outside = np.array([0.6, 0.0])

    lp_in = build_logposterior_fisher(theta_inside, theta0, f, prior_bounds=bounds)
    assert np.isfinite(lp_in)

    lp_out = build_logposterior_fisher(theta_outside, theta0, f, prior_bounds=bounds)
    assert lp_out == -np.inf
