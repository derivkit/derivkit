"""Unit tests for the derivkit.forecasting.expansions module."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import (
    build_delta_chi2_dali,
    build_delta_chi2_fisher,
    build_logposterior_dali,
    build_logposterior_fisher,
    build_subspace,
)


def _spd_fisher(p: int, seed: int = 0) -> np.ndarray:
    """Returns a random symmetric positive definite Fisher matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(p, p))
    f = a.T @ a
    f += 1e-2 * np.eye(p)
    return f


def _toy_dali_doublet(p: int) -> dict[int, tuple[np.ndarray, ...]]:
    """Returns deterministic introduced-at-order DALI tensors for order 2 testing."""
    f = _spd_fisher(p, seed=0)

    d1 = np.zeros((p, p, p), dtype=float)
    d2 = np.zeros((p, p, p, p), dtype=float)

    for i in range(p):
        d1[i, i, i] = 0.3 * (i + 1)
        for j in range(p):
            d2[i, i, j, j] = 0.2 * (i + 1) * (j + 1)

    return {1: (f,), 2: (d1, d2)}


def _toy_dali_triplet(p: int) -> dict[int, tuple[np.ndarray, ...]]:
    """Returns deterministic introduced-at-order DALI tensors for order 3 testing."""
    dali = _toy_dali_doublet(p)

    t1 = np.zeros((p, p, p, p), dtype=float)
    t2 = np.zeros((p, p, p, p, p), dtype=float)
    t3 = np.zeros((p, p, p, p, p, p), dtype=float)

    for i in range(p):
        t1[i, i, i, i] = 0.05 * (i + 1)
        t2[i, i, i, i, i] = 0.01 * (i + 1)
        t3[i, i, i, i, i, i] = 0.002 * (i + 1)

    dali[3] = (t1, t2, t3)
    return dali


def _manual_d1_3(d1: np.ndarray, d: np.ndarray) -> float:
    """Returns D1[d,d,d] computed via einsum for testing."""
    return float(np.einsum("ijk,i,j,k->", d1, d, d, d))


def _manual_d2_4(d2: np.ndarray, d: np.ndarray) -> float:
    """Returns D2[d,d,d,d] computed via einsum for testing."""
    return float(np.einsum("ijkl,i,j,k,l->", d2, d, d, d, d))


def test_build_subspace_fisher_slices_correct_block() -> None:
    """Tests that build_subspace slices Fisher and theta0 consistently."""
    p = 5
    theta0 = np.arange(p, dtype=float)
    f = np.arange(p * p, dtype=float).reshape(p, p)
    idx = [0, 2, 4]

    out = build_subspace(idx, theta0=theta0, fisher=f)

    assert set(out.keys()) == {"theta0", "fisher"}
    assert np.allclose(out["theta0"], theta0[idx])
    assert np.allclose(out["fisher"], f[np.ix_(idx, idx)])


def test_build_subspace_dali_mode_returns_sliced_dict() -> None:
    """Tests that build_subspace slices theta0 and every tensor in dict-form DALI."""
    p = 6
    theta0 = np.linspace(-0.5, 0.5, p)
    dali = _toy_dali_triplet(p)

    idx = [1, 4, 5]
    out = build_subspace(idx, theta0=theta0, dali=dali)

    assert set(out.keys()) == {"theta0", "dali"}
    assert out["theta0"].shape == (len(idx),)

    dali_sub = out["dali"]
    assert set(dali_sub.keys()) == {1, 2, 3}

    f_sub = dali_sub[1][0]
    d1_sub, d2_sub = dali_sub[2]
    t1_sub, t2_sub, t3_sub = dali_sub[3]

    n = len(idx)
    assert f_sub.shape == (n, n)
    assert d1_sub.shape == (n, n, n)
    assert d2_sub.shape == (n, n, n, n)
    assert t1_sub.shape == (n, n, n, n)
    assert t2_sub.shape == (n, n, n, n, n)
    assert t3_sub.shape == (n, n, n, n, n, n)


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


def test_delta_chi2_dali_reduces_to_fisher_when_higher_tensors_zero() -> None:
    """Tests that delta_chi2_dali reduces to Fisher when higher-order tensors are zero."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.2, -0.1, 0.05])
    f = _spd_fisher(p, seed=8)

    d1 = np.zeros((p, p, p), dtype=float)
    d2 = np.zeros((p, p, p, p), dtype=float)
    dali = {1: (f,), 2: (d1, d2)}

    chi2_f = build_delta_chi2_fisher(theta, theta0, f)
    chi2_d = build_delta_chi2_dali(theta, theta0, dali, forecast_order=2)
    assert chi2_d == pytest.approx(chi2_f)


def test_delta_chi2_dali_order2_matches_formula() -> None:
    """Tests that build_delta_chi2_dali(order=2) matches the documented coefficients."""
    p = 3
    theta0 = np.array([0.0, 0.1, -0.2])
    theta = np.array([0.2, -0.1, 0.05])

    dali = _toy_dali_doublet(p)
    f = dali[1][0]
    d1, d2 = dali[2]

    d = theta - theta0
    quad = float(d @ f @ d)
    d1_3 = _manual_d1_3(d1, d)
    d2_4 = _manual_d2_4(d2, d)

    expected = quad + (1.0 / 3.0) * d1_3 + (1.0 / 12.0) * d2_4
    got = build_delta_chi2_dali(theta, theta0, dali, forecast_order=2)
    assert got == pytest.approx(expected)


def test_logposterior_dali_no_prior_is_minus_half_delta_chi2() -> None:
    """Tests that logposterior_dali without prior is -0.5 * delta_chi2_dali."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    dali = _toy_dali_doublet(p)

    chi2 = build_delta_chi2_dali(theta, theta0, dali, forecast_order=2)
    lp = build_logposterior_dali(theta, theta0, dali, forecast_order=2)
    assert lp == pytest.approx(-0.5 * chi2)


def test_logposterior_dali_prior_short_circuit_to_minus_inf() -> None:
    """Tests that logposterior_dali returns -inf if logprior is -inf."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    dali = _toy_dali_doublet(p)

    def logprior(_: np.ndarray) -> float:
        return -np.inf

    lp = build_logposterior_dali(theta, theta0, dali, logprior=logprior)
    assert lp == -np.inf


def test_logposterior_dali_rejects_prior_spec_and_logprior_together() -> None:
    """Tests that logposterior_dali raises if both prior spec and logprior are given."""
    p = 3
    theta0 = np.zeros(p)
    theta = np.array([0.1, 0.2, -0.1])
    dali = _toy_dali_doublet(p)

    with pytest.raises(ValueError, match="either `logprior` or"):
        build_logposterior_dali(
            theta,
            theta0,
            dali,
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


def test_build_subspace_fisher_only_includes_theta0_when_provided():
    """Tests that build_subspace includes theta0 subvector when provided in Fisher-only mode."""
    p = 5
    theta0 = np.linspace(0.0, 1.0, p)
    f = _spd_fisher(p, seed=21)
    idx = [4, 0, 2]
    out = build_subspace(idx, fisher=f, theta0=theta0)
    assert set(out.keys()) == {"fisher", "theta0"}
    assert out["theta0"].shape == (len(idx),)
    assert np.allclose(out["theta0"], theta0[idx])
    assert np.allclose(out["fisher"], f[np.ix_(idx, idx)])


def test_build_subspace_rejects_non_integer_indices() -> None:
    """Tests that build_subspace raises TypeError when idx contains non-integers."""
    p = 4
    theta0 = np.zeros(p)
    f = _spd_fisher(p, seed=24)

    with pytest.raises(TypeError, match="idx must contain integer"):
        build_subspace([0, 1.5], fisher=f, theta0=theta0)  # type: ignore[list-item]


def test_build_subspace_fisher_only_slices_theta0_and_fisher():
    """Tests that build_subspace returns sliced theta0 and Fisher in Fisher-only mode."""
    p = 6
    theta0 = np.linspace(0.0, 1.0, p)
    f = np.arange(p * p, dtype=float).reshape(p, p)
    idx = [5, 1, 3]

    out = build_subspace(idx, fisher=f, theta0=theta0)

    assert set(out.keys()) == {"theta0", "fisher"}
    assert out["theta0"].shape == (len(idx),)
    assert out["fisher"].shape == (len(idx), len(idx))
    assert np.allclose(out["theta0"], theta0[idx])
    assert np.allclose(out["fisher"], f[np.ix_(idx, idx)])


def test_build_subspace_rejects_out_of_bounds_indices_fisher_only() -> None:
    """Tests that build_subspace raises IndexError on out-of-bounds idx in Fisher-only mode."""
    p = 4
    theta0 = np.zeros(p)
    f = _spd_fisher(p, seed=25)

    with pytest.raises(IndexError, match="out-of-bounds"):
        build_subspace([0, 4], fisher=f, theta0=theta0)


def test_build_subspace_rejects_out_of_bounds_indices_dali_mode() -> None:
    """Tests that build_subspace raises IndexError on out-of-bounds idx in DALI mode."""
    p = 4
    theta0 = np.zeros(p)
    dali = _toy_dali_doublet(p)

    with pytest.raises(IndexError, match="out-of-bounds"):
        build_subspace([3, 4], theta0=theta0, dali=dali)


def test_build_subspace_requires_exactly_one_of_fisher_or_dali() -> None:
    """Tests that build_subspace raises if both or neither of fisher/dali are provided."""
    p = 3
    theta0 = np.zeros(p)
    f = _spd_fisher(p, seed=0)
    dali = _toy_dali_doublet(p)

    with pytest.raises(ValueError, match="exactly one of"):
        build_subspace([0, 1], theta0=theta0)

    with pytest.raises(ValueError, match="exactly one of"):
        build_subspace([0, 1], theta0=theta0, fisher=f, dali=dali)


def test_build_subspace_raises_on_non_square_fisher():
    """Tests that build_subspace raises ValueError when Fisher is not square."""
    theta0 = np.zeros(2)
    f = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError):
        build_subspace([0], fisher=f, theta0=theta0)


def test_delta_chi2_dali_order3_matches_formula() -> None:
    """Tests that build_delta_chi2_dali(order=3) includes the triplet coefficients."""
    p = 3
    theta0 = np.array([0.0, 0.1, -0.2])
    theta = np.array([0.2, -0.1, 0.05])

    dali = _toy_dali_triplet(p)
    f = dali[1][0]
    d1, d2 = dali[2]
    t1, t2, t3 = dali[3]

    d = theta - theta0
    quad = float(d @ f @ d)

    d1_3 = float(np.einsum("ijk,i,j,k->", d1, d, d, d))
    d2_4 = float(np.einsum("ijkl,i,j,k,l->", d2, d, d, d, d))
    t1_4 = float(np.einsum("ijkl,i,j,k,l->", t1, d, d, d, d))
    t2_5 = float(np.einsum("ijklm,i,j,k,l,m->", t2, d, d, d, d, d))
    t3_6 = float(np.einsum("ijklmn,i,j,k,l,m,n->", t3, d, d, d, d, d, d))

    expected = (
        quad
        + (1.0 / 3.0) * d1_3
        + (1.0 / 12.0) * d2_4
        + (1.0 / 3.0) * t1_4
        + (1.0 / 6.0) * t2_5
        + (1.0 / 36.0) * t3_6
    )

    got = build_delta_chi2_dali(theta, theta0, dali, forecast_order=3)
    assert got == pytest.approx(expected)


def test_delta_chi2_dali_rejects_forecast_order_1() -> None:
    """Tests that build_delta_chi2_dali rejects forecast_order=1."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.ones(p)
    dali = _toy_dali_doublet(p)

    with pytest.raises(ValueError, match="requires forecast_order >= 2"):
        build_delta_chi2_dali(theta, theta0, dali, forecast_order=1)


def test_delta_chi2_dali_rejects_missing_required_keys() -> None:
    """Tests that build_delta_chi2_dali rejects dicts missing required tensor keys."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.ones(p)
    f = _spd_fisher(p, seed=0)

    with pytest.raises(ValueError, match="must contain keys 1 and 2"):
        build_delta_chi2_dali(theta, theta0, {1: (f,)}, forecast_order=2)


def test_delta_chi2_dali_rejects_unsupported_forecast_order() -> None:
    """Tests that build_delta_chi2_dali rejects unsupported forecast_order values."""
    p = 2
    theta0 = np.zeros(p)
    theta = np.ones(p)
    dali = _toy_dali_doublet(p)

    with pytest.raises(ValueError, match=r"forecast_order=4|not supported|Supported values"):
        build_delta_chi2_dali(theta, theta0, dali, forecast_order=4)
