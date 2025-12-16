"""Unit tests for sampling utilities in the forecasting integration module."""

import numpy as np
import pytest

from derivkit.forecasting.integrations.sampling_utils import (
    apply_hard_bounds_mask,
    fisher_to_cov,
    init_walkers_from_fisher,
    log_gaussian_proposal,
    proposal_cov_from_fisher,
    proposal_samples_from_fisher,
    stabilized_cholesky,
)


def test_proposal_cov_from_fisher_scales_pinv():
    """Tests that proposal_cov_from_fisher returns the correct covariance matrix."""
    fisher = np.array([[4.0, 0.0], [0.0, 1.0]])
    cov1 = proposal_cov_from_fisher(fisher, proposal_scale=1.0)
    cov2 = proposal_cov_from_fisher(fisher, proposal_scale=2.0)

    assert cov1.shape == (2, 2)
    assert np.allclose(cov1, np.linalg.pinv(fisher))
    assert np.allclose(cov2, 4.0 * cov1)


def test_proposal_cov_from_fisher_singular_fisher_uses_pinv():
    """Tests that proposal_cov_from_fisher handles singular Fisher matrices."""
    fisher = np.array([[1.0, 1.0], [1.0, 1.0]])
    cov = proposal_cov_from_fisher(fisher, proposal_scale=1.0)
    assert cov.shape == (2, 2)
    # Moore-Penrose property: fisher @ cov @ fisher == fisher
    assert np.allclose(fisher @ cov @ fisher, fisher, atol=1e-10, rtol=1e-10)


def test_stabilized_cholesky_rejects_non_square():
    """Tests that stabilized_cholesky raises ValueError for non-square matrices."""
    with pytest.raises(ValueError):
        stabilized_cholesky(np.ones((2, 3)))


def test_stabilized_cholesky_factor_reconstructs_cov_for_pd_matrix():
    """Tests that stabilized_cholesky returns a valid Cholesky factor for PD matrices."""
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    lower_triangle = stabilized_cholesky(cov)
    assert lower_triangle.shape == (2, 2)
    assert np.allclose(lower_triangle, np.tril(lower_triangle))
    assert np.allclose(lower_triangle @ lower_triangle.T, cov, rtol=1e-10, atol=1e-10)


def test_stabilized_cholesky_succeeds_for_near_singular_pd():
    """Tests that stabilized_cholesky works for near-singular positive definite matrices."""
    cov = np.array([[1.0, 0.0], [0.0, 1e-16]])
    lower_triangle = stabilized_cholesky(cov)
    assert np.all(np.isfinite(lower_triangle))
    assert np.all(np.diag(lower_triangle) > 0.0)


def test_proposal_samples_shape_and_reproducibility():
    """Tests that proposal_samples_from_fisher returns samples of correct shape and is reproducible with a seed."""
    theta0 = np.array([1.0, -1.0])
    fisher = np.array([[2.0, 0.0], [0.0, 0.5]])

    s1 = proposal_samples_from_fisher(theta0, fisher, nsamp=5, proposal_scale=1.5, seed=123)
    s2 = proposal_samples_from_fisher(theta0, fisher, nsamp=5, proposal_scale=1.5, seed=123)
    s3 = proposal_samples_from_fisher(theta0, fisher, nsamp=5, proposal_scale=1.5, seed=124)

    assert s1.shape == (5, 2)
    assert np.allclose(s1, s2)
    assert not np.allclose(s1, s3)


def test_proposal_samples_mean_close_to_theta0_for_many_draws():
    """Tests that the mean of many proposal samples is close to theta0."""
    theta0 = np.array([0.2, -0.7])
    fisher = np.array([[3.0, 0.0], [0.0, 4.0]])
    nsamp = 40000

    s = proposal_samples_from_fisher(theta0, fisher, nsamp=nsamp, proposal_scale=1.0, seed=0)
    m = np.mean(s, axis=0)
    assert np.allclose(m, theta0, atol=5e-3, rtol=0)


def test_apply_hard_bounds_mask_none_returns_same_object():
    """Tests that apply_hard_bounds_mask returns the same samples when bounds are None."""
    samples = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = apply_hard_bounds_mask(samples, None)
    assert out is samples


def test_apply_hard_bounds_mask_filters_and_keeps_shape():
    """Tests that apply_hard_bounds_mask correctly filters samples based on bounds."""
    samples = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
        [-1.0, 0.5],
        [0.2, 10.0],
    ])
    bounds = [(0.0, None), (None, 3.0)]
    out = apply_hard_bounds_mask(samples, bounds)
    assert out.shape[1] == 2
    assert np.all(out[:, 0] >= 0.0)
    assert np.all(out[:, 1] <= 3.0)
    assert np.allclose(out, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_apply_hard_bounds_mask_returns_empty_if_all_rejected():
    """Tests that apply_hard_bounds_mask returns an empty array if all samples are rejected."""
    samples = np.array([[0.0, 0.0], [0.1, -0.1]])
    hard_bounds = [(10.0, 11.0), (None, None)]

    out = apply_hard_bounds_mask(samples, hard_bounds)

    assert out.shape == (0, 2)


def test_log_gaussian_proposal_peak_at_theta0_for_identity_cov():
    """Tests that log_gaussian_proposal peaks at theta0 for identity covariance."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    samples = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, -2.0],
    ])

    logq = log_gaussian_proposal(samples, theta0, fisher, proposal_scale=1.0)
    assert logq.shape == (3,)
    assert logq[0] > logq[1]
    assert logq[0] > logq[2]


def test_log_gaussian_proposal_matches_manual_for_simple_case():
    """Tests that log_gaussian_proposal matches manual calculation for a simple case."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    samples = np.array([[1.0, 2.0]])

    p = 2
    cov = np.eye(2) + 1e-12 * np.eye(2)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    inv_cov = np.linalg.inv(cov)
    d = samples - theta0[None, :]
    quad = (d @ inv_cov @ d.T).item()  # 1x2 @ 2x2 @ 2x1

    expected = -0.5 * (quad + p * np.log(2.0 * np.pi) + logdet)

    got = log_gaussian_proposal(samples, theta0, fisher, proposal_scale=1.0)[0]
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_init_walkers_no_bounds_matches_sampling_shape_and_seed_reproducible():
    """Tests that init_walkers_from_fisher without bounds returns correct shape and is reproducible."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)

    w1 = init_walkers_from_fisher(theta0, fisher, nwalkers=10, init_scale=0.1, seed=42, hard_bounds=None)
    w2 = init_walkers_from_fisher(theta0, fisher, nwalkers=10, init_scale=0.1, seed=42, hard_bounds=None)

    assert w1.shape == (10, 2)
    assert np.allclose(w1, w2)


def test_init_walkers_with_bounds_all_within_bounds():
    """Tests that init_walkers_from_fisher with bounds returns walkers within the specified bounds."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]

    walkers = init_walkers_from_fisher(theta0, fisher, nwalkers=20, init_scale=0.2, seed=1, hard_bounds=bounds)
    assert walkers.shape == (20, 2)
    assert np.all(walkers[:, 0] >= -0.5) and np.all(walkers[:, 0] <= 0.5)
    assert np.all(walkers[:, 1] >= -0.5) and np.all(walkers[:, 1] <= 0.5)


def test_init_walkers_raises_when_bounds_impossible_or_too_tight():
    """Tests that init_walkers_from_fisher raises RuntimeError when bounds reject all samples."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    bounds = [(10.0, 10.1), (None, None)]

    with pytest.raises(RuntimeError, match=r"Failed to initialize emcee walkers within hard_bounds"):
        init_walkers_from_fisher(
            theta0, fisher, nwalkers=10, init_scale=1e-6, seed=0, hard_bounds=bounds
        )


def test_fisher_to_cov_matches_pinv_default_rcond():
    """Tests that fisher_to_cov matches np.linalg.pinv when rcond is None."""
    fisher = np.array([[4.0, 1.0], [1.0, 2.0]])
    cov = fisher_to_cov(fisher, rcond=None)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, np.linalg.pinv(fisher))


def test_fisher_to_cov_uses_rcond_when_provided():
    """Tests that fisher_to_cov passes rcond through to np.linalg.pinv."""
    fisher = np.diag([1.0, 1e-12])
    cov_loose = fisher_to_cov(fisher, rcond=1e-6)
    cov_tight = fisher_to_cov(fisher, rcond=1e-15)

    assert cov_loose.shape == (2, 2)
    assert cov_tight.shape == (2, 2)

    assert np.allclose(cov_loose, np.linalg.pinv(fisher, rcond=1e-6))
    assert np.allclose(cov_tight, np.linalg.pinv(fisher, rcond=1e-15))
    assert not np.allclose(cov_loose, cov_tight)


def test_fisher_to_cov_rejects_non_square():
    """Tests that fisher_to_cov raises ValueError for non-square inputs."""
    with pytest.raises(ValueError):
        fisher_to_cov(np.ones((2, 3)))


def test_apply_hard_bounds_mask_raises_on_length_mismatch():
    """Tests that apply_hard_bounds_mask raises ValueError if bounds length != p."""
    samples = np.zeros((5, 2))
    bounds = [(0.0, 1.0)]  # length 1 but p=2
    with pytest.raises(ValueError, match=r"hard_bounds must have length p=2"):
        apply_hard_bounds_mask(samples, bounds)


def test_stabilized_cholesky_succeeds_for_singular_psd_matrix():
    """Tests that stabilized_cholesky succeeds for a singular PSD covariance via jitter."""
    cov = np.array([[1.0, 0.0], [0.0, 0.0]])
    lower_triangle = stabilized_cholesky(cov)
    assert lower_triangle.shape == (2, 2)
    assert np.all(np.isfinite(lower_triangle))
    assert np.all(np.diag(lower_triangle) > 0.0)

    p = cov.shape[0]
    jitter = 1e-12 * float(np.trace(cov)) / max(p, 1)
    cov_reg = cov + jitter * np.eye(p)
    assert np.allclose(lower_triangle @ lower_triangle.T, cov_reg, rtol=1e-10, atol=1e-10)


def test_log_gaussian_proposal_invariant_under_joint_shift():
    """Tests that log_gaussian_proposal is invariant when shifting theta0 and samples together."""
    theta0 = np.array([0.3, -0.4])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    samples = np.array([[0.3, -0.4], [1.0, 0.2], [-0.5, 1.1]])

    logq1 = log_gaussian_proposal(samples, theta0, fisher, proposal_scale=1.2)

    shift = np.array([10.0, -7.0])
    logq2 = log_gaussian_proposal(samples + shift[None, :], theta0 + shift, fisher, proposal_scale=1.2)

    assert logq1.shape == (3,)
    assert np.allclose(logq1, logq2, rtol=1e-12, atol=1e-12)


def test_init_walkers_with_bounds_returns_exact_count_even_with_rejection():
    """Tests that init_walkers_from_fisher returns exactly nwalkers under bounds with some rejection."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)

    # Pick bounds/scale so that:
    # - some draws are rejected (so we exercise the filtering path),
    # - but it's extremely unlikely that an entire batch is rejected (since that would raise).
    bounds = [(-0.3, 0.3), (-0.3, 0.3)]

    walkers = init_walkers_from_fisher(
        theta0,
        fisher,
        nwalkers=25,
        init_scale=0.5,
        seed=123,
        hard_bounds=bounds,
    )
    assert walkers.shape == (25, 2)
    assert np.all(walkers[:, 0] >= -0.3) and np.all(walkers[:, 0] <= 0.3)
    assert np.all(walkers[:, 1] >= -0.3) and np.all(walkers[:, 1] <= 0.3)


def test_init_walkers_raises_when_bounds_impossible_message():
    """Tests that init_walkers_from_fisher raises RuntimeError when bounds reject all samples."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    bounds = [(10.0, 10.1), (None, None)]

    with pytest.raises(RuntimeError, match=r"Failed to initialize emcee walkers within hard_bounds"):
        init_walkers_from_fisher(
            theta0,
            fisher,
            nwalkers=10,
            init_scale=1e-6,
            seed=0,
            hard_bounds=bounds,
        )
