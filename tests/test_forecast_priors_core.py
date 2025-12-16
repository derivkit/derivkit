"""Unit tests for derivkit.forecasting.priors.core module."""

import numpy as np
import pytest

from derivkit.forecasting.priors.core import (
    build_prior,
    prior_beta,
    prior_gaussian,
    prior_gaussian_diag,
    prior_gaussian_mixture,
    prior_half_cauchy,
    prior_half_normal,
    prior_jeffreys,
    prior_log_normal,
    prior_log_uniform,
    prior_none,
    prior_uniform,
)


def test_prior_none_returns_zero():
    """Tests that prior_none always returns 0.0 regardless of input."""
    p = prior_none()
    assert p(np.array([1.0, 2.0])) == 0.0
    assert p(np.array([])) == 0.0


def test_prior_uniform_hard_bounds():
    """Tests that prior_uniform enforces hard bounds correctly."""
    p = prior_uniform(bounds=[(0.0, 1.0), (None, 2.0)])
    assert np.isfinite(p(np.array([0.0, -100.0])))
    assert np.isfinite(p(np.array([1.0, 2.0])))
    assert p(np.array([-1.0, 0.0])) == -np.inf
    assert p(np.array([0.5, 3.0])) == -np.inf


def test_prior_gaussian_matches_quadratic_form_with_inv_cov():
    """Tests that prior_gaussian with inv_cov matches the expected quadratic form."""
    mu = np.array([1.0, -1.0])
    inv_cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    p = prior_gaussian(mean=mu, inv_cov=inv_cov)

    th = np.array([2.0, 1.0])
    d = th - mu
    expected = -0.5 * (d @ inv_cov @ d)
    assert p(th) == pytest.approx(expected)


def test_prior_gaussian_requires_exactly_one_of_cov_or_inv_cov():
    """Tests that prior_gaussian raises ValueError if neither or both cov and inv_cov are provided."""
    mu = np.zeros(2)
    cov = np.eye(2)
    inv_cov = np.eye(2)
    with pytest.raises(ValueError):
        prior_gaussian(mean=mu, cov=cov, inv_cov=inv_cov)
    with pytest.raises(ValueError):
        prior_gaussian(mean=mu)


def test_prior_gaussian_shape_checks():
    """Tests that prior_gaussian raises ValueError for invalid mean or inv_cov shapes."""
    mu = np.zeros(2)
    with pytest.raises(ValueError):
        prior_gaussian(mean=np.zeros((2, 1)), inv_cov=np.eye(2))
    with pytest.raises(ValueError):
        prior_gaussian(mean=mu, inv_cov=np.eye(3))  # wrong inv_cov shape

    p = prior_gaussian(mean=mu, inv_cov=np.eye(2))
    with pytest.raises(ValueError):
        p(np.array([1.0, 2.0, 3.0]))  # theta wrong length


def test_prior_gaussian_diag_matches_independent_quadratic():
    """Tests that prior_gaussian_diag matches the expected independent quadratic form."""
    mu = np.array([0.0, 1.0, -2.0])
    sig = np.array([2.0, 0.5, 1.0])
    p = prior_gaussian_diag(mean=mu, sigma=sig)

    th = np.array([2.0, 0.0, -1.0])
    expected = -0.5 * np.sum(((th - mu) / sig) ** 2)
    assert p(th) == pytest.approx(expected)


def test_prior_gaussian_diag_validates_sigma_and_shapes():
    """Tests that prior_gaussian_diag raises ValueError for invalid sigma values or shapes."""
    mu = np.zeros(2)
    with pytest.raises(ValueError):
        prior_gaussian_diag(mean=mu, sigma=np.ones((2, 1)))
    with pytest.raises(ValueError):
        prior_gaussian_diag(mean=mu, sigma=np.array([1.0, -1.0]))


def test_log_uniform_and_jeffreys_are_same_shape():
    """Tests that prior_log_uniform and prior_jeffreys give the same result."""
    th = np.array([2.0, 1.0])
    p1 = prior_log_uniform(index=0)
    p2 = prior_jeffreys(index=0)
    assert p1(th) == pytest.approx(-np.log(2.0))
    assert p2(th) == pytest.approx(-np.log(2.0))


def test_log_uniform_domain_gate():
    """Tests that prior_log_uniform returns -inf for non-positive inputs."""
    p = prior_log_uniform(index=0)
    assert p(np.array([1.0])) == pytest.approx(0.0)
    assert p(np.array([0.0])) == -np.inf
    assert p(np.array([-1.0])) == -np.inf


def test_half_normal_domain_and_sigma_validation():
    """Tests that prior_half_normal validates sigma and enforces domain correctly."""
    with pytest.raises(ValueError):
        prior_half_normal(index=0, sigma=0.0)
    with pytest.raises(ValueError):
        prior_half_normal(index=0, sigma=-1.0)

    p = prior_half_normal(index=0, sigma=2.0)
    assert p(np.array([-1.0])) == -np.inf
    assert p(np.array([0.0])) == pytest.approx(0.0)
    assert p(np.array([2.0])) == pytest.approx(-0.5 * (2.0 / 2.0) ** 2)


def test_half_cauchy_domain_and_scale_validation():
    """Tests that prior_half_cauchy validates scale and enforces domain correctly."""
    with pytest.raises(ValueError):
        prior_half_cauchy(index=0, scale=0.0)
    with pytest.raises(ValueError):
        prior_half_cauchy(index=0, scale=-3.0)

    p = prior_half_cauchy(index=0, scale=2.0)
    assert p(np.array([-1.0])) == -np.inf
    assert p(np.array([0.0])) == pytest.approx(0.0)
    t = 2.0 / 2.0
    assert p(np.array([2.0])) == pytest.approx(-np.log1p(t * t))


def test_log_normal_domain_and_sigma_validation():
    """Tests that prior_log_normal validates sigma_log and enforces domain correctly."""
    with pytest.raises(ValueError):
        prior_log_normal(index=0, mean_log=0.0, sigma_log=0.0)
    with pytest.raises(ValueError):
        prior_log_normal(index=0, mean_log=0.0, sigma_log=-1.0)

    p = prior_log_normal(index=0, mean_log=0.0, sigma_log=1.0)
    assert p(np.array([0.0])) == -np.inf
    x = np.e
    lx = np.log(x)
    expected = -0.5 * ((lx - 0.0) / 1.0) ** 2 - lx
    assert p(np.array([x])) == pytest.approx(expected)


def test_beta_domain_and_param_validation():
    """Tests that prior_beta validates alpha/beta and enforces domain correctly."""
    with pytest.raises(ValueError):
        prior_beta(index=0, alpha=0.0, beta=1.0)
    with pytest.raises(ValueError):
        prior_beta(index=0, alpha=1.0, beta=0.0)

    p = prior_beta(index=0, alpha=2.0, beta=5.0)
    assert p(np.array([0.0])) == -np.inf
    assert p(np.array([1.0])) == -np.inf

    x = 0.25
    expected = (2.0 - 1.0) * np.log(x) + (5.0 - 1.0) * np.log1p(-x)
    assert p(np.array([x])) == pytest.approx(expected)


def test_gaussian_mixture_matches_manual_logsumexp_for_equal_covs():
    """Tests that prior_gaussian_mixture matches manual log-sum-exp calculation for equal covariances."""
    mus = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    w = np.array([0.25, 0.75])

    p = prior_gaussian_mixture(means=mus, covs=covs, weights=w, include_component_norm=True)

    th = np.array([1.0])
    lw = np.log(w / np.sum(w))
    # per-component log norm = -0.5 log|C| = -0.5 log(1)=0
    vals = np.array([
        lw[0] - 0.5 * ((1.0 - 0.0) ** 2) / 1.0,
        lw[1] - 0.5 * ((1.0 - 2.0) ** 2) / 1.0,
    ])
    m = np.max(vals)
    expected = m + np.log(np.sum(np.exp(vals - m)))

    assert p(th) == pytest.approx(expected)


def test_gaussian_mixture_weight_validation():
    """Tests that prior_gaussian_mixture raises ValueError for invalid weights or missing weights/log_weights."""
    mus = np.array([[0.0], [1.0]])
    covs = np.array([[[1.0]], [[1.0]]])

    with pytest.raises(ValueError):
        prior_gaussian_mixture(means=mus, covs=covs)

    with pytest.raises(ValueError):
        prior_gaussian_mixture(means=mus, covs=covs, weights=np.array([1.0, -1.0]))

    with pytest.raises(ValueError):
        prior_gaussian_mixture(means=mus, covs=covs, weights=np.array([0.0, 0.0]))

    with pytest.raises(ValueError):
        prior_gaussian_mixture(means=mus, covs=covs, weights=np.array([1.0, 1.0]), log_weights=np.array([0.0, 0.0]))


def test_gaussian_mixture_allows_zero_weight_component():
    """Tests that prior_gaussian_mixture handles zero-weight components correctly."""
    mus = np.array([[0.0], [10.0]])
    covs = np.array([[[1.0]], [[1.0]]])

    p = prior_gaussian_mixture(means=mus, covs=covs, weights=np.array([1.0, 0.0]))
    th = np.array([0.0])
    assert np.isfinite(p(th))


def test_gaussian_mixture_include_component_norm_changes_relative_weight_when_covs_differ():
    """Tests that prior_gaussian_mixture's include_component_norm flag affects results when covariances differ."""
    mus = np.array([[0.0], [0.0]])
    covs = np.array([[[1.0]], [[4.0]]])

    th = np.array([0.0])
    w = np.array([0.5, 0.5])

    p_norm = prior_gaussian_mixture(means=mus, covs=covs, weights=w, include_component_norm=True)
    p_shape = prior_gaussian_mixture(means=mus, covs=covs, weights=w, include_component_norm=False)

    # at theta=mean, quadratic terms are 0; only weights + component_norm differ
    # so these two should not be equal (unless bug)
    assert p_norm(th) != pytest.approx(p_shape(th))


def test_build_prior_empty_terms_no_bounds_is_flat():
    """Tests that build_prior with no terms and no bounds returns a flat prior."""
    p = build_prior()
    assert p(np.array([123.0, -5.0])) == 0.0


def test_build_prior_empty_terms_with_bounds_is_uniform():
    """Tests that build_prior with no terms but with bounds returns a uniform prior within bounds."""
    p = build_prior(bounds=[(0.0, 1.0)])
    assert np.isfinite(p(np.array([0.5])))
    assert p(np.array([-1.0])) == -np.inf
    assert p(np.array([2.0])) == -np.inf


def test_build_prior_tuple_term_and_global_bounds():
    """Tests that build_prior with a tuple term and global bounds works correctly."""
    mu = np.array([0.0, 0.0])
    sig = np.array([1.0, 2.0])

    p = build_prior(
        terms=[("gaussian_diag", {"mean": mu, "sigma": sig})],
        bounds=[(None, None), (0.0, None)],
    )

    # inside bounds it should be finite and equal to the gaussian_diag value
    expected = -0.5 * ((0.0 - 0.0) ** 2 / 1.0**2 + (1.0 - 0.0) ** 2 / 2.0**2)
    assert p(np.array([0.0, 1.0])) == pytest.approx(expected)
    assert p(np.array([0.0, -1.0])) == -np.inf


def test_build_prior_dict_term_with_term_bounds():
    """Tests that build_prior with a dict term and term-specific bounds works correctly."""
    p = build_prior(
        terms=[
            {"name": "beta", "params": {"index": 0, "alpha": 2.0, "beta": 2.0}, "bounds": [(0.0, 1.0)]}
        ]
    )
    assert np.isfinite(p(np.array([0.5])))
    assert p(np.array([-0.1])) == -np.inf
    assert p(np.array([1.1])) == -np.inf


def test_build_prior_uniform_special_case_accepts_bounds_in_params_or_top_level_not_both():
    """Tests that build_prior handles uniform prior bounds correctly in different specifications."""
    p1 = build_prior(terms=[("uniform", {"bounds": [(0.0, 1.0)]})])
    assert np.isfinite(p1(np.array([0.25])))
    assert p1(np.array([2.0])) == -np.inf

    p2 = build_prior(terms=[{"name": "uniform", "params": {}, "bounds": [(0.0, 1.0)]}])
    assert np.isfinite(p2(np.array([0.25])))
    assert p2(np.array([2.0])) == -np.inf
    with pytest.raises(ValueError):
        build_prior(terms=[{"name": "uniform", "params": {"bounds": [(0.0, 1.0)]}, "bounds": [(0.0, 1.0)]}])


def test_build_prior_rejects_unknown_prior_name_and_bad_term_format():
    """Tests that build_prior raises errors for unknown prior names and invalid term formats."""
    with pytest.raises(ValueError):
        build_prior(terms=[("not_a_prior", {})])

    with pytest.raises(TypeError):
        build_prior(terms=[("gaussian_diag", "not_a_dict")])

    with pytest.raises(TypeError):
        build_prior(terms=[("gaussian_diag", {}) , ("extra", {}, "oops")])  # wrong tuple length


def test_build_prior_global_bounds_apply_last():
    """Tests that build_prior global bounds are applied after term-specific bounds."""
    p = build_prior(
        terms=[("gaussian_diag", {"mean": np.array([0.0]), "sigma": np.array([1.0])})],
        bounds=[(0.0, 1.0)],
    )
    assert np.isfinite(p(np.array([0.5])))
    assert p(np.array([-0.5])) == -np.inf
    assert p(np.array([2.0])) == -np.inf
