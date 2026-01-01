"""Tests for derivkit.forecasting.integrations.getdist_fisher_samples module."""

import numpy as np
import pytest

from derivkit.forecasting.integrations.getdist_fisher_samples import (
    fisher_to_getdist_gaussiannd,
    fisher_to_getdist_samples,
)
from derivkit.forecasting.integrations.sampling_utils import fisher_to_cov


def test_fisher_to_getdist_gaussiannd_mean_and_cov_and_names_and_label():
    """Tests that fisher_to_getdist_gaussiannd returns GaussianND with expected mean/cov and metadata."""
    theta0 = np.array([1.0, -2.0])
    fisher = np.array([[4.0, 0.0], [0.0, 1.0]])
    names = ["p0", "p1"]
    labels = [r"p_0", r"p_1"]

    g = fisher_to_getdist_gaussiannd(theta0, fisher, names=names, labels=labels, label="X")

    assert np.allclose(np.asarray(g.means[0]), theta0)
    assert np.allclose(np.asarray(g.covs[0]), fisher_to_cov(fisher))
    assert list(g.names) == names
    assert g.label == "X"

    # Labels: GaussianND doesn't expose `.labels` in this getdist version.
    # Check via the parameter names/labels container if present.
    if hasattr(g, "paramNames") and g.paramNames is not None:
        assert [p.name for p in g.paramNames.names] == names
        assert [p.label for p in g.paramNames.names] == labels


def test_fisher_to_getdist_gaussiannd_raises_on_names_labels_mismatch():
    """Tests that fisher_to_getdist_gaussiannd raises if names/labels length != p."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)

    with pytest.raises(ValueError, match=r"`names` must have length 2"):
        fisher_to_getdist_gaussiannd(theta0, fisher, names=["a"], labels=["a", "b"])

    with pytest.raises(ValueError, match=r"`labels` must have length 2"):
        fisher_to_getdist_gaussiannd(theta0, fisher, names=["a", "b"], labels=["a"])


def test_fisher_to_getdist_samples_returns_mcsamples_and_reproducible():
    """Tests that fisher_to_getdist_samples returns MCSamples and is reproducible with seed."""
    theta0 = np.array([0.5, -0.5])
    fisher = np.array([[2.0, 0.0], [0.0, 0.5]])
    names = ["x", "y"]
    labels = ["x", "y"]

    m1 = fisher_to_getdist_samples(
        theta0, fisher, names=names, labels=labels, n_samples=2000, seed=123, store_loglikes=True
    )
    m2 = fisher_to_getdist_samples(
        theta0, fisher, names=names, labels=labels, n_samples=2000, seed=123, store_loglikes=True
    )

    s1 = np.asarray(m1.samples)
    s2 = np.asarray(m2.samples)

    assert s1.shape == (2000, 2)
    assert np.allclose(s1, s2)
    assert m1.loglikes is not None
    assert np.asarray(m1.loglikes).shape == (2000,)


def test_fisher_to_getdist_samples_store_loglikes_false_sets_loglikes_none():
    """Tests that store_loglikes=False returns MCSamples with loglikes=None."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m = fisher_to_getdist_samples(
        theta0, fisher, names=names, labels=labels, n_samples=100, seed=0, store_loglikes=False
    )
    assert m.loglikes is None


def test_fisher_to_getdist_samples_hard_bounds_filters_samples():
    """Tests that hard_bounds filters samples and reduces count."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    # Narrow bounds to force filtering.
    bounds = [(-0.1, 0.1), (-0.1, 0.1)]
    m = fisher_to_getdist_samples(
        theta0,
        fisher,
        names=names,
        labels=labels,
        n_samples=5000,
        seed=0,
        hard_bounds=bounds,
        store_loglikes=True,
    )

    s = np.asarray(m.samples)
    assert s.shape[1] == 2
    assert s.shape[0] < 5000
    assert np.all(s[:, 0] >= -0.1) and np.all(s[:, 0] <= 0.1)
    assert np.all(s[:, 1] >= -0.1) and np.all(s[:, 1] <= 0.1)


def test_fisher_to_getdist_samples_raises_when_both_logprior_and_prior_terms_given():
    """Tests that specifying both logprior and prior_terms/prior_bounds raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    def lp(_th: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError, match=r"Ambiguous prior specification"):
        fisher_to_getdist_samples(
            theta0,
            fisher,
            names=names,
            labels=labels,
            n_samples=10,
            seed=0,
            logprior=lp,
            prior_bounds=[(None, None), (None, None)],
        )


def test_fisher_to_getdist_samples_applies_logprior_and_rejects_outside_support():
    """Tests that a logprior can reject samples and loglikes are computed on survivors."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    # Hard support: keep only points in a small box.
    def logprior(th: np.ndarray) -> float:
        th = np.asarray(th, float)
        if np.all(np.abs(th) <= 0.2):
            return 0.0
        return -np.inf

    m = fisher_to_getdist_samples(
        theta0,
        fisher,
        names=names,
        labels=labels,
        n_samples=5000,
        seed=1,
        logprior=logprior,
        store_loglikes=True,
    )

    s = np.asarray(m.samples)
    assert s.shape[0] < 5000
    assert np.all(np.abs(s) <= 0.2 + 1e-12)
    assert m.loglikes is not None
    assert np.asarray(m.loglikes).shape == (s.shape[0],)


def test_fisher_to_getdist_samples_raises_if_prior_rejects_all():
    """Tests that fisher_to_getdist_samples raises RuntimeError when prior rejects all samples."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    def logprior(_th: np.ndarray) -> float:
        """A function that rejects all samples and returns -inf."""
        return -np.inf

    with pytest.raises(RuntimeError, match=r"All .* rejected"):
        fisher_to_getdist_samples(
            theta0,
            fisher,
            names=names,
            labels=labels,
            n_samples=50,
            seed=0,
            logprior=logprior,
            store_loglikes=True,
        )


def test_fisher_to_getdist_samples_raises_on_names_labels_mismatch():
    """Tests that fisher_to_getdist_samples raises if names/labels length != p."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.eye(2)

    with pytest.raises(ValueError, match=r"`labels` must have length 2"):
        fisher_to_getdist_samples(
            theta0, fisher, names=["a", "b"], labels=["a"], n_samples=10, seed=0
        )


def test_fisher_to_getdist_gaussiannd_tight_rcond_succeeds_and_matches_pinv():
    """Tests that fisher_to_getdist_gaussiannd succeeds when rcond does not truncate modes."""
    theta0 = np.array([0.0, 0.0])
    fisher = np.diag([1.0, 1e-6])
    names = ["a", "b"]
    labels = ["a", "b"]

    g = fisher_to_getdist_gaussiannd(theta0, fisher, names=names, labels=labels, rcond=1e-12)
    assert np.allclose(np.asarray(g.covs[0]), np.linalg.pinv(fisher, rcond=1e-12))
