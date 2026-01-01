"""Unit tests for derivkit.forecasting.integrations.getdist_dali_samples module."""

import numpy as np
import pytest

from derivkit.forecasting.integrations.getdist_dali_samples import (
    dali_to_getdist_emcee,
    dali_to_getdist_importance,
)


def _toy_dali_inputs(p: int = 2):
    """Small consistent (theta0, fisher, G, H) for testing."""
    theta0 = np.zeros(p)
    fisher = np.eye(p)
    g = np.zeros((p, p, p))
    h = np.zeros((p, p, p, p))
    return theta0, fisher, g, h


def test_dali_to_getdist_importance_returns_mcsamples_and_shapes():
    """Tests that dali_to_getdist_importance returns weighted MCSamples with expected shapes."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m = dali_to_getdist_importance(
        theta0,
        fisher,
        g,
        h,
        names=names,
        labels=labels,
        n_samples=2000,
        kernel_scale=1.2,
        seed=123,
    )

    samples = np.asarray(m.samples)
    weights = np.asarray(m.weights)
    loglikes = np.asarray(m.loglikes)

    assert samples.ndim == 2 and samples.shape[1] == 2
    assert weights.shape == (samples.shape[0],)
    assert loglikes.shape == (samples.shape[0],)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)


def test_dali_to_getdist_importance_reproducible_with_seed():
    """Tests that dali_to_getdist_importance is reproducible with a seed."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m1 = dali_to_getdist_importance(
        theta0, fisher, g, h, names=names, labels=labels, n_samples=1500, seed=7
    )
    m2 = dali_to_getdist_importance(
        theta0, fisher, g, h, names=names, labels=labels, n_samples=1500, seed=7
    )

    assert np.allclose(np.asarray(m1.samples), np.asarray(m2.samples))
    assert np.allclose(np.asarray(m1.weights), np.asarray(m2.weights))
    assert np.allclose(np.asarray(m1.loglikes), np.asarray(m2.loglikes))


def test_dali_to_getdist_importance_raises_on_names_labels_mismatch():
    """Tests that dali_to_getdist_importance raises if names and/or labels length != p."""
    theta0, fisher, g, h = _toy_dali_inputs(2)

    with pytest.raises(ValueError, match=r"names must have length p=2"):
        dali_to_getdist_importance(
            theta0, fisher, g, h, names=["a"], labels=["a", "b"], n_samples=10
        )

    with pytest.raises(ValueError, match=r"labels must have length p=2"):
        dali_to_getdist_importance(
            theta0, fisher, g, h, names=["a", "b"], labels=["a"], n_samples=10
        )


def test_dali_to_getdist_importance_sampler_bounds_filters_samples():
    """Tests that sampler_bounds filters proposal samples before posterior evaluation."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(-0.1, 0.1), (-0.1, 0.1)]

    m = dali_to_getdist_importance(
        theta0,
        fisher,
        g,
        h,
        names=names,
        labels=labels,
        n_samples=5000,
        seed=0,
        sampler_bounds=sampler_bounds,
    )

    s = np.asarray(m.samples)
    assert s.shape[0] < 5000
    assert np.all(s[:, 0] >= -0.1) and np.all(s[:, 0] <= 0.1)
    assert np.all(s[:, 1] >= -0.1) and np.all(s[:, 1] <= 0.1)


def test_dali_to_getdist_importance_raises_if_all_rejected_by_sampler_bounds():
    """Tests that dali_to_getdist_importance raises if sampler_bounds reject all samples."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(10.0, 11.0), (None, None)]  # impossible around theta0=0

    with pytest.raises(RuntimeError, match=r"All kernel samples rejected"):
        dali_to_getdist_importance(
            theta0, fisher, g, h, names=names, labels=labels, n_samples=50,
            seed=0, sampler_bounds=sampler_bounds
        )


def test_dali_to_getdist_importance_raises_if_all_rejected_by_logposterior():
    """Tests that dali_to_getdist_importance raises if posterior/prior makes all logpost -inf."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    def logprior(_th: np.ndarray) -> float:
        """A function that rejects all samples and returns -inf."""
        return -np.inf

    with pytest.raises(RuntimeError, match=r"All kernel samples were rejected"):
        dali_to_getdist_importance(
            theta0,
            fisher,
            g,
            h,
            names=names,
            labels=labels,
            n_samples=200,
            seed=1,
            logprior=logprior,
        )


def test_dali_to_getdist_emcee_raises_on_names_labels_mismatch():
    """Tests that dali_to_getdist_emcee raises if names/labels length != p."""
    theta0, fisher, g, h = _toy_dali_inputs(2)

    with pytest.raises(ValueError, match=r"names must have length p=2"):
        dali_to_getdist_emcee(
            theta0,
            fisher,
            g,
            h,
            names=["a"],
            labels=["a", "b"],
            n_steps=20,
            burn=0,
            thin=1,
            n_walkers=8,
        )

    with pytest.raises(ValueError, match=r"labels must have length p=2"):
        dali_to_getdist_emcee(
            theta0,
            fisher,
            g,
            h,
            names=["a", "b"],
            labels=["a"],
            n_steps=20,
            burn=0,
            thin=1,
            n_walkers=8,
        )


def test_dali_to_getdist_emcee_returns_mcsamples_and_loglikes_shapes():
    """Tests that dali_to_getdist_emcee returns MCSamples with consistent sample/loglikes shapes."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m = dali_to_getdist_emcee(
        theta0,
        fisher,
        g,
        h,
        names=names,
        labels=labels,
        n_steps=60,
        burn=10,
        thin=2,
        n_walkers=8,
        init_scale=0.2,
        seed=0,
    )

    samples = np.asarray(m.samples)
    loglikes = np.asarray(m.loglikes)

    # GetDist may internally concatenate list-of-chains into a single array.
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert loglikes.ndim == 1 and loglikes.shape[0] == samples.shape[0]
    assert np.all(np.isfinite(loglikes))


def test_dali_to_getdist_emcee_sampler_bounds_respected():
    """Tests that dali_to_getdist_emcee runs with sampler_bounds and returns valid samples."""
    theta0, fisher, g, h = _toy_dali_inputs(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(-0.2, 0.2), (-0.2, 0.2)]

    m = dali_to_getdist_emcee(
        theta0,
        fisher,
        g,
        h,
        names=names,
        labels=labels,
        n_steps=80,
        burn=10,
        thin=2,
        n_walkers=10,
        init_scale=0.3,
        seed=2,
        sampler_bounds=sampler_bounds,
    )

    s = np.asarray(m.samples)
    ll = np.asarray(m.loglikes)

    assert s.ndim == 2 and s.shape[1] == 2
    assert ll.ndim == 1 and ll.shape[0] == s.shape[0]
    assert np.all(np.isfinite(ll))
