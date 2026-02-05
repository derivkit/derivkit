"""Unit tests for derivkit.forecasting.getdist_dali_samples module."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.forecasting.getdist_dali_samples import (
    dali_to_getdist_emcee,
    dali_to_getdist_importance,
)


def _toy_dali_dict(p: int = 2) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, ...]]]:
    """Creates toy (theta0, dali) inputs for testing in dict[multiplet] form up to order 3."""
    theta0 = np.zeros(p, dtype=float)

    # Fisher
    f = np.eye(p, dtype=float)

    # Doublet tensors (introduced at order 2)
    d1 = np.zeros((p, p, p), dtype=float)          # ndim=3
    d2 = np.zeros((p, p, p, p), dtype=float)       # ndim=4

    # Triplet tensors (introduced at order 3)
    t1 = np.zeros((p, p, p, p), dtype=float)       # ndim=4
    t2 = np.zeros((p, p, p, p, p), dtype=float)    # ndim=5
    t3 = np.zeros((p, p, p, p, p, p), dtype=float) # ndim=6

    dali = {
        1: (f,),
        2: (d1, d2),
        3: (t1, t2, t3),
    }
    return theta0, dali


def test_dali_to_getdist_importance_returns_mcsamples_and_shapes():
    """Tests that dali_to_getdist_importance returns weighted MCSamples with expected shapes."""
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m = dali_to_getdist_importance(
        theta0,
        dali,
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
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m1 = dali_to_getdist_importance(
        theta0,
        dali,
        names=names,
        labels=labels,
        n_samples=1500,
        seed=7,
    )
    m2 = dali_to_getdist_importance(
        theta0,
        dali,
        names=names,
        labels=labels,
        n_samples=1500,
        seed=7,
    )

    assert np.allclose(np.asarray(m1.samples), np.asarray(m2.samples))
    assert np.allclose(np.asarray(m1.weights), np.asarray(m2.weights))
    assert np.allclose(np.asarray(m1.loglikes), np.asarray(m2.loglikes))


def test_dali_to_getdist_importance_raises_on_names_labels_mismatch():
    """Tests that dali_to_getdist_importance raises if names and/or labels length != p."""
    theta0, dali = _toy_dali_dict(2)

    with pytest.raises(ValueError, match=r"names must have length p=2"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=["a"],
            labels=["a", "b"],
            n_samples=10,
        )

    with pytest.raises(ValueError, match=r"labels must have length p=2"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=["a", "b"],
            labels=["a"],
            n_samples=10,
        )


def test_dali_to_getdist_importance_sampler_bounds_filters_samples():
    """Tests that sampler_bounds filters proposal samples before posterior evaluation."""
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(-0.1, 0.1), (-0.1, 0.1)]

    m = dali_to_getdist_importance(
        theta0,
        dali,
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
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(10.0, 11.0), (None, None)]  # impossible around theta0=0

    with pytest.raises(RuntimeError, match=r"All kernel samples rejected"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=names,
            labels=labels,
            n_samples=50,
            seed=0,
            sampler_bounds=sampler_bounds,
        )


def test_dali_to_getdist_importance_raises_if_all_rejected_by_logposterior():
    """Tests that dali_to_getdist_importance raises if posterior/prior makes all logpost -inf."""
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    def logprior(_th: np.ndarray) -> float:
        """Reject all samples."""
        return -np.inf

    with pytest.raises(RuntimeError, match=r"All kernel samples were rejected"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=names,
            labels=labels,
            n_samples=200,
            seed=1,
            logprior=logprior,
        )


def test_dali_to_getdist_emcee_raises_on_names_labels_mismatch():
    """Tests that dali_to_getdist_emcee raises if names/labels length != p."""
    theta0, dali = _toy_dali_dict(2)

    with pytest.raises(ValueError, match=r"names must have length p=2"):
        dali_to_getdist_emcee(
            theta0,
            dali,
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
            dali,
            names=["a", "b"],
            labels=["a"],
            n_steps=20,
            burn=0,
            thin=1,
            n_walkers=8,
        )


def test_dali_to_getdist_emcee_returns_mcsamples_and_loglikes_shapes():
    """Tests that dali_to_getdist_emcee returns MCSamples with consistent sample/loglikes shapes."""
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]

    m = dali_to_getdist_emcee(
        theta0,
        dali,
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
    theta0, dali = _toy_dali_dict(2)
    names = ["a", "b"]
    labels = ["a", "b"]
    sampler_bounds = [(-0.2, 0.2), (-0.2, 0.2)]

    m = dali_to_getdist_emcee(
        theta0,
        dali,
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
    assert np.all(s[:, 0] >= -0.2) and np.all(s[:, 0] <= 0.2)
    assert np.all(s[:, 1] >= -0.2) and np.all(s[:, 1] <= 0.2)


def test_dali_to_getdist_importance_rejects_non_dict_dali():
    """Tests that dali_to_getdist_importance requires the dict DALI form."""
    theta0 = np.zeros(2, dtype=float)
    with pytest.raises(TypeError, match=r"dali must be the dict form"):
        dali_to_getdist_importance(theta0, np.eye(2), names=["a", "b"], labels=["a", "b"], n_samples=10)


def test_dali_to_getdist_emcee_rejects_non_dict_dali():
    """Tests that dali_to_getdist_emcee requires the dict DALI form."""
    theta0 = np.zeros(2, dtype=float)
    with pytest.raises(TypeError, match=r"dali must be the dict form"):
        dali_to_getdist_emcee(theta0, np.eye(2), names=["a", "b"], labels=["a", "b"], n_steps=10, burn=0, thin=1)


def test_dali_to_getdist_importance_raises_if_missing_fisher_key():
    """Tests that dali_to_getdist_importance requires dali[1] == (F,)."""
    theta0, dali = _toy_dali_dict(2)
    dali = {k: v for k, v in dali.items() if k != 1}

    with pytest.raises(ValueError, match=r"start at key=1"):
        dali_to_getdist_importance(theta0, dali, names=["a", "b"], labels=["a", "b"], n_samples=10)


def test_dali_to_getdist_emcee_raises_if_missing_fisher_key():
    """Tests that dali_to_getdist_emcee requires dali[1] == (F,)."""
    theta0, dali = _toy_dali_dict(2)
    dali = {k: v for k, v in dali.items() if k != 1}

    with pytest.raises(ValueError, match=r"start at key=1"):
        dali_to_getdist_emcee(theta0, dali, names=["a", "b"], labels=["a", "b"], n_steps=10, burn=0, thin=1)


def test_importance_raises_on_ambiguous_prior_spec():
    """Tests that logprior cannot be combined with prior_terms/prior_bounds."""
    theta0, dali = _toy_dali_dict(2)

    def logprior(_th: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError, match=r"Ambiguous prior specification"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=["a", "b"],
            labels=["a", "b"],
            n_samples=10,
            logprior=logprior,
            prior_bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        )


def test_emcee_raises_on_ambiguous_prior_spec():
    """Tests that logprior cannot be combined with prior_terms/prior_bounds."""
    theta0, dali = _toy_dali_dict(2)

    def logprior(_th: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError, match=r"Ambiguous prior specification"):
        dali_to_getdist_emcee(
            theta0,
            dali,
            names=["a", "b"],
            labels=["a", "b"],
            n_steps=10,
            burn=0,
            thin=1,
            logprior=logprior,
            prior_bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        )


def test_importance_raises_on_empty_support_intersection():
    """Tests that incompatible prior_bounds and sampler_bounds raise ValueError."""
    theta0, dali = _toy_dali_dict(2)

    with pytest.raises(ValueError, match=r"Empty support for parameter index 0"):
        dali_to_getdist_importance(
            theta0,
            dali,
            names=["a", "b"],
            labels=["a", "b"],
            n_samples=10,
            prior_bounds=[(0.0, 0.1), (None, None)],
            sampler_bounds=[(0.2, 0.3), (None, None)],
        )


def test_importance_accepts_forecast_order_3():
    """Smoke test that forecast_order=3 runs with a triplet DALI dict."""
    theta0, dali = _toy_dali_dict(2)
    m = dali_to_getdist_importance(
        theta0,
        dali,
        forecast_order=3,
        names=["a", "b"],
        labels=["a", "b"],
        n_samples=200,
        seed=0,
    )
    s = np.asarray(m.samples)
    assert s.ndim == 2 and s.shape[1] == 2
