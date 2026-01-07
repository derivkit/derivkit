"""Tests for the LikelihoodKit high-level likelihoods interface."""

import numpy as np
import numpy.testing as npt

from derivkit.likelihood_kit import LikelihoodKit
from derivkit.likelihoods.gaussian import build_gaussian_likelihood
from derivkit.likelihoods.poisson import build_poissonian_likelihood


def test_likelihoodkit_gaussian_matches_builder():
    """Tests that Gaussian method returns the same result as the underlying helper."""
    data = np.linspace(-5.0, 5.0, 50)
    theta = np.array([0.0])
    cov = np.array([[1.5]])

    lkit = LikelihoodKit(data=data, model_parameters=theta)

    grid_direct, pdf_direct = build_gaussian_likelihood(
        data=data,
        model_parameters=theta,
        cov=cov,
        return_log=False,
    )
    grid_kit, pdf_kit = lkit.gaussian(cov=cov, return_log=False)

    assert len(grid_direct) == len(grid_kit)
    for g_direct, g_kit in zip(grid_direct, grid_kit):
        npt.assert_allclose(g_direct, g_kit)

    npt.assert_allclose(pdf_direct, pdf_kit)


def test_likelihoodkit_gaussian_return_log_true_matches_builder():
    """Tests that Gaussian method correctly forwards the return_log flag."""
    data = np.linspace(-2.0, 2.0, 20)
    theta = np.array([0.0])
    cov = np.array([[0.7]])

    lkit = LikelihoodKit(data=data, model_parameters=theta)

    _, logpdf_direct = build_gaussian_likelihood(
        data=data,
        model_parameters=theta,
        cov=cov,
        return_log=True,
    )
    _, logpdf_kit = lkit.gaussian(cov=cov, return_log=True)

    npt.assert_allclose(logpdf_direct, logpdf_kit)


def test_likelihoodkit_poissonian_matches_builder():
    """Tests that Poissonian method returns the same result as the underlying helper."""
    counts = np.array([1, 2, 3, 4])
    mu = np.array([0.5, 1.0, 1.5, 2.0])

    lkit = LikelihoodKit(data=counts, model_parameters=mu)

    counts_direct, pmf_direct = build_poissonian_likelihood(
        data=counts,
        model_parameters=mu,
        return_log=False,
    )
    counts_kit, pmf_kit = lkit.poissonian(return_log=False)

    npt.assert_allclose(counts_direct, counts_kit)
    npt.assert_allclose(pmf_direct, pmf_kit)


def test_likelihoodkit_poissonian_return_log_true_matches_builder():
    """Tests that Poissonian method correctly forwards the return_log flag."""
    counts = np.array([0, 1, 2, 3])
    mu = np.array([0.1, 0.5, 1.0, 2.0])

    lkit = LikelihoodKit(data=counts, model_parameters=mu)

    counts_direct, logpmf_direct = build_poissonian_likelihood(
        data=counts,
        model_parameters=mu,
        return_log=True,
    )
    counts_kit, logpmf_kit = lkit.poissonian(return_log=True)

    npt.assert_allclose(counts_direct, counts_kit)
    npt.assert_allclose(logpmf_direct, logpmf_kit)


def test_likelihoodkit_stores_data_and_parameters_as_arrays():
    """Tests that LikelihoodKit stores data and model_parameters as numpy arrays."""
    data = [1.0, 2.0, 3.0]
    theta = [0.1, 0.2, 0.3]

    lkit = LikelihoodKit(data=data, model_parameters=theta)

    assert isinstance(lkit.data, np.ndarray)
    assert isinstance(lkit.model_parameters, np.ndarray)
    npt.assert_allclose(lkit.data, np.asarray(data))
    npt.assert_allclose(lkit.model_parameters, np.asarray(theta))
