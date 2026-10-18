"""Integration tests for nonlinear DALI systematic-bias shifts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from derivkit.forecasting.dali_bias import (
    build_dali_bias_tensor,
    build_dali_map_shift,
    build_dali_mean_shift,
)
from derivkit.forecasting.expansions import build_logposterior_dali
from derivkit.forecasting.forecast_core import get_forecast_tensors
from derivkit.forecasting.getdist_dali_samples import (
    dali_to_getdist_importance,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]


THETA0 = np.array([0.1, -0.1])
COV = np.array([[0.04, 0.006], [0.006, 0.025]])
DELTA_NU = np.array([0.015, -0.008])

BOUNDS = [
    (THETA0[0] - 0.8, THETA0[0] + 0.8),
    (THETA0[1] - 0.8, THETA0[1] + 0.8),
]


def nonlinear_model(theta):
    """Smooth nonlinear two-parameter forward model."""
    u, v = theta
    return np.array([
        u + 0.3 * u**2 + 0.2 * v**2,
        v + 0.25 * u * v + 0.15 * u**2,
    ])


def nonlinear_case():
    """Construct DALI and bias tensors using shared derivatives."""
    derivatives = {}

    dali = get_forecast_tensors(
        nonlinear_model,
        THETA0,
        COV,
        forecast_order=3,
        derivatives=derivatives,
    )

    bias = build_dali_bias_tensor(
        nonlinear_model,
        THETA0,
        COV,
        DELTA_NU,
        bias_order=3,
        derivatives=derivatives,
    )

    return dali, bias


def scaled_bias(bias, amplitude):
    """Scale precomputed bias tensors for an amplitude-scaled mismatch."""
    return {
        order: amplitude * tensor
        for order, tensor in bias.items()
    }


def dali_logpost(theta, dali, bias=None):
    """Evaluate the third-order DALI log posterior."""
    return build_logposterior_dali(
        theta,
        THETA0,
        dali,
        bias=bias,
        forecast_order=3,
    )


def numerical_map(dali, bias=None):
    """Optimize the local DALI posterior."""
    result = minimize(
        lambda theta: -dali_logpost(theta, dali, bias),
        THETA0,
        method="BFGS",
        options={"gtol": 1e-9},
    )

    assert result.success or np.linalg.norm(result.jac) < 1e-6
    return result.x


def simpson_weights(x):
    """Return Simpson quadrature weights."""
    n = len(x)
    weights = np.ones(n)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * (x[1] - x[0]) / 3.0


def posterior_grid(dali, bias=None, n_grid=301):
    """Return grid points and normalized posterior quadrature weights."""
    assert n_grid % 2 == 1

    u = np.linspace(*BOUNDS[0], n_grid)
    v = np.linspace(*BOUNDS[1], n_grid)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    points = np.column_stack([uu.ravel(), vv.ravel()])

    logp = np.array([
        dali_logpost(theta, dali, bias)
        for theta in points
    ]).reshape(uu.shape)

    density = np.exp(logp - np.max(logp))

    wu = simpson_weights(u)
    wv = simpson_weights(v)

    weights = (density * wu[:, None] * wv[None, :]).ravel()
    weights /= weights.sum()

    return points, weights


def numerical_mean(dali, bias=None, n_grid=301):
    """Compute the bounded DALI posterior mean by quadrature."""
    points, weights = posterior_grid(dali, bias, n_grid=n_grid)
    return np.sum(points * weights[:, None], axis=0)


def test_nonlinear_dali_map_shift():
    """Validate perturbative MAP shifts and convergence with amplitude."""
    dali, bias = nonlinear_case()
    derivatives = {}

    baseline = numerical_map(dali)
    errors = []

    for amplitude in (1.0, 0.5, 0.25):
        current_bias = scaled_bias(bias, amplitude)

        analytic = build_dali_map_shift(
            nonlinear_model,
            THETA0,
            COV,
            amplitude * DELTA_NU,
            expansion_order=3,
            derivatives=derivatives,
        )

        numerical = numerical_map(dali, current_bias) - baseline

        errors.append(np.linalg.norm(analytic - numerical))

        assert_allclose(analytic, numerical, atol=1e-3, rtol=0)

    assert errors[2] < errors[0]


def test_nonlinear_dali_mean_shift():
    """Validate the local analytic posterior mean shift."""
    dali, bias = nonlinear_case()
    derivatives = {}

    map_shift = build_dali_map_shift(
        nonlinear_model,
        THETA0,
        COV,
        DELTA_NU,
        expansion_order=3,
        derivatives=derivatives,
    )

    analytic = build_dali_mean_shift(
        nonlinear_model,
        THETA0,
        COV,
        DELTA_NU,
        mean_method="analytic",
        derivatives=derivatives,
    )

    numerical = numerical_mean(dali, bias) - numerical_mean(dali)

    # This is a local analytic approximation, not exact quadrature.
    assert_allclose(analytic, numerical, atol=2e-2, rtol=0)

    # The nonlinear posterior allows different MAP and mean shifts.
    assert np.linalg.norm(analytic - map_shift) > 1e-6


def test_nonlinear_dali_end_to_end():
    """Validate tensor construction and GetDist against the exact model."""
    dali, bias = nonlinear_case()

    inv_cov = np.linalg.inv(COV)
    reference = nonlinear_model(THETA0) + DELTA_NU

    def exact_logpost(theta):
        residual = nonlinear_model(theta) - reference
        return -0.5 * residual @ inv_cov @ residual

    # The quadratic forward model has no third or higher derivatives.
    # Third-order DALI includes all terms in the quartic log likelihood.
    for displacement in (
        np.array([0.02, 0.01]),
        np.array([-0.03, 0.02]),
        np.array([0.01, -0.025]),
        np.array([0.3, -0.2]),
    ):
        theta = THETA0 + displacement

        dali_difference = (
            dali_logpost(theta, dali, bias)
            - dali_logpost(THETA0, dali, bias)
        )

        exact_difference = (
            exact_logpost(theta)
            - exact_logpost(THETA0)
        )

        assert_allclose(
            dali_difference,
            exact_difference,
            atol=1e-5,
            rtol=0,
        )

    samples = dali_to_getdist_importance(
        THETA0,
        dali,
        bias=bias,
        forecast_order=3,
        names=["u", "v"],
        labels=["u", "v"],
        n_samples=30_000,
        kernel_scale=1.5,
        sampler_bounds=BOUNDS,
        seed=42,
    )

    sampled_mean = np.average(
        samples.samples,
        axis=0,
        weights=samples.weights,
    )

    reference_mean = numerical_mean(dali, bias)

    assert_allclose(
        sampled_mean,
        reference_mean,
        atol=0.02,
        rtol=0,
    )


def test_nonlinear_dali_mean_response():
    """Validate the full non-Gaussian posterior mean response."""
    dali, bias = nonlinear_case()

    points, weights = posterior_grid(dali)
    derivatives = {}

    baseline_mean = np.sum(points * weights[:, None], axis=0)

    for amplitude in (0.25, 0.5, 1.0):
        predicted = build_dali_mean_shift(
            nonlinear_model,
            THETA0,
            COV,
            amplitude * DELTA_NU,
            mean_method="semi-analytic",
            forecast_order=3,
            theta=points,
            weights=weights,
            response_order=4,
            derivatives=derivatives,
        )

        numerical = (
            numerical_mean(dali, scaled_bias(bias, amplitude))
            - baseline_mean
        )

        assert_allclose(predicted, numerical, atol=1e-4, rtol=0)


def test_nonlinear_dali_grid_convergence():
    """Verify that the numerical reference mean is grid converged."""
    dali, bias = nonlinear_case()

    coarse = numerical_mean(dali, bias, n_grid=151)
    fine = numerical_mean(dali, bias, n_grid=301)

    assert_allclose(coarse, fine, atol=1e-5, rtol=0)
