"""Unit tests for ``derivkit.forecasting.dali_bias``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from derivkit.forecasting.dali_bias import (
    _mean_map_offset,
    build_dali_bias_tensor,
    build_dali_map_shift,
    build_dali_mean_shift,
)


def linear_model(theta):
    """Returns a linear two-parameter, two-observable model."""
    x, y = theta
    return np.array(
        [
            2.0 * x + 3.0 * y,
            -x + 4.0 * y,
        ]
    )


def quadratic_model(theta):
    """Returns a quadratic two-parameter, two-observable model."""
    x, y = theta
    return np.array(
        [
            x + 2.0 * y + x**2 + 3.0 * x * y + 2.0 * y**2,
            2.0 * x - y + 2.0 * x**2 - x * y + y**2,
        ]
    )


def cubic_model(theta):
    """Returns a cubic two-parameter, two-observable model."""
    x, y = theta
    return np.array(
        [
            x + x**2 + x * y + x**3 + 2.0 * x**2 * y + 3.0 * x * y**2 + y**3,
            y + x**2 - y**2 - x**3 + x**2 * y - 2.0 * x * y**2 + 2.0 * y**3,
        ]
    )


def smooth_model(theta):
    """Returns a smooth nonlinear two-parameter, two-observable model."""
    x, y = theta
    return np.array([np.sin(x) + np.cos(y), np.tanh(x + y)])


def test_build_dali_bias_tensor_linear_model():
    """Tests first-order bias tensor for a linear model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=1,
    )

    expected = np.array([5.0, 2.0])

    assert set(bias) == {1}
    assert bias[1].shape == (2,)
    assert_allclose(bias[1], expected)


def test_build_dali_bias_tensor_linear_model_higher_orders_zero():
    """Tests that higher-order bias tensors vanish for a linear model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )

    assert set(bias) == {1, 2, 3}
    assert_allclose(bias[1], np.array([5.0, 2.0]))
    assert_allclose(bias[2], np.zeros((2, 2)), atol=1e-10)
    assert_allclose(bias[3], np.zeros((2, 2, 2)), atol=1e-10)


def test_build_dali_bias_tensor_quadratic_model():
    """Tests first- and second-order bias tensors for a quadratic model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        quadratic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=2,
    )

    expected_first = np.array([0.0, 5.0])
    expected_second = np.array(
        [
            [0.0, 7.0],
            [7.0, 6.0],
        ]
    )

    assert bias[1].shape == (2,)
    assert bias[2].shape == (2, 2)
    assert_allclose(bias[1], expected_first, atol=1e-10)
    assert_allclose(bias[2], expected_second, atol=1e-10)


def test_build_dali_bias_tensor_cubic_model():
    """Tests bias tensors through third order for a cubic model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )

    expected_first = np.array([2.0, -1.0])
    expected_second = np.array(
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    )
    expected_third = np.array(
        [
            [[18.0, 6.0], [6.0, 16.0]],
            [[6.0, 16.0], [16.0, 0.0]],
        ]
    )

    assert bias[1].shape == (2,)
    assert bias[2].shape == (2, 2)
    assert bias[3].shape == (2, 2, 2)
    assert_allclose(bias[1], expected_first, atol=1e-10)
    assert_allclose(bias[2], expected_second, atol=1e-10)
    assert_allclose(bias[3], expected_third, atol=1e-10)


def test_build_dali_bias_tensor_non_diagonal_covariance():
    """Tests inverse-covariance weighting for correlated observables."""
    theta0 = np.array([0.0, 0.0])
    cov = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )
    delta_nu = np.array([1.0, 2.0])

    bias = build_dali_bias_tensor(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=1,
    )

    expected = np.array([-0.2, 3.0])

    assert_allclose(bias[1], expected)


def test_build_dali_bias_tensor_zero_mismatch():
    """Tests that a zero data mismatch produces zero bias tensors."""
    theta0 = np.array([0.0, 0.0])
    cov = np.array(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )
    delta_nu = np.zeros(2)

    bias = build_dali_bias_tensor(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )

    assert_allclose(bias[1], np.zeros(2))
    assert_allclose(bias[2], np.zeros((2, 2)))
    assert_allclose(bias[3], np.zeros((2, 2, 2)))


def test_build_dali_bias_tensor_scales_linearly_with_mismatch():
    """Tests linear scaling of bias tensors with the data mismatch."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([0.5, -1.5])

    bias = build_dali_bias_tensor(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )
    scaled_bias = build_dali_bias_tensor(
        cubic_model,
        theta0,
        cov,
        4.0 * delta_nu,
        bias_order=3,
    )

    for order in range(1, 4):
        assert_allclose(scaled_bias[order], 4.0 * bias[order], atol=1e-10)


def test_build_dali_bias_tensor_tensor_symmetry():
    """Tests symmetry of higher-order bias tensors over parameter axes."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )

    assert_allclose(bias[2], bias[2].T, atol=1e-10)
    assert_allclose(bias[3], bias[3].transpose(1, 0, 2), atol=1e-10)
    assert_allclose(bias[3], bias[3].transpose(0, 2, 1), atol=1e-10)
    assert_allclose(bias[3], bias[3].transpose(2, 1, 0), atol=1e-10)


def test_build_dali_bias_tensor_nonzero_expansion_point():
    """Tests bias tensors when derivatives are evaluated away from the origin."""
    theta0 = np.array([1.0, -1.0])
    cov = np.eye(2)
    delta_nu = np.array([1.0, 0.0])

    bias = build_dali_bias_tensor(
        quadratic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=2,
    )

    expected_first = np.array([0.0, 1.0])
    expected_second = np.array(
        [
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    assert_allclose(bias[1], expected_first, atol=1e-10)
    assert_allclose(bias[2], expected_second, atol=1e-10)


def test_build_dali_bias_tensor_accepts_list_inputs():
    """Tests that array-like inputs are accepted."""
    bias = build_dali_bias_tensor(
        linear_model,
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [2.0, -1.0],
        bias_order=1,
    )

    assert_allclose(bias[1], np.array([5.0, 2.0]))


def test_build_dali_bias_tensor_delta_nu_column_vector():
    """Tests that the data mismatch is flattened to one dimension."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([[2.0], [-1.0]])

    bias = build_dali_bias_tensor(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=1,
    )

    assert_allclose(bias[1], np.array([5.0, 2.0]))


@pytest.mark.parametrize("bias_order", [1, 2, 3, 4])
def test_build_dali_bias_tensor_requested_order(bias_order):
    """Tests that only orders 1 through bias_order are returned."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([1.0, 1.0])

    bias = build_dali_bias_tensor(
        cubic_model, theta0, cov, delta_nu, bias_order=bias_order
    )

    assert set(bias) == set(range(1, bias_order + 1))
    assert 0 not in bias


@pytest.mark.parametrize("bias_order", [0, -1, 5])
def test_build_dali_bias_tensor_unsupported_order_raises(bias_order):
    """Tests that unsupported bias orders raise ValueError."""
    with pytest.raises(ValueError, match="bias_order=.* is not supported"):
        build_dali_bias_tensor(
            linear_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 1.0],
            bias_order=bias_order,
        )


@pytest.mark.parametrize("bias_order", [
    "invalid",
    {},
    1.5,
    np.array([1, 2]),
])
def test_build_dali_bias_tensor_invalid_order_type_raises(bias_order):
    """Tests that invalid bias-order inputs are rejected."""
    with pytest.raises((TypeError, ValueError)):
        build_dali_bias_tensor(
            linear_model, [0.0, 0.0], np.eye(2),
            [1.0, 1.0], bias_order=bias_order
        )


def test_build_dali_bias_tensor_empty_theta0_raises():
    """Tests that an empty expansion point raises ValueError."""
    with pytest.raises(ValueError, match="theta0 must be non-empty 1D"):
        build_dali_bias_tensor(
            linear_model,
            [],
            np.eye(2),
            [1.0, 1.0],
            bias_order=1,
        )


def test_build_dali_bias_tensor_mismatch_size_raises():
    """Tests that the mismatch size must match the number of observables."""
    with pytest.raises(ValueError, match="Expected 2 elements in delta_nu"):
        build_dali_bias_tensor(
            linear_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 2.0, 3.0],
            bias_order=1,
        )


def test_build_dali_bias_tensor_model_output_size_raises():
    """Tests that model output size must match the covariance dimension."""

    def three_observable_model(theta):
        x, y = theta
        return np.array([x, y, x + y])

    with pytest.raises(ValueError, match="Expected 2 observables from model"):
        build_dali_bias_tensor(
            three_observable_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 1.0],
            bias_order=1,
        )


def test_build_dali_bias_tensor_invalid_covariance_shape_raises():
    """Tests that a non-square covariance matrix raises ValueError."""
    with pytest.raises(ValueError):
        build_dali_bias_tensor(
            linear_model,
            [0.0, 0.0],
            np.ones((2, 3)),
            [1.0, 1.0],
            bias_order=1,
        )


def test_build_dali_bias_tensor_smooth_model():
    """Tests bias tensors for a smooth non-polynomial model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        smooth_model, theta0, cov, delta_nu, bias_order=3
    )

    expected_first = np.array([1.0, -1.0])
    expected_second = np.array([[0.0, 0.0], [0.0, -2.0]])
    expected_third = np.array([
        [[0.0, 2.0], [2.0, 2.0]],
        [[2.0, 2.0], [2.0, 2.0]],
    ])

    assert_allclose(bias[1], expected_first, atol=1e-5)
    assert_allclose(bias[2], expected_second, atol=1e-5)
    assert_allclose(bias[3], expected_third, atol=1e-4)


# ---------------------------------------------------------------------
# MAP and posterior-mean systematic shifts
# ---------------------------------------------------------------------


def scalar_linear_model(theta):
    """Returns a one-parameter linear model."""
    return np.array([theta[0]])


def scalar_quadratic_model(theta):
    """Returns a one-parameter quadratic model."""
    x = theta[0]
    return np.array([x + 0.5 * x**2])


def nonlinear_scalar_model(theta):
    """Returns a quadratic model with an asymmetric posterior."""
    x = theta[0]
    return np.array([x + 0.3 * x**2, 0.7 * x + 0.2 * x**2])


def nonlinear_scalar_reference():
    """Returns a numerical reference for the unbiased nonlinear posterior."""
    x = np.linspace(-2.0, 2.0, 4001)
    theta = x[:, None]
    predictions = np.array([nonlinear_scalar_model(t) for t in theta])

    precision = np.array([[1.0, 0.25], [0.25, 1.5]])
    cov = np.linalg.inv(precision)

    log_weights = -0.5 * np.einsum(
        "ni,ij,nj->n", predictions, precision, predictions
    )
    weights = np.exp(log_weights - logsumexp(log_weights))

    return theta, predictions, cov, precision, weights


def coupled_quadratic_model(theta):
    """Returns a two-parameter model with nonlinear cross-coupling."""
    x, y = theta
    return np.array([
        x + 0.35 * y + 0.15 * x**2 + 0.2 * x * y,
        y - 0.25 * x + 0.1 * y**2 + 0.15 * x * y,
    ])


def coupled_quadratic_reference(n=301):
    """Returns a two-dimensional numerical posterior reference."""
    axis = np.linspace(-4.0, 4.0, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    theta = np.column_stack([x.ravel(), y.ravel()])
    predictions = np.array([coupled_quadratic_model(t) for t in theta])

    cov = np.array([[0.16, 0.045], [0.045, 0.25]])
    precision = np.linalg.inv(cov)

    log_weights = -0.5 * np.einsum(
        "ni,ij,nj->n", predictions, precision, predictions
    )
    weights = np.exp(log_weights - logsumexp(log_weights))

    return theta, predictions, cov, precision, weights


def test_build_dali_map_shift_linear_gaussian():
    """Tests the MAP shift against the exact linear-Gaussian solution."""
    theta0 = np.zeros(2)
    delta_nu = np.array([0.2, -0.1])

    shift = build_dali_map_shift(
        linear_model, theta0, np.eye(2), delta_nu
    )

    inverse_jacobian = 1/11 * np.array([[4, -3], [1, 2]])
    expected = inverse_jacobian @ delta_nu

    assert_allclose(shift, expected, atol=1e-10)


def test_build_dali_mean_shift_linear_gaussian():
    """Tests that the analytic mean shift equals the Fisher bias for a linear model."""
    theta0 = np.zeros(2)
    delta_nu = np.array([0.2, -0.1])

    expected = 1/10 * np.array([1, 0])
    shift = build_dali_mean_shift(
        linear_model, theta0, np.eye(2), delta_nu,
        mean_method="analytic",
    )

    assert_allclose(shift, expected, atol=1e-10)


def test_build_dali_mean_shift_semi_analytic_linear_gaussian():
    """Tests the semi-analytic mean against an exact Gaussian response."""
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    theta = (np.sqrt(2.0) * nodes)[:, None]

    shift = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.15],
        mean_method="semi-analytic",
        theta=theta,
        weights=weights,
        response_order=4,
    )

    assert_allclose(shift, [0.15], atol=2e-6)


def test_build_dali_mean_shift_semi_analytic_two_parameter_gaussian():
    """Tests the semi-analytic response for two correlated model parameters."""
    inverse_jacobian = 1/11 * np.array([[4, -3], [1, 2]])
    delta_nu = np.array([0.1, -0.05])
    expected = inverse_jacobian @ delta_nu

    nodes, weights = np.polynomial.hermite.hermgauss(12)
    x, y = np.meshgrid(
        np.sqrt(2.0) * nodes,
        np.sqrt(2.0) * nodes,
        indexing="ij",
    )

    samples = (
        np.column_stack([x.ravel(), y.ravel()])
        @ inverse_jacobian.T
    )
    quadrature_weights = np.outer(weights, weights).ravel()

    shift = build_dali_mean_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        delta_nu,
        mean_method="semi-analytic",
        theta=samples,
        weights=quadrature_weights,
        response_order=4,
    )

    assert_allclose(shift, expected, atol=2e-5)


@pytest.mark.parametrize(
    "model",
    [linear_model, quadratic_model, cubic_model],
)
def test_dali_shifts_zero_mismatch(model):
    """Tests that all systematic shifts vanish for zero mismatch."""
    theta0 = np.zeros(2)
    cov = np.eye(2)
    delta_nu = np.zeros(2)

    samples = np.array([
        [-0.5, -0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.5, 0.5],
    ])

    map_shift = build_dali_map_shift(
        model, theta0, cov, delta_nu
    )

    analytic_mean = build_dali_mean_shift(
        model, theta0, cov, delta_nu,
        mean_method="analytic",
    )

    semi_analytic_mean = build_dali_mean_shift(
        model, theta0, cov, delta_nu,
        mean_method="semi-analytic",
        theta=samples,
    )

    assert_allclose(map_shift, np.zeros(2), atol=1e-12)
    assert_allclose(analytic_mean, np.zeros(2), atol=1e-12)
    assert_allclose(semi_analytic_mean, np.zeros(2), atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_dali_map_shift_first_order_fisher_limit(order):
    """Tests that linear-Gaussian MAP shifts equal Fisher bias at every order."""
    theta0 = np.zeros(2)
    delta_nu = np.array([0.02, -0.01])

    # expected = inverse_fisher @ jacobian.T @ delta_nu
    expected = 1e-2 * np.array([1, 0])

    shift = build_dali_map_shift(
        linear_model,
        theta0,
        np.eye(2),
        delta_nu,
        expansion_order=order,
    )

    assert_allclose(shift, expected, atol=1e-10)


@pytest.mark.parametrize("amplitude", [0.01, 0.02, 0.04])
def test_dali_map_shift_against_nonlinear_optimization(amplitude):
    """Tests the MAP expansion against direct nonlinear likelihood optimization."""
    theta0 = np.zeros(1)
    _, _, cov, precision, _ = nonlinear_scalar_reference()
    delta_nu = np.array([0.4, -0.2])

    def negative_log_likelihood(x):
        residual = nonlinear_scalar_model([x]) - amplitude * delta_nu
        return 0.5 * residual @ precision @ residual

    optimum = minimize_scalar(
        negative_log_likelihood,
        bounds=(-0.5, 0.5),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert optimum.success

    predicted = build_dali_map_shift(
        nonlinear_scalar_model,
        theta0,
        cov,
        amplitude * delta_nu,
        expansion_order=3,
    )

    assert_allclose(predicted, [optimum.x], atol=2e-5, rtol=0)


def test_dali_map_expansion_convergence():
    """Tests that higher MAP orders improve the small-systematic expansion."""
    cov = np.eye(2)
    theta0 = np.zeros(1)
    delta_nu = np.array([0.5, 0.2])
    amplitudes = [0.04, 0.02]

    errors = {1: [], 2: [], 3: []}

    for amplitude in amplitudes:

        def negative_log_likelihood(x):
            residual = nonlinear_scalar_model([x]) - amplitude * delta_nu
            return 0.5 * residual @ residual

        optimum = minimize_scalar(
            negative_log_likelihood,
            bounds=(-0.5, 0.5),
            method="bounded",
            options={"xatol": 1e-14},
        )
        assert optimum.success

        for order in errors:
            predicted = build_dali_map_shift(
                nonlinear_scalar_model,
                theta0,
                cov,
                amplitude * delta_nu,
                expansion_order=order,
            )
            errors[order].append(abs(predicted[0] - optimum.x))

    for order in errors:
        assert errors[order][1] < errors[order][0]

    assert errors[2][0] < errors[1][0]
    assert errors[3][0] < errors[2][0]


@pytest.mark.parametrize("amplitude", [0.01, 0.02, 0.04])
def test_dali_map_shift_coupled_nonlinear(amplitude):
    """Tests both nonlinear MAP components against direct optimization."""
    cov = np.array([[0.16, 0.045], [0.045, 0.25]])
    inv_cov = 1e3/(16*25*100 - 45**2) * np.array([[250, -45], [-45, 160]])
    delta_nu = np.array([0.3, -0.2])

    def negative_log_likelihood(theta):
        residual = coupled_quadratic_model(theta) - amplitude * delta_nu
        return 0.5 * residual @ inv_cov @ residual

    optimum = minimize(
        negative_log_likelihood,
        np.zeros(2),
        method="BFGS",
        options={"gtol": 1e-12},
    )

    assert np.linalg.norm(optimum.jac) < 1e-6

    predicted = build_dali_map_shift(
        coupled_quadratic_model,
        np.zeros(2),
        cov,
        amplitude * delta_nu,
        expansion_order=3,
    )

    assert_allclose(predicted, optimum.x, atol=2e-5, rtol=0)


def test_dali_map_shift_metadata():
    """Tests MAP metadata and its Gaussian/non-Gaussian decomposition."""
    theta0 = np.array([0.1, -0.1])

    shift, metadata = build_dali_map_shift(
        quadratic_model,
        theta0,
        np.eye(2),
        [0.01, -0.02],
        return_metadata=True,
    )

    assert set(metadata) == {
        "estimator",
        "baseline",
        "displacement",
        "terms",
        "gaussian_terms",
        "non_gaussian_terms",
        "gaussian_contribution",
        "non_gaussian_contribution",
    }

    assert set(metadata["terms"]) == {1, 2, 3}
    assert set(metadata["gaussian_terms"]) == {1, 2, 3}
    assert set(metadata["non_gaussian_terms"]) == {1, 2, 3}

    assert_allclose(metadata["baseline"], theta0)
    assert_allclose(metadata["estimator"], theta0 + shift)
    assert_allclose(metadata["displacement"], shift)
    assert_allclose(sum(metadata["terms"].values()), shift)

    assert_allclose(
        metadata["gaussian_contribution"]
        + metadata["non_gaussian_contribution"],
        shift,
    )

    for order in (1, 2, 3):
        assert_allclose(
            metadata["gaussian_terms"][order]
            + metadata["non_gaussian_terms"][order],
            metadata["terms"][order],
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_dali_map_shift_metadata_orders(order):
    """Tests that MAP metadata contains only requested expansion orders."""
    _, metadata = build_dali_map_shift(
        quadratic_model,
        np.zeros(2),
        np.eye(2),
        [0.01, 0.02],
        expansion_order=order,
        return_metadata=True,
    )

    expected = set(range(1, order + 1))

    assert set(metadata["terms"]) == expected
    assert set(metadata["gaussian_terms"]) == expected
    assert set(metadata["non_gaussian_terms"]) == expected


def test_dali_map_shift_gaussian_terms_linear_model():
    """Tests that non-Gaussian MAP corrections vanish for a linear model."""
    _, metadata = build_dali_map_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        [0.02, -0.01],
        return_metadata=True,
    )

    assert_allclose(
        metadata["non_gaussian_contribution"],
        np.zeros(2),
        atol=1e-12,
    )

    for term in metadata["non_gaussian_terms"].values():
        assert_allclose(term, np.zeros(2), atol=1e-12)


def test_dali_analytic_mean_metadata():
    """Tests analytic mean metadata and the fiducial displacement."""
    theta0 = np.zeros(1)

    shift, metadata = build_dali_mean_shift(
        nonlinear_scalar_model,
        theta0,
        np.eye(2),
        [0.01, -0.005],
        mean_method="analytic",
        return_metadata=True,
    )

    assert set(metadata) == {
        "method",
        "estimator",
        "baseline",
        "displacement",
        "map_shift",
    }

    assert metadata["method"] == "analytic"
    assert_allclose(metadata["estimator"], metadata["baseline"] + shift)
    assert_allclose(metadata["displacement"], metadata["estimator"] - theta0)

    assert np.all(np.isfinite(shift))
    assert np.all(np.isfinite(metadata["map_shift"]))


def test_dali_analytic_mean_zero_mismatch_baseline():
    """Tests that an unbiased mean offset is not a systematic shift."""
    theta0 = np.zeros(1)

    shift, metadata = build_dali_mean_shift(
        nonlinear_scalar_model,
        theta0,
        np.eye(2),
        np.zeros(2),
        mean_method="analytic",
        return_metadata=True,
    )

    assert_allclose(shift, [0.0], atol=1e-12)
    assert_allclose(metadata["estimator"], metadata["baseline"])
    assert_allclose(
        metadata["displacement"],
        metadata["baseline"] - theta0,
    )


@pytest.mark.parametrize("nonlinearity", [0.1, 0.05, 0.025])
def test_dali_analytic_mean_shift_weak_nonlinearity(nonlinearity):
    """Tests analytic mean shifts against integration for weak nonlinearity."""

    def model(theta):
        x = theta[0]
        return np.array([x + nonlinearity * x**2, 0.8 * x])

    x = np.linspace(-5.0, 5.0, 20001)
    predictions = np.column_stack([
        x + nonlinearity * x**2,
        0.8 * x,
    ])

    cov = np.eye(2)
    delta_nu = np.array([0.1, -0.05])

    def posterior_mean(mismatch):
        residual = predictions - mismatch
        log_weights = -0.5 * np.sum(residual**2, axis=1)
        weights = np.exp(log_weights - logsumexp(log_weights))
        return np.average(x, weights=weights)

    exact_shift = (
        posterior_mean(delta_nu)
        - posterior_mean(np.zeros(2))
    )

    predicted = build_dali_mean_shift(
        model,
        np.zeros(1),
        cov,
        delta_nu,
        mean_method="analytic",
    )

    assert np.isfinite(predicted[0])
    assert abs(predicted[0] - exact_shift) < 0.02


@pytest.mark.parametrize("amplitude", [0.02, 0.05, 0.1])
def test_dali_semi_analytic_mean_against_direct_integration(amplitude):
    """Tests fourth-order mean response against direct posterior integration."""
    theta, predictions, cov, precision, weights = nonlinear_scalar_reference()
    delta_nu = np.array([0.3, -0.2])

    unbiased_mean = np.average(theta[:, 0], weights=weights)

    residual = predictions - amplitude * delta_nu
    log_biased = -0.5 * np.einsum(
        "ni,ij,nj->n", residual, precision, residual
    )
    biased_weights = np.exp(log_biased - logsumexp(log_biased))

    exact_shift = (
        np.average(theta[:, 0], weights=biased_weights)
        - unbiased_mean
    )

    predicted = build_dali_mean_shift(
        nonlinear_scalar_model,
        [0.0],
        cov,
        amplitude * delta_nu,
        mean_method="semi-analytic",
        forecast_order=3,
        theta=theta,
        weights=weights,
        response_order=4,
    )

    assert_allclose(predicted, [exact_shift], atol=2e-7, rtol=0)


def test_dali_semi_analytic_mean_nongaussian_convergence():
    """Tests fourth-order response convergence for a non-Gaussian posterior."""
    theta, predictions, cov, precision, weights = nonlinear_scalar_reference()
    delta_nu = np.array([0.3, -0.2])

    mean0 = np.average(theta[:, 0], weights=weights)
    errors = {}

    for amplitude in (0.2, 0.1):
        residual = predictions - amplitude * delta_nu
        log_biased = -0.5 * np.einsum(
            "ni,ij,nj->n", residual, precision, residual
        )
        biased_weights = np.exp(log_biased - logsumexp(log_biased))
        exact = np.average(theta[:, 0], weights=biased_weights) - mean0

        errors[amplitude] = {}

        for order in (1, 2, 3, 4):
            predicted = build_dali_mean_shift(
                nonlinear_scalar_model,
                [0.0],
                cov,
                amplitude * delta_nu,
                mean_method="semi-analytic",
                forecast_order=3,
                theta=theta,
                weights=weights,
                response_order=order,
            )
            errors[amplitude][order] = abs(predicted[0] - exact)

    for order in (1, 2, 3, 4):
        assert errors[0.1][order] < errors[0.2][order]

    assert errors[0.2][4] < errors[0.2][3]
    assert errors[0.2][3] < errors[0.2][2]


@pytest.mark.parametrize("amplitude", [0.02, 0.05, 0.1])
def test_dali_semi_analytic_mean_coupled_nonlinear(amplitude):
    """Tests both mean components against direct two-dimensional integration."""
    theta, predictions, cov, precision, weights = coupled_quadratic_reference()
    delta_nu = np.array([0.3, -0.2])

    mean0 = np.average(theta, axis=0, weights=weights)

    residual = predictions - amplitude * delta_nu
    log_biased = -0.5 * np.einsum(
        "ni,ij,nj->n", residual, precision, residual
    )
    biased_weights = np.exp(log_biased - logsumexp(log_biased))

    exact_shift = (
        np.average(theta, axis=0, weights=biased_weights)
        - mean0
    )

    predicted = build_dali_mean_shift(
        coupled_quadratic_model,
        np.zeros(2),
        cov,
        amplitude * delta_nu,
        mean_method="semi-analytic",
        forecast_order=3,
        theta=theta,
        weights=weights,
        response_order=4,
    )

    assert_allclose(predicted, exact_shift, atol=2e-6, rtol=0)


def test_dali_semi_analytic_mean_asymmetric_discrete_reference():
    """Tests the mean response against independently reweighted discrete nodes."""
    theta = np.array([[-1.0], [0.0], [2.0]])
    weights = np.array([2.0, 3.0, 5.0])
    delta_nu = np.array([0.05])

    biased_weights = weights * np.exp(delta_nu[0] * theta[:, 0])

    exact_shift = (
        np.average(theta[:, 0], weights=biased_weights)
        - np.average(theta[:, 0], weights=weights)
    )

    predicted = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        delta_nu,
        mean_method="semi-analytic",
        forecast_order=1,
        theta=theta,
        weights=weights,
        response_order=4,
    )

    assert_allclose(predicted, [exact_shift], atol=2e-6)


def test_dali_semi_analytic_mean_metadata():
    """Tests response coefficients, baseline, estimator, and displacement."""
    theta, _, cov, _, weights = nonlinear_scalar_reference()
    theta0 = np.zeros(1)

    shift, metadata = build_dali_mean_shift(
        nonlinear_scalar_model,
        theta0,
        cov,
        [0.01, -0.005],
        mean_method="semi-analytic",
        theta=theta,
        weights=weights,
        response_order=4,
        return_metadata=True,
    )

    assert set(metadata) == {
        "method",
        "estimator",
        "baseline",
        "displacement",
        "coefficients",
        "terms",
    }

    assert metadata["method"] == "semi-analytic"
    assert set(metadata["coefficients"]) == {1, 2, 3, 4}
    assert set(metadata["terms"]) == {1, 2, 3, 4}

    expected_baseline = np.average(theta, axis=0, weights=weights)

    assert_allclose(metadata["baseline"], expected_baseline)
    assert_allclose(metadata["estimator"], expected_baseline + shift)
    assert_allclose(
        metadata["displacement"],
        expected_baseline + shift - theta0,
    )
    assert_allclose(sum(metadata["terms"].values()), shift)

    for order in range(1, 5):
        assert_allclose(
            metadata["coefficients"][order],
            metadata["terms"][order],
        )


def test_dali_semi_analytic_mean_baseline_offset():
    """Tests that a non-Gaussian baseline mean offset is not a systematic shift."""
    theta, _, cov, _, weights = nonlinear_scalar_reference()

    shift, metadata = build_dali_mean_shift(
        nonlinear_scalar_model,
        [0.0],
        cov,
        np.zeros(2),
        mean_method="semi-analytic",
        theta=theta,
        weights=weights,
        return_metadata=True,
    )

    expected_baseline = np.average(theta, axis=0, weights=weights)

    assert abs(expected_baseline[0]) > 1e-4
    assert_allclose(shift, [0.0], atol=1e-12)
    assert_allclose(metadata["baseline"], expected_baseline)
    assert_allclose(metadata["estimator"], expected_baseline)
    assert_allclose(metadata["displacement"], expected_baseline)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_dali_semi_analytic_mean_response_orders(order):
    """Tests that metadata contains only the requested response orders."""
    samples = np.array([[-1.0], [0.0], [1.0]])

    _, metadata = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.1],
        mean_method="semi-analytic",
        theta=samples,
        response_order=order,
        return_metadata=True,
    )

    expected = set(range(1, order + 1))

    assert set(metadata["coefficients"]) == expected
    assert set(metadata["terms"]) == expected


def test_dali_semi_analytic_mean_equal_weights_default():
    """Tests that omitted weights are equivalent to equal weights."""
    theta = np.array([
        [-1.0],
        [0.0],
        [0.5],
        [1.5],
    ])

    default = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.1],
        mean_method="semi-analytic",
        theta=theta,
    )

    explicit = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.1],
        mean_method="semi-analytic",
        theta=theta,
        weights=np.ones(len(theta)),
    )

    assert_allclose(default, explicit, atol=1e-12)


def test_dali_semi_analytic_mean_weight_scaling():
    """Tests that rescaling posterior weights leaves the response unchanged."""
    theta, _, cov, _, weights = nonlinear_scalar_reference()

    first = build_dali_mean_shift(
        nonlinear_scalar_model,
        [0.0],
        cov,
        [0.01, -0.005],
        mean_method="semi-analytic",
        theta=theta,
        weights=weights,
    )

    second = build_dali_mean_shift(
        nonlinear_scalar_model,
        [0.0],
        cov,
        [0.01, -0.005],
        mean_method="semi-analytic",
        theta=theta,
        weights=10.0 * weights,
    )

    assert_allclose(first, second, atol=1e-12)


def test_dali_semi_analytic_mean_nonzero_fiducial():
    """Tests displacement relative to a nonzero fiducial parameter."""
    theta0 = np.array([0.5])
    theta = np.array([[0.0], [0.5], [1.0]])
    weights = np.array([1.0, 2.0, 1.0])

    shift, metadata = build_dali_mean_shift(
        scalar_linear_model,
        theta0,
        [[1.0]],
        [0.05],
        mean_method="semi-analytic",
        theta=theta,
        weights=weights,
        return_metadata=True,
    )

    assert_allclose(metadata["baseline"], [0.5])
    assert_allclose(metadata["estimator"], [0.5] + shift)
    assert_allclose(metadata["displacement"], shift)


@pytest.mark.parametrize("order", [0, -1, 4, True, 1.5])
def test_dali_map_shift_invalid_order(order):
    """Tests rejection of unsupported MAP expansion orders."""
    with pytest.raises((TypeError, ValueError), match="expansion_order"):
        build_dali_map_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            expansion_order=order,
        )


@pytest.mark.parametrize("order", [0, -1, 5, True, 1.5])
def test_dali_semi_analytic_invalid_response_order(order):
    """Tests rejection of unsupported mean-response orders."""
    with pytest.raises(ValueError, match="response_order"):
        build_dali_mean_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            mean_method="semi-analytic",
            theta=np.array([[0.0], [1.0]]),
            response_order=order,
        )


@pytest.mark.parametrize("order", [0, -1, 5, True, 1.5])
def test_dali_semi_analytic_invalid_forecast_order(order):
    """Tests rejection of unsupported DALI forecast orders."""
    with pytest.raises(ValueError, match="forecast_order"):
        build_dali_mean_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            mean_method="semi-analytic",
            theta=np.array([[0.0], [1.0]]),
            forecast_order=order,
        )


def test_dali_semi_analytic_requires_theta():
    """Tests that semi-analytic mean forecasts require posterior nodes."""
    with pytest.raises(ValueError, match="theta is required"):
        build_dali_mean_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            mean_method="semi-analytic",
        )


@pytest.mark.parametrize(
    "weights",
    [
        np.array([-1.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([np.nan, 1.0]),
        np.array([1.0]),
    ],
)
def test_dali_semi_analytic_invalid_weights(weights):
    """Tests rejection of invalid posterior weights."""
    with pytest.raises(ValueError, match="weights|theta must"):
        build_dali_mean_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            mean_method="semi-analytic",
            theta=np.array([[0.0], [1.0]]),
            weights=weights,
        )


def test_dali_mean_shift_invalid_method():
    """Tests rejection of unsupported mean approximation methods."""
    with pytest.raises(ValueError, match="mean_method"):
        build_dali_mean_shift(
            scalar_linear_model,
            [0.0],
            [[1.0]],
            [0.1],
            mean_method="response",
        )


def test_dali_analytic_mean_ignores_samples():
    """Tests that posterior samples are not required by the analytic method."""
    expected = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.1],
        mean_method="analytic",
    )

    actual = build_dali_mean_shift(
        scalar_linear_model,
        [0.0],
        [[1.0]],
        [0.1],
        mean_method="analytic",
        theta=np.array([[0.0], [1.0]]),
        weights=np.array([1.0, 2.0]),
    )

    assert_allclose(actual, expected)


def test_dali_map_shift_accepts_column_mismatch():
    """Tests that MAP shifts accept column-vector data mismatches."""
    delta_nu = np.array([0.1, -0.05])

    flat = build_dali_map_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        delta_nu,
    )

    column = build_dali_map_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        delta_nu[:, None],
    )

    assert_allclose(flat, column)


def test_dali_mean_shift_accepts_column_mismatch():
    """Tests that analytic mean shifts accept column-vector mismatches."""
    delta_nu = np.array([0.1, -0.05])

    flat = build_dali_mean_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        delta_nu,
    )

    column = build_dali_mean_shift(
        linear_model,
        np.zeros(2),
        np.eye(2),
        delta_nu[:, None],
    )

    assert_allclose(flat, column)


def test_dali_bias_reuses_cached_derivatives():
    """Tests that cached derivatives avoid additional model evaluations."""
    theta0 = np.zeros(2)
    cov = np.eye(2)
    delta_nu = np.array([0.2, -0.1])
    derivatives = {}

    reference = build_dali_bias_tensor(
        quadratic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
        derivatives=derivatives,
    )

    assert set(derivatives) == {1, 2, 3}

    calls = []

    def cached_model(theta):
        calls.append(np.asarray(theta).copy())
        if len(calls) > 1:
            raise RuntimeError("Cached derivatives should not be recomputed.")
        return quadratic_model(theta)

    result = build_dali_bias_tensor(
        cached_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
        derivatives=derivatives,
    )

    assert len(calls) == 1
    assert_allclose(calls[0], theta0)

    for order in (1, 2, 3):
        assert_allclose(result[order], reference[order])


def test_dali_map_shift_singular_fisher():
    """Tests the MAP shift against an independent pseudoinverse solution."""

    def model(theta):
        x, y = theta
        return np.array([x + y, 2.0 * (x + y)])

    theta0 = np.zeros(2)
    cov = np.eye(2)
    delta_nu = np.array([0.1, 0.2])

    # expected = pseudo_inverse_fisher @ jacobian.T @ delta_nu
    expected = 1/20 * np.array([1, 1])

    shift = build_dali_map_shift(
        model,
        theta0,
        cov,
        delta_nu,
        expansion_order=1,
    )

    assert_allclose(shift, expected, atol=1e-10)


def test_dali_map_response_coefficients_against_exact_optimization():
    """Tests MAP response coefficients against independently optimized modes."""
    theta0 = np.zeros(1)
    cov = np.eye(2)
    delta_nu = np.array([0.4, -0.2])

    _, metadata = build_dali_map_shift(
        nonlinear_scalar_model,
        theta0,
        cov,
        delta_nu,
        expansion_order=3,
        return_metadata=True,
    )

    def exact_map(amplitude):
        def objective(x):
            residual = (
                nonlinear_scalar_model([x])
                - amplitude * delta_nu
            )
            return 0.5 * residual @ residual

        result = minimize_scalar(
            objective,
            bounds=(-0.5, 0.5),
            method="bounded",
            options={"xatol": 1e-14},
        )
        assert result.success
        return result.x

    amplitudes = np.linspace(-0.08, 0.08, 17)
    exact = np.array([exact_map(a) for a in amplitudes])

    design = np.column_stack([
        amplitudes**n for n in range(6)
    ])
    coefficients = np.linalg.lstsq(design, exact, rcond=None)[0]

    for order in (1, 2, 3):
        assert_allclose(
            metadata["terms"][order][0],
            coefficients[order],
            atol=1e-4,
            rtol=1e-3,
        )


def test_build_dali_bias_tensor_fourth_order_reference():
    """Tests fourth-order mismatch tensors against analytic derivatives."""

    def model(theta):
        x, y = theta
        return np.array([
            x**4 + 2.0 * x**3 * y,
            3.0 * x**2 * y**2 + y**4,
        ])

    theta0 = np.zeros(2)
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias_tensor(
        model,
        theta0,
        cov,
        delta_nu,
        bias_order=4,
    )

    expected = np.zeros((2, 2, 2, 2))

    expected[0, 0, 0, 0] = 48.0
    expected[1, 1, 1, 1] = -24.0

    for axis in range(4):
        index = [0, 0, 0, 0]
        index[axis] = 1
        expected[tuple(index)] = 24.0

    for i in range(4):
        for j in range(i + 1, 4):
            index = [0, 0, 0, 0]
            index[i] = 1
            index[j] = 1
            expected[tuple(index)] = -12.0

    assert_allclose(bias[4], expected, atol=1e-5)


def test_dali_analytic_mean_mismatched_tensor_orders():
    """Tests rejection of inconsistent DALI and bias tensor orders."""
    dali = {1: (), 2: (), 3: ()}
    bias = {1: np.zeros(1), 2: np.zeros((1, 1))}

    with pytest.raises(ValueError, match="same tensor orders"):
        _mean_map_offset(dali, bias, np.zeros(1))
