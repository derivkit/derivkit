"""Unit tests for ``derivkit.forecasting.dali_bias``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from derivkit.forecasting.dali_bias import (
    build_dali_bias_shifts,
    build_dali_bias_tensor,
    build_dali_map_shift,
    build_dali_map_shift_from_bias,
    build_dali_mean_map_offset_from_bias,
    build_dali_mean_response,
    build_dali_mean_response_coefficients,
    build_dali_mean_response_displacement,
    build_dali_mean_response_shift,
    build_dali_mean_shift,
    build_dali_mean_shift_from_bias,
    build_dali_response_posterior_mean,
    build_dali_response_potential,
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



def scalar_linear_model(theta):
    """Returns a one-parameter linear model."""
    return np.array([theta[0]])


def scalar_quadratic_model(theta):
    """Returns a one-parameter quadratic model."""
    x = theta[0]
    return np.array([x + 0.5 * x**2])


def symmetric_reference():
    """Returns a symmetric, normalized discrete reference posterior."""
    theta = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    weights = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    return theta, weights


def test_build_dali_response_potential_polynomial():
    """Tests the potential against a hand-evaluated cubic polynomial."""
    theta = np.array([[2.0], [-1.0]])
    theta0 = np.array([1.0])
    bias = {
        1: np.array([2.0]),
        2: np.array([[6.0]]),
        3: np.array([[[12.0]]]),
    }

    potential = build_dali_response_potential(theta, theta0, bias)

    assert_allclose(potential, np.array([7.0, -8.0]))


def test_build_dali_response_potential_zero_mismatch():
    """Tests that zero mismatch tensors produce no posterior deformation."""
    theta = np.array([[-2.0], [0.0], [3.0]])
    bias = {
        1: np.zeros(1),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    potential = build_dali_response_potential(theta, [0.5], bias)

    assert_allclose(potential, np.zeros(3))


def test_build_dali_mean_response_coefficients_symmetric_reference():
    """Tests all four coefficients against exact discrete moments."""
    theta, weights = symmetric_reference()
    bias = {
        1: np.array([1.0]),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0], bias, order=4
    )

    assert_allclose(coefficients["mean0"], [0.0], atol=1e-14)
    assert_allclose(coefficients["first"], [1.0], atol=1e-14)
    assert_allclose(coefficients["second"], [0.0], atol=1e-14)
    assert_allclose(coefficients["third"], [-1.0 / 12.0], atol=1e-14)
    assert_allclose(coefficients["fourth"], [0.0], atol=1e-14)


def test_build_dali_mean_response_coefficients_zero_mismatch():
    """Tests that every response coefficient vanishes without a systematic."""
    theta = np.array([[-1.0], [0.0], [2.0]])
    weights = np.array([1.0, 2.0, 3.0])
    bias = {
        1: np.zeros(1),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0], bias, order=4
    )

    assert_allclose(coefficients["mean0"], [5.0 / 6.0])
    for name in ("first", "second", "third", "fourth"):
        assert_allclose(coefficients[name], [0.0], atol=1e-14)


def test_build_dali_mean_response_coefficients_nonlinear_amplitude():
    """Tests higher amplitude derivatives against an exact Gaussian response."""
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    theta = (np.sqrt(2.0) * nodes)[:, None]
    bias = {
        1: np.array([2.0]),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }
    bias_second = {
        1: np.array([3.0]),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0], bias,
        bias_second=bias_second, order=4,
    )

    assert_allclose(coefficients["first"], [2.0], atol=1e-12)
    assert_allclose(coefficients["second"], [1.5], atol=1e-12)
    assert_allclose(coefficients["third"], [0.0], atol=1e-12)
    assert_allclose(coefficients["fourth"], [0.0], atol=1e-12)


def test_build_dali_mean_response_coefficients_two_parameters():
    """Tests the response of an independent two-parameter Gaussian posterior."""
    nodes, weights_1d = np.polynomial.hermite.hermgauss(12)
    x, y = np.meshgrid(np.sqrt(2.0) * nodes, 2.0 * nodes, indexing="ij")
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing="ij")
    theta = np.column_stack([x.ravel(), y.ravel()])
    weights = (wx * wy).ravel()

    bias = {
        1: np.array([1.5, -2.0]),
        2: np.zeros((2, 2)),
        3: np.zeros((2, 2, 2)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0, 0.0], bias, order=4
    )

    assert_allclose(coefficients["first"], [1.5, -4.0], atol=1e-12)
    for name in ("second", "third", "fourth"):
        assert_allclose(coefficients[name], [0.0, 0.0], atol=1e-12)


def test_build_dali_mean_response_against_exact_reweighting():
    """Tests fourth-order response against independently reweighted samples."""
    theta, weights = symmetric_reference()
    bias = {
        1: np.array([1.0]),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0], bias, order=4
    )

    amplitude = 0.1
    tilted_weights = weights * np.exp(amplitude * theta[:, 0])
    exact_mean = np.average(theta[:, 0], weights=tilted_weights)
    predicted = build_dali_mean_response_shift(
        coefficients, amplitude, order=4
    )

    assert_allclose(predicted, [exact_mean], atol=2e-7)


def test_build_dali_mean_response_shift_polynomial():
    """Tests evaluation of known response coefficients at finite amplitude."""
    coefficients = {
        "mean0": np.array([1.0]),
        "theta0": np.array([0.0]),
        "first": np.array([2.0]),
        "second": np.array([3.0]),
        "third": np.array([4.0]),
        "fourth": np.array([5.0]),
    }

    shift = build_dali_mean_response_shift(coefficients, 0.5, order=4)

    assert_allclose(shift, [2.5625])


def test_build_dali_mean_response_shift_zero_amplitude():
    """Tests that the systematic shift vanishes at zero amplitude."""
    coefficients = {
        "mean0": np.array([1.0, -2.0]),
        "first": np.array([3.0, 4.0]),
        "second": np.array([1.0, 1.0]),
        "third": np.array([2.0, 2.0]),
        "fourth": np.array([5.0, 5.0]),
    }

    assert_allclose(
        build_dali_mean_response_shift(coefficients, 0.0),
        [0.0, 0.0],
    )


def test_build_dali_mean_response_displacement_baseline():
    """Tests that displacement includes the unbiased mean offset."""
    coefficients = {
        "mean0": np.array([2.0]),
        "theta0": np.array([0.5]),
        "first": np.array([3.0]),
    }

    displacement = build_dali_mean_response_displacement(
        coefficients, 0.2, order=1
    )

    assert_allclose(displacement, [2.1])


def test_build_dali_response_posterior_mean_components():
    """Tests posterior mean, systematic shift, and fiducial displacement."""
    coefficients = {
        "mean0": np.array([2.0]),
        "theta0": np.array([0.5]),
        "first": np.array([3.0]),
    }

    result = build_dali_response_posterior_mean(
        coefficients, 0.2, order=1
    )

    assert_allclose(result["mean"], [2.6])
    assert_allclose(result["shift"], [0.6])
    assert_allclose(result["displacement"], [2.1])


@pytest.mark.parametrize("order", [0, 5])
def test_build_dali_mean_response_invalid_order(order):
    """Tests rejection of unsupported response orders."""
    theta, weights = symmetric_reference()
    bias = {
        1: np.ones(1),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    with pytest.raises(ValueError, match="order must be"):
        build_dali_mean_response_coefficients(
            theta, weights, [0.0], bias, order=order
        )

    with pytest.raises(ValueError, match="order must be"):
        build_dali_mean_response_shift(
            {"mean0": np.zeros(1)}, 0.1, order=order
        )


def test_build_dali_mean_response_missing_coefficient():
    """Tests rejection of an unavailable response coefficient."""
    coefficients = {
        "mean0": np.zeros(1),
        "first": np.ones(1),
    }

    with pytest.raises(ValueError, match="not available"):
        build_dali_mean_response_shift(coefficients, 0.1, order=2)


def test_build_dali_map_shift_linear_gaussian():
    """Tests the MAP shift against the exact linear-Gaussian solution."""
    theta0 = np.array([0.0, 0.0])
    delta_nu = np.array([0.2, -0.1])

    shift = build_dali_map_shift(
        linear_model, theta0, np.eye(2), delta_nu
    )

    expected = np.linalg.solve(
        np.array([[2.0, 3.0], [-1.0, 4.0]]),
        delta_nu,
    )

    assert_allclose(shift, expected, atol=1e-10)


def test_build_dali_mean_shift_linear_gaussian():
    """Tests that the analytical mean equals the exact Gaussian MAP."""
    theta0 = np.zeros(2)
    delta_nu = np.array([0.2, -0.1])
    expected = np.linalg.solve(
        np.array([[2.0, 3.0], [-1.0, 4.0]]),
        delta_nu,
    )

    shift = build_dali_mean_shift(
        linear_model, theta0, np.eye(2), delta_nu
    )

    assert_allclose(shift, expected, atol=1e-10)


def test_build_dali_mean_response_linear_gaussian():
    """Tests the full posterior response against an exact Gaussian mean shift."""
    nodes, weights_1d = np.polynomial.hermite.hermgauss(12)
    theta = (np.sqrt(2.0) * nodes)[:, None]

    shift = build_dali_mean_response(
        scalar_linear_model, [0.0], [[1.0]], [0.3],
        theta, weights_1d, amplitude=0.5,
    )

    assert_allclose(shift, [0.15], atol=1e-10)


@pytest.mark.parametrize("model", [linear_model, quadratic_model, cubic_model])
def test_build_dali_bias_shifts_zero_mismatch(model):
    """Tests that all three systematic shifts vanish for identical data."""
    theta = np.array([
        [-0.5, -0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.5, 0.5],
    ])
    weights = np.ones(4)

    shifts = build_dali_bias_shifts(
        model, np.zeros(2), np.eye(2), np.zeros(2),
        theta, weights,
    )

    assert set(shifts) == {"map", "analytic_mean", "response_mean"}
    for shift in shifts.values():
        assert_allclose(shift, np.zeros(2), atol=1e-10)


def test_build_dali_bias_shifts_linear_gaussian():
    """Tests all three methods against the same exact Gaussian solution."""
    nodes, weights_1d = np.polynomial.hermite.hermgauss(12)
    theta = (np.sqrt(2.0) * nodes)[:, None]
    delta_nu = np.array([0.3])
    amplitude = 0.5

    shifts = build_dali_bias_shifts(
        scalar_linear_model, [0.0], [[1.0]], delta_nu,
        theta, weights_1d, amplitude=amplitude,
    )

    for shift in shifts.values():
        assert_allclose(shift, [0.15], atol=1e-10)


def test_build_dali_bias_shifts_zero_amplitude():
    """Tests that all three methods vanish at zero systematic amplitude."""
    theta, weights = symmetric_reference()

    shifts = build_dali_bias_shifts(
        scalar_quadratic_model, [0.0], [[1.0]], [0.2],
        theta, weights, amplitude=0.0,
    )

    for shift in shifts.values():
        assert_allclose(shift, [0.0], atol=1e-10)


def scalar_dali_tensors(fisher=2.0, d1=0.1, d2=0.04,
                        t1=0.03, t2=0.02, t3=0.01):
    """Returns prescribed one-parameter DALI tensors."""
    return {
        1: (np.array([[fisher]]),),
        2: (np.full((1, 1, 1), d1), np.full((1,) * 4, d2)),
        3: (np.full((1,) * 4, t1), np.full((1,) * 5, t2),
            np.full((1,) * 6, t3)),
    }


def scalar_bias_tensors(b1=0.4, b2=0.2, b3=0.06):
    """Returns prescribed one-parameter systematic bias tensors."""
    return {
        1: np.array([b1]),
        2: np.array([[b2]]),
        3: np.array([[[b3]]]),
    }


@pytest.mark.parametrize(("order", "expected"), [
    (1, 0.2),
    (2, 0.217),
    (3, 0.21863),
])
def test_dali_map_shift_scalar_expansion(order, expected):
    """Tests each MAP expansion order against analytical scalar values."""
    shift = build_dali_map_shift_from_bias(
        scalar_dali_tensors(), scalar_bias_tensors(),
        expansion_order=order,
    )

    assert_allclose(shift, [expected], atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_dali_map_shift_zero_bias(order):
    """Tests that nonlinear posterior geometry alone causes no MAP shift."""
    bias = scalar_bias_tensors(b1=0.0, b2=0.0, b3=0.0)

    shift = build_dali_map_shift_from_bias(
        scalar_dali_tensors(), bias, expansion_order=order,
    )

    assert_allclose(shift, [0.0], atol=1e-14)


def test_dali_mean_map_offset_cubic_asymmetry():
    """Tests the analytical mean–MAP offset for a cubic posterior."""
    fisher = 2.0
    d1 = 0.1
    dali = scalar_dali_tensors(
        fisher=fisher, d1=d1, d2=0.0, t1=0.0, t2=0.0, t3=0.0,
    )
    bias = scalar_bias_tensors(b1=0.0, b2=0.0, b3=0.0)

    offset = build_dali_mean_map_offset_from_bias(dali, bias, np.zeros(1))

    assert_allclose(offset, [-1.5 * d1 / fisher**2], atol=1e-12)


def test_dali_mean_shift_removes_unbiased_asymmetry():
    """Tests that an unbiased non-Gaussian mean offset is not a systematic shift."""
    dali = scalar_dali_tensors(
        d1=0.1, d2=0.0, t1=0.0, t2=0.0, t3=0.0,
    )
    bias = scalar_bias_tensors(b1=0.0, b2=0.0, b3=0.0)

    shift = build_dali_mean_shift_from_bias(dali, bias, np.zeros(1))

    assert_allclose(shift, [0.0], atol=1e-14)



def nonlinear_scalar_model(theta):
    """Returns a quadratic model with an asymmetric posterior."""
    x = theta[0]
    return np.array([x + 0.3 * x**2, 0.7 * x + 0.2 * x**2])


def nonlinear_scalar_reference():
    """Returns a normalized numerical reference for a nonlinear posterior."""
    theta = np.linspace(-2.0, 2.0, 4001)
    predictions = np.column_stack([
        theta + 0.3 * theta**2,
        0.7 * theta + 0.2 * theta**2,
    ])
    precision = np.array([[1.0, 0.25], [0.25, 1.5]])
    log_weights = -0.5 * np.einsum(
        "ni,ij,nj->n", predictions, precision, predictions
    )
    weights = np.exp(log_weights - logsumexp(log_weights))
    return theta[:, None], predictions, precision, weights


@pytest.mark.parametrize("amplitude", [0.01, 0.02, 0.04])
def test_dali_map_shift_against_nonlinear_optimization(amplitude):
    """Tests the MAP expansion against direct nonlinear likelihood optimization."""
    theta0 = np.array([0.0])
    cov = np.array([[1.0, -1.0 / 6.0], [-1.0 / 6.0, 2.0 / 3.0]])
    delta_nu = np.array([0.4, -0.2])

    def negative_log_likelihood(x):
        residual = nonlinear_scalar_model([x]) - amplitude * delta_nu
        return 0.5 * residual @ np.linalg.solve(cov, residual)

    optimum = minimize_scalar(
        negative_log_likelihood, bounds=(-0.5, 0.5), method="bounded",
        options={"xatol": 1e-14},
    )
    assert optimum.success

    predicted = build_dali_map_shift(
        nonlinear_scalar_model, theta0, cov, amplitude * delta_nu,
        expansion_order=3,
    )

    assert_allclose(predicted, [optimum.x], atol=2e-5, rtol=0)


def test_dali_map_expansion_convergence():
    """Tests that higher MAP orders improve the small-systematic expansion."""
    cov = np.eye(2)
    theta0 = np.array([0.0])
    delta_nu = np.array([0.5, 0.2])
    amplitudes = [0.04, 0.02]

    errors = {1: [], 2: [], 3: []}

    for amplitude in amplitudes:
        def negative_log_likelihood(x):
            residual = nonlinear_scalar_model([x]) - amplitude * delta_nu
            return 0.5 * residual @ residual

        optimum = minimize_scalar(
            negative_log_likelihood, bounds=(-0.5, 0.5), method="bounded",
            options={"xatol": 1e-14},
        )
        assert optimum.success

        for order in errors:
            predicted = build_dali_map_shift(
                nonlinear_scalar_model, theta0, cov,
                amplitude * delta_nu, expansion_order=order,
            )
            errors[order].append(abs(predicted[0] - optimum.x))

    for order in errors:
        assert errors[order][1] < errors[order][0]

    assert errors[2][0] < errors[1][0]
    assert errors[3][0] < errors[2][0]


@pytest.mark.parametrize("amplitude", [0.02, 0.05, 0.1])
def test_dali_mean_response_against_direct_integration(amplitude):
    """Tests the fourth-order mean response against direct posterior integration."""
    theta, predictions, precision, weights = nonlinear_scalar_reference()
    delta_nu = np.array([0.3, -0.2])
    cov = np.linalg.inv(precision)

    unbiased_mean = np.average(theta[:, 0], weights=weights)
    residual = predictions - amplitude * delta_nu
    log_biased = -0.5 * np.einsum(
        "ni,ij,nj->n", residual, precision, residual
    )
    biased_weights = np.exp(log_biased - logsumexp(log_biased))
    exact_shift = np.average(theta[:, 0], weights=biased_weights) - unbiased_mean

    predicted = build_dali_mean_response(
        nonlinear_scalar_model, [0.0], cov, delta_nu,
        theta, weights, amplitude=amplitude, order=4,
    )

    assert_allclose(predicted, [exact_shift], atol=2e-7, rtol=0)


def test_dali_mean_response_nongaussian_convergence():
    """Tests fourth-order convergence for a non-Gaussian posterior."""
    theta, predictions, precision, weights = nonlinear_scalar_reference()
    delta_nu = np.array([0.3, -0.2])
    cov = np.linalg.inv(precision)
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
            predicted = build_dali_mean_response(
                nonlinear_scalar_model, [0.0], cov, delta_nu,
                theta, weights, amplitude=amplitude, order=order,
            )
            errors[amplitude][order] = abs(predicted[0] - exact)

    for order in (1, 2, 3, 4):
        assert errors[0.1][order] < errors[0.2][order]

    assert errors[0.2][4] < errors[0.2][3]
    assert errors[0.2][3] < errors[0.2][2]


def test_dali_mean_response_asymmetric_discrete_reference():
    """Tests four nonzero response coefficients against exact finite sums."""
    theta = np.array([[-1.0], [0.0], [2.0]])
    weights = np.array([2.0, 3.0, 5.0])
    bias = {
        1: np.array([1.0]),
        2: np.zeros((1, 1)),
        3: np.zeros((1, 1, 1)),
    }

    coefficients = build_dali_mean_response_coefficients(
        theta, weights, [0.0], bias, order=4,
    )

    assert_allclose(coefficients["mean0"], [0.8])
    assert_allclose(coefficients["first"], [1.56])
    assert_allclose(coefficients["second"], [-0.228])
    assert_allclose(coefficients["third"], [-0.6736])
    assert_allclose(coefficients["fourth"], [0.18668])


def test_dali_mean_map_offset_quintic():
    """Tests the quintic contribution to the analytical mean–MAP offset."""
    dali = scalar_dali_tensors(
        fisher=2.0, d1=0.0, d2=0.0,
        t1=0.0, t2=0.12, t3=0.0,
    )
    bias = scalar_bias_tensors(b1=0.0, b2=0.0, b3=0.0)

    offset = build_dali_mean_map_offset_from_bias(
        dali, bias, np.zeros(1),
    )

    assert_allclose(offset, [-0.01875], atol=1e-12)



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


@pytest.mark.parametrize("amplitude", [0.01, 0.02, 0.04])
def test_dali_map_shift_coupled_nonlinear(amplitude):
    """Tests both nonlinear MAP components against direct optimization."""
    cov = np.array([[0.16, 0.045], [0.045, 0.25]])
    delta_nu = np.array([0.3, -0.2])

    def negative_log_likelihood(theta):
        residual = coupled_quadratic_model(theta) - amplitude * delta_nu
        return 0.5 * residual @ np.linalg.solve(cov, residual)

    optimum = minimize(
        negative_log_likelihood, np.zeros(2), method="BFGS",
        options={"gtol": 1e-12},
    )

    assert np.linalg.norm(optimum.jac) < 1e-6

    predicted = build_dali_map_shift(
        coupled_quadratic_model, np.zeros(2), cov,
        amplitude * delta_nu, expansion_order=3,
    )

    assert_allclose(predicted, optimum.x, atol=2e-5, rtol=0)


@pytest.mark.parametrize("amplitude", [0.02, 0.05, 0.1])
def test_dali_mean_response_coupled_nonlinear(amplitude):
    """Tests both response components against direct two-dimensional integration."""
    theta, predictions, cov, precision, weights = coupled_quadratic_reference()
    delta_nu = np.array([0.3, -0.2])

    mean0 = np.average(theta, axis=0, weights=weights)
    residual = predictions - amplitude * delta_nu
    log_biased = -0.5 * np.einsum(
        "ni,ij,nj->n", residual, precision, residual
    )
    biased_weights = np.exp(log_biased - logsumexp(log_biased))
    exact_shift = np.average(theta, axis=0, weights=biased_weights) - mean0

    predicted = build_dali_mean_response(
        coupled_quadratic_model, np.zeros(2), cov, delta_nu,
        theta, weights, amplitude=amplitude, order=4,
    )

    assert_allclose(predicted, exact_shift, atol=2e-6, rtol=0)


@pytest.mark.parametrize("nonlinearity", [0.1, 0.05, 0.025])
def test_dali_analytic_mean_shift_weak_nonlinearity(nonlinearity):
    """Tests analytical mean shifts against integration as nonlinearity vanishes."""
    def model(theta):
        x = theta[0]
        return np.array([x + nonlinearity * x**2, 0.8 * x])

    theta = np.linspace(-5.0, 5.0, 20001)
    predictions = np.column_stack([
        theta + nonlinearity * theta**2,
        0.8 * theta,
    ])
    cov = np.eye(2)
    delta_nu = np.array([0.1, -0.05])

    def posterior_mean(mismatch):
        residual = predictions - mismatch
        log_weights = -0.5 * np.sum(residual**2, axis=1)
        weights = np.exp(log_weights - logsumexp(log_weights))
        return np.average(theta, weights=weights)

    exact_shift = posterior_mean(delta_nu) - posterior_mean(np.zeros(2))
    predicted = build_dali_mean_shift(
        model, np.zeros(1), cov, delta_nu,
    )

    assert np.isfinite(predicted[0])
    assert abs(predicted[0] - exact_shift) < 0.02
