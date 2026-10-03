"""Unit tests for ``derivkit.forecasting.dali_bias``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.forecasting.dali_bias import build_dali_bias


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


def test_build_dali_bias_linear_model():
    """Tests first-order bias tensor for a linear model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_linear_model_higher_orders_zero():
    """Tests that higher-order bias tensors vanish for a linear model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_quadratic_model():
    """Tests first- and second-order bias tensors for a quadratic model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_cubic_model():
    """Tests bias tensors through third order for a cubic model."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_non_diagonal_covariance():
    """Tests inverse-covariance weighting for correlated observables."""
    theta0 = np.array([0.0, 0.0])
    cov = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )
    delta_nu = np.array([1.0, 2.0])

    bias = build_dali_bias(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=1,
    )

    expected = np.array([-0.2, 3.0])

    assert_allclose(bias[1], expected)


def test_build_dali_bias_zero_mismatch():
    """Tests that a zero data mismatch produces zero bias tensors."""
    theta0 = np.array([0.0, 0.0])
    cov = np.array(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )
    delta_nu = np.zeros(2)

    bias = build_dali_bias(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )

    assert_allclose(bias[1], np.zeros(2))
    assert_allclose(bias[2], np.zeros((2, 2)))
    assert_allclose(bias[3], np.zeros((2, 2, 2)))


def test_build_dali_bias_scales_linearly_with_mismatch():
    """Tests linear scaling of bias tensors with the data mismatch."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([0.5, -1.5])

    bias = build_dali_bias(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
    )
    scaled_bias = build_dali_bias(
        cubic_model,
        theta0,
        cov,
        4.0 * delta_nu,
        bias_order=3,
    )

    for order in range(1, 4):
        assert_allclose(scaled_bias[order], 4.0 * bias[order], atol=1e-10)


def test_build_dali_bias_tensor_symmetry():
    """Tests symmetry of higher-order bias tensors over parameter axes."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([2.0, -1.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_nonzero_expansion_point():
    """Tests bias tensors when derivatives are evaluated away from the origin."""
    theta0 = np.array([1.0, -1.0])
    cov = np.eye(2)
    delta_nu = np.array([1.0, 0.0])

    bias = build_dali_bias(
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


def test_build_dali_bias_accepts_list_inputs():
    """Tests that array-like inputs are accepted."""
    bias = build_dali_bias(
        linear_model,
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [2.0, -1.0],
        bias_order=1,
    )

    assert_allclose(bias[1], np.array([5.0, 2.0]))


def test_build_dali_bias_delta_nu_column_vector():
    """Tests that the data mismatch is flattened to one dimension."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([[2.0], [-1.0]])

    bias = build_dali_bias(
        linear_model,
        theta0,
        cov,
        delta_nu,
        bias_order=1,
    )

    assert_allclose(bias[1], np.array([5.0, 2.0]))


@pytest.mark.parametrize("bias_order", [1, 2, 3, 4])
def test_build_dali_bias_requested_order(bias_order):
    """Tests that only bias tensors through the requested order are returned."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    delta_nu = np.array([1.0, 1.0])

    bias = build_dali_bias(
        cubic_model,
        theta0,
        cov,
        delta_nu,
        bias_order=bias_order,
    )

    assert set(bias) == set(range(1, bias_order + 1))


@pytest.mark.parametrize("bias_order", [0, -1, 5])
def test_build_dali_bias_unsupported_order_raises(bias_order):
    """Tests that unsupported bias orders raise ValueError."""
    with pytest.raises(ValueError, match="bias_order=.* is not supported"):
        build_dali_bias(
            linear_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 1.0],
            bias_order=bias_order,
        )


def test_build_dali_bias_invalid_order_type_raises():
    """Tests that an invalid bias-order type raises TypeError."""
    with pytest.raises(TypeError, match="bias_order must be an int"):
        build_dali_bias(
            linear_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 1.0],
            bias_order="invalid",
        )


def test_build_dali_bias_empty_theta0_raises():
    """Tests that an empty expansion point raises ValueError."""
    with pytest.raises(ValueError, match="theta0 must be non-empty 1D"):
        build_dali_bias(
            linear_model,
            [],
            np.eye(2),
            [1.0, 1.0],
            bias_order=1,
        )


def test_build_dali_bias_mismatch_size_raises():
    """Tests that the mismatch size must match the number of observables."""
    with pytest.raises(ValueError, match="Expected 2 elements in delta_nu"):
        build_dali_bias(
            linear_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 2.0, 3.0],
            bias_order=1,
        )


def test_build_dali_bias_model_output_size_raises():
    """Tests that model output size must match the covariance dimension."""

    def three_observable_model(theta):
        x, y = theta
        return np.array([x, y, x + y])

    with pytest.raises(ValueError, match="Expected 2 observables from model"):
        build_dali_bias(
            three_observable_model,
            [0.0, 0.0],
            np.eye(2),
            [1.0, 1.0],
            bias_order=1,
        )


def test_build_dali_bias_invalid_covariance_shape_raises():
    """Tests that a non-square covariance matrix raises ValueError."""
    with pytest.raises(ValueError):
        build_dali_bias(
            linear_model,
            [0.0, 0.0],
            np.ones((2, 3)),
            [1.0, 1.0],
            bias_order=1,
        )
