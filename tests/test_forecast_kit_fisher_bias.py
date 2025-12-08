"""Tests for Fisher-bias estimation in forecast_kit/fisher.py."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.fisher import build_delta_nu, build_fisher_bias

A_LINEAR = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)  # this is a phantom use of theta
    return np.zeros(3, dtype=float)


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    _ = np.atleast_1d(theta)  # this is a phantom use of theta
    return np.zeros(2, dtype=float)


def linear_model_matrix(theta, mat):
    """Linear model y = M @ theta for a given matrix M."""
    theta = np.atleast_1d(theta)
    return mat @ theta


linear_model = partial(linear_model_matrix, mat=A_LINEAR)


def check_linear_bias(theta0, cov, fisher, delta_nu, method="finite", **dk_kwargs):
    """Checks that the bias computed for a linear model matches the analytic result."""
    bias_vec, delta_theta = build_fisher_bias(
        function=linear_model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        method=method,
        **dk_kwargs,
    )

    cinv_delta = np.linalg.solve(cov, delta_nu)
    expected_bias = A_LINEAR.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(delta_theta, expected_delta_theta, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "theta0,cov,fisher,delta_nu",
    [
        (
            np.array([0.1, -0.2]),
            np.diag([2.0, 1.0, 0.5]),
            np.array([[2.0, 0.1], [0.1, 1.5]]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([0.0, 0.5]),
            np.array([[2.0, 0.3, 0.0],
                      [0.3, 1.5, 0.1],
                      [0.0, 0.1, 0.8]]),
            np.array([[1.0, 0.2], [0.2, 1.3]]),
            np.array([0.5, -1.0, 2.0]),
        ),
    ],
)
def test_build_fisher_bias_linear_cases(theta0, cov, fisher, delta_nu):
    """Tests build_fisher_bias on linear model with various inputs."""
    check_linear_bias(theta0, cov, fisher, delta_nu, method="finite")


@pytest.mark.parametrize(
    "fisher, delta_nu, expected_exception",
    [
        (np.ones((2, 3)), np.zeros(3), ValueError),
        (np.eye(2), np.zeros(2), ValueError),
        (np.eye(2), np.array([0.0, np.nan, 1.0]), FloatingPointError),
    ],
)
def test_build_fisher_bias_exceptions(fisher, delta_nu, expected_exception):
    """Tests that invalid Fisher / delta_nu inputs raise appropriate errors."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    with pytest.raises(expected_exception):
        build_fisher_bias(
            function=linear_model,
            theta0=theta0,
            cov=cov,
            fisher_matrix=fisher,
            delta_nu=delta_nu,
        )


def test_build_fisher_bias_accepts_2d_delta_nu_and_flattens():
    """Tests that 2D delta_nu inputs are flattened correctly."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(4)

    mat = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )

    linear_model_4 = partial(linear_model_matrix, mat=mat)

    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    delta_nu_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2) -> flat length 4

    bias_vec, delta_theta = build_fisher_bias(
        function=linear_model_4,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta_nu_2d,
        method="finite",
    )


    assert bias_vec.shape == (2,)
    assert delta_theta.shape == (2,)
    assert np.all(np.isfinite(bias_vec))
    assert np.all(np.isfinite(delta_theta))


def test_build_delta_nu_1d_ok():
    """Tests that build_delta_nu works for 1D inputs."""
    cov = np.eye(4)

    data_with = np.array([1.0, 2.0, 3.0, 4.0])
    data_without = np.array([0.5, 1.0, 1.5, 2.0])

    delta = build_delta_nu(
        cov=cov,
        data_with=data_with,
        data_without=data_without,
    )

    expected = data_with - data_without
    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected)


def test_build_delta_nu_2d_flattens_row_major():
    """Tests that build_delta_nu works for 2D inputs, flattening in row-major order."""
    cov = np.eye(4)

    data_with = np.array([[1.0, 2.0], [3.0, 4.0]])
    data_without = np.array([[0.5, 1.5], [2.0, 2.5]])

    delta = build_delta_nu(
        cov=cov,
        data_with=data_with,
        data_without=data_without,
    )

    expected_2d = data_with - data_without
    expected_flat = expected_2d.ravel(order="C")

    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected_flat)


@pytest.mark.parametrize(
    "data_with, data_without, cov, expected_exception",
    [
        # shape mismatch between with/without
        (np.zeros(3), np.zeros(4), np.eye(3), ValueError),
        # ndim > 2
        (np.zeros((2, 2, 1)), np.zeros((2, 2, 1)), np.eye(4), ValueError),
        # wrong length vs number of observables (flattened length 4 vs n_obs=5)
        (np.zeros((2, 2)), np.zeros((2, 2)), np.eye(5), ValueError),
        # non-finite values
        (np.array([0.0, np.nan, 1.0]), np.zeros(3), np.eye(3), FloatingPointError),
    ],
)
def test_build_delta_nu_exceptions(data_with, data_without, cov, expected_exception):
    """Tests that invalid inputs to build_delta_nu raise appropriate errors."""
    with pytest.raises(expected_exception):
        build_delta_nu(
            cov=cov,
            data_with=data_with,
            data_without=data_without,
        )


@pytest.mark.parametrize(
    "method, dk_kwargs",
    [
        # Default methods with their default settings
        ("adaptive", {}),
        ("finite", {}),
        ("local_polynomial", {}),

        # Finite-difference with various extrapolations (no levels)
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "richardson"}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "ridders"}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "gauss-richardson"}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "gre"}),

        # Finite-difference with various extrapolations, stencil 7 (no levels)
        ("finite", {"stepsize": 1e-2, "num_points": 7, "extrapolation": "richardson"}),
        ("finite", {"stepsize": 1e-2, "num_points": 7, "extrapolation": "ridders"}),
        ("finite", {"stepsize": 1e-2, "num_points": 7, "extrapolation": "gauss-richardson"}),
        ("finite", {"stepsize": 1e-2, "num_points": 7, "extrapolation": "gre"}),

        # Finite-difference with extrapolations + explicit levels
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "richardson", "levels": 3}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "ridders", "levels": 3}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "gauss-richardson", "levels": 3}),
        ("finite", {"stepsize": 1e-2, "num_points": 5, "extrapolation": "gre", "levels": 3}),
    ],
)
def test_build_fisher_bias_methods_and_extrapolations(method, dk_kwargs):
    """Tests that all supported methods + finite-diff extrapolations work and match the analytic bias."""
    theta0 = np.array([0.1, -0.2])
    cov = np.diag([2.0, 1.0, 0.5])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    delta_nu = np.array([1.0, 2.0, 3.0])

    check_linear_bias(theta0, cov, fisher, delta_nu, method=method, **dk_kwargs)
