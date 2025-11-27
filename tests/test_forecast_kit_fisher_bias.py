"""Tests for Fisher-bias estimation in LikelihoodExpansion."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion

A_LINEAR = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    return np.zeros(3, dtype=float)


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    return np.zeros(2, dtype=float)


def linear_model_matrix(theta, mat):
    """Linear model y = M @ theta for a given matrix M."""
    theta = np.atleast_1d(theta)
    return mat @ theta


linear_model = partial(linear_model_matrix, mat=A_LINEAR)


def check_linear_bias(theta0, cov, fisher, delta_nu, method="finite", **dk_kwargs):
    """Checks that the bias computed for a linear model matches the analytic result."""
    lx = LikelihoodExpansion(function=linear_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
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
    lx = LikelihoodExpansion(function=linear_model, theta0=theta0, cov=cov)

    with pytest.raises(expected_exception):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


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

    lx = LikelihoodExpansion(function=linear_model_4, theta0=theta0, cov=cov)

    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    delta_nu_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2) -> flat length 4

    bias_vec, delta_theta = lx.build_fisher_bias(
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
    theta0 = np.array([0.0])
    cov = np.eye(4)

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    data_with = np.array([1.0, 2.0, 3.0, 4.0])
    data_without = np.array([0.5, 1.0, 1.5, 2.0])

    delta = lx.build_delta_nu(data_with=data_with, data_without=data_without)

    expected = data_with - data_without
    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected)


def test_build_delta_nu_2d_flattens_row_major():
    """Tests that build_delta_nu works for 2D inputs, flattening in row-major order."""
    theta0 = np.array([0.0])
    cov = np.eye(4)

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    data_with = np.array([[1.0, 2.0], [3.0, 4.0]])
    data_without = np.array([[0.5, 1.5], [2.0, 2.5]])

    delta = lx.build_delta_nu(data_with=data_with, data_without=data_without)

    expected_2d = data_with - data_without
    expected_flat = expected_2d.ravel(order="C")

    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected_flat)


def test_build_delta_nu_shape_mismatch_raises():
    """Tests that shape mismatch raises ValueError."""
    theta0 = np.array([0.0])
    cov = np.eye(3)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.zeros(3)
    b = np.zeros(4)

    with pytest.raises(ValueError):
        lx.build_delta_nu(data_with=a, data_without=b)


def test_build_delta_nu_rejects_ndim_greater_than_two():
    """Tests that inputs with ndim > 2 raise ValueError."""
    theta0 = np.array([0.0])
    cov = np.eye(4)

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    a = np.zeros((2, 2, 1))
    b = np.zeros((2, 2, 1))

    with pytest.raises(ValueError):
        lx.build_delta_nu(data_with=a, data_without=b)


def test_build_delta_nu_wrong_length_vs_n_observables_raises():
    """Tests that wrong-length inputs vs number of observables raise ValueError."""
    theta0 = np.array([0.0])
    cov = np.eye(5)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.zeros((2, 2))  # length 4
    b = np.zeros((2, 2))

    with pytest.raises(ValueError):
        lx.build_delta_nu(data_with=a, data_without=b)


def test_build_delta_nu_rejects_nonfinite_values():
    """Tests that non-finite values in inputs raise FloatingPointError."""
    theta0 = np.array([0.0])
    cov = np.eye(3)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.array([0.0, np.nan, 1.0])
    b = np.zeros(3)

    with pytest.raises(FloatingPointError):
        lx.build_delta_nu(data_with=a, data_without=b)


@pytest.mark.parametrize("method", ["adaptive", "finite", "local_polynomial"])
def test_build_fisher_bias_supports_all_methods(method):
    """Tests that all supported differentiation methods work without error."""
    theta0 = np.array([0.1, -0.2])
    cov = np.diag([2.0, 1.0, 0.5])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    delta_nu = np.array([1.0, 2.0, 3.0])

    lx = LikelihoodExpansion(function=linear_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        method=method,
        n_workers=2,
    )

    diag = np.diag(cov)
    cinv_delta = delta_nu / diag
    expected_bias = A_LINEAR.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    assert bias_vec.shape == expected_bias.shape
    assert delta_theta.shape == expected_delta_theta.shape
    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(delta_theta, expected_delta_theta, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "extrapolation", ["richardson", "ridders", "gauss-richardson", "gre"],
)
@pytest.mark.parametrize("levels", [None, 3])
def test_build_fisher_bias_finite_supports_extrapolations(extrapolation, levels):
    """Tests that finite-difference method supports various extrapolations."""
    theta0 = np.array([0.1, -0.2])
    cov = np.diag([2.0, 1.0, 0.5])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])
    delta_nu = np.array([1.0, 2.0, 3.0])

    lx = LikelihoodExpansion(function=linear_model, theta0=theta0, cov=cov)

    fd_kwargs = dict(
        stepsize=1e-2,
        num_points=5,
        extrapolation=extrapolation,
    )
    if levels is not None:
        fd_kwargs["levels"] = levels

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        method="finite",
        n_workers=2,
        **fd_kwargs,
    )

    diag = np.diag(cov)
    cinv_delta = delta_nu / diag
    expected_bias = A_LINEAR.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    assert bias_vec.shape == expected_bias.shape
    assert delta_theta.shape == expected_delta_theta.shape
    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(delta_theta, expected_delta_theta, rtol=1e-6, atol=1e-9)
