"""Tests for Fisher-bias estimation in forecast_kit/fisher.py."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.fisher import (
    build_delta_nu,
    build_fisher_bias,
    build_fisher_matrix,
)
from derivkit.utils.linalg import invert_covariance

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

def _linear_model(design_matrix, theta):
    """Returns design_matrix @ theta for linear-model tests."""
    theta = np.asarray(theta, dtype=float)
    return design_matrix @ theta

def model_quadratic(theta: np.ndarray) -> np.ndarray:
    """A simple 2D→2D quadratic model for basic multi-parameter tests."""
    t0, t1 = np.asarray(theta, float)
    return np.array([t0**2, 2.0 * t0 * t1], float)


def model_cubic(theta: np.ndarray) -> np.ndarray:
    """Returns a linear combination of cubes."""
    return np.asarray([np.sum(np.asarray(theta)**4)])


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

    data_biased = np.array([1.0, 2.0, 3.0, 4.0])
    data_unbiased = np.array([0.5, 1.0, 1.5, 2.0])

    delta = build_delta_nu(
        cov=cov,
        data_biased=data_biased,
        data_unbiased=data_unbiased,
    )

    expected = data_biased - data_unbiased
    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected)


def test_build_delta_nu_2d_flattens_row_major():
    """Tests that build_delta_nu works for 2D inputs, flattening in row-major order."""
    cov = np.eye(4)

    data_biased = np.array([[1.0, 2.0], [3.0, 4.0]])
    data_unbiased = np.array([[0.5, 1.5], [2.0, 2.5]])

    delta = build_delta_nu(
        cov=cov,
        data_biased=data_biased,
        data_unbiased=data_unbiased,
    )

    expected_2d = data_biased - data_unbiased
    expected_flat = expected_2d.ravel(order="C")

    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected_flat)


@pytest.mark.parametrize(
    "data_biased, data_unbiased, cov, expected_exception",
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
def test_build_delta_nu_exceptions(data_biased, data_unbiased, cov, expected_exception):
    """Tests that invalid inputs to build_delta_nu raise appropriate errors."""
    with pytest.raises(expected_exception):
        build_delta_nu(
            cov=cov,
            data_biased=data_biased,
            data_unbiased=data_unbiased,
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


def test_fisher_bias_matches_lstsq_identity_cov():
    """Tests that Fisher bias matches ordinary least squares when covariance is identity."""
    design_matrix = np.array(
        [[1.0, 2.0],
         [0.5, -1.0]],
        dtype=float,
    )

    linear_model = partial(_linear_model, design_matrix)

    covariance = np.eye(2)
    theta0 = np.zeros(2)

    fisher_matrix = build_fisher_matrix(linear_model, theta0, covariance)

    delta_nu = np.array([1.0, -0.5], dtype=float)
    bias_vec, delta_theta = build_fisher_bias(
        function=linear_model,
        theta0=theta0,
        cov=covariance,
        fisher_matrix=fisher_matrix,
        delta_nu=delta_nu,
        n_workers=1,
    )

    # Reference values for identity covariance: ordinary least squares
    expected_bias = design_matrix.T @ delta_nu
    theta_lstsq, *_ = np.linalg.lstsq(design_matrix, delta_nu, rcond=None)

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fisher_matrix @ delta_theta, bias_vec, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(delta_theta, theta_lstsq, rtol=1e-10, atol=1e-12)



def test_fisher_bias_matches_gls_weighted_cov():
    """Fisher bias should match generalized least squares when covariance is non-identity."""
    design_matrix = np.array(
        [[1.0, 2.0, 0.0],
         [0.0, 1.0, 1.0]],
        dtype=float,
    )  # n_observables = 2, n_parameters = 3

    # Bind design_matrix so the model signature is model(theta) -> y
    linear_model = partial(_linear_model, design_matrix)

    covariance = np.array(
        [[2.0, 0.3],
         [0.3, 1.0]],
        dtype=float,
    )
    inv_covariance = np.linalg.inv(covariance)
    theta0 = np.zeros(3)

    fisher_matrix = build_fisher_matrix(linear_model, theta0, covariance)

    delta_nu = np.array([0.7, -1.2], dtype=float)
    bias_vec, delta_theta = build_fisher_bias(
        function=linear_model,
        theta0=theta0,
        cov=covariance,
        fisher_matrix=fisher_matrix,
        delta_nu=delta_nu,
        n_workers=1,
    )

    expected_bias = design_matrix.T @ (inv_covariance @ delta_nu)
    expected_delta_theta = np.linalg.pinv(fisher_matrix) @ expected_bias

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(delta_theta, expected_delta_theta, rtol=1e-10, atol=1e-12)


def test_fisher_bias_singular_fisher_uses_pinv_baseline():
    """Tests that the Fisher bias equation is satisfied for singular Fisher matrices."""
    design_matrix = np.array([[1.0, 1.0],
                              [1.0, 1.0]], float)  # rank-1
    model = partial(_linear_model, design_matrix)
    cov = np.eye(2)
    theta0 = np.zeros(2)

    fisher = build_fisher_matrix(model, theta0, cov)
    delta = np.array([1.0, -0.5], float)

    bias_vec, delta_theta = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta,
    )

    expected_bias = design_matrix.T @ delta

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fisher @ delta_theta, expected_bias, rtol=1e-10, atol=1e-12)


def test_fisher_bias_singular_covariance_matches_pinv_baseline(caplog):
    """Tests that build_fisher_bias with singular covariance matches pinv baseline."""
    design_matrix = np.array([[1.0, 0.0],
                              [1.0, 0.0]], float)
    model = partial(_linear_model, design_matrix)

    cov = np.array([[1.0, 1.0],
                    [1.0, 1.0]], float)  # rank-1
    theta0 = np.zeros(2)

    fisher = build_fisher_matrix(model, theta0, cov)

    delta = np.array([2.0, -1.0], float)

    bias_vec, dtheta = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta,
    )

    assert 1 == len([x for x in caplog.records
        if "covariance solve" in x.message
        and "WARNING" == x.levelname
    ])

    c_pinv = np.linalg.pinv(cov)
    expected_bias = design_matrix.T @ (c_pinv @ delta)
    expected_dtheta = np.linalg.pinv(fisher) @ expected_bias

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-8, atol=1e-10)


def test_fisher_bias_quadratic_small_systematic():
    """End-to-end test of Fisher bias against quadratic model with small systematic."""
    theta0 = np.array([1.2, -0.7], float)
    cov = np.diag([0.8, 1.1])
    cov_inv = np.diag(1.0 / np.diag(cov))

    delta = np.array([0.03, -0.02], float)

    fisher = build_fisher_matrix(model_quadratic, theta0, cov)
    bias, dtheta = build_fisher_bias(
        function=model_quadratic,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta,
    )

    jac = np.array([[2.0 * theta0[0], 0.0],
                    [2.0 * theta0[1], 2.0 * theta0[0]]], float)
    expected_fisher = jac.T @ cov_inv @ jac
    expected_bias = jac.T @ (cov_inv @ delta)
    expected_dtheta = np.linalg.pinv(expected_fisher) @ expected_bias

    np.testing.assert_allclose(fisher, expected_fisher, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(bias, expected_bias, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=0.0, atol=1e-9)



def test_fisher_bias_linear_ground_truth_end_to_end():
    """End-to-end test of Fisher bias against linear-model analytic solution."""
    # 4 observables, 3 parameters
    matrix = np.array([[1.0, 2.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [2.0, 0.0, 1.0],
                  [-1.0, 0.5, 0.5]], float)
    model = partial(_linear_model, matrix)

    cov = np.diag([0.5, 1.2, 2.0, 0.8])
    cov_inv = np.diag(1.0 / np.diag(cov))

    theta0 = np.zeros(3)

    # Two data vectors: "with systematics" = y + s, "without" = y
    y = model(theta0)
    s = np.array([0.3, -0.1, 0.05, 0.2], float)
    d_with, d_without = y + s, y

    delta = build_delta_nu(cov=cov, data_biased=d_with, data_unbiased=d_without)
    fisher_matrix = build_fisher_matrix(model, theta0, cov)

    bias, dtheta = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher_matrix,
        delta_nu=delta,
    )

    expected_bias = matrix.T @ (cov_inv @ s)
    expected_fisher = matrix.T @ cov_inv @ matrix
    expected_dtheta = np.linalg.pinv(expected_fisher) @ expected_bias

    np.testing.assert_allclose(bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-11, atol=1e-12)


def test_fisher_bias_linear_full_cov_gls_formula():
    """End-to-end test of Fisher bias against linear-model analytic solution with full cov."""
    # 3 observables, 2 parameters
    matrix = np.array([[1.0,  0.0],
                  [2.0, -1.0],
                  [0.5, 1.0]], float)
    model = partial(_linear_model, matrix)

    cov = np.array([[ 1.0,  0.2, -0.1],
                  [ 0.2,  2.0,  0.3],
                  [-0.1,  0.3,  1.5]], float)
    cov_inv = np.linalg.inv(cov)

    theta0 = np.zeros(2)
    fisher = build_fisher_matrix(model, theta0, cov)

    s = np.array([0.4, -0.2, 0.1], float)  # “with” – “without”
    bias, dtheta = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=s,
    )

    expected_bias = matrix.T @ (cov_inv @ s)
    expected_fisher = matrix.T @ cov_inv @ matrix
    expected_dtheta = np.linalg.pinv(expected_fisher) @ expected_bias

    np.testing.assert_allclose(bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-11, atol=1e-12)



def test_fisher_bias_raises_on_wrong_shapes():
    """Test that build_fisher_bias raises on mismatched shapes."""
    model = partial(_linear_model, np.eye(2))
    theta0 = np.zeros(2)
    cov = np.eye(2)

    # Wrong Fisher shape (3x3 vs 2 params); should raise an exception.
    fisher_bad = np.eye(3)
    with pytest.raises(ValueError, match=r"fisher_matrix must be square;|shape.*\(3, 3\).*"):
        build_fisher_bias(
            function=model,
            theta0=theta0,
            cov=cov,
            fisher_matrix=fisher_bad,
            delta_nu=np.zeros(2),
        )

    # Fisher shape OK (2x2), but delta_nu length wrong (3 vs n_obs=2)
    fisher_ok = np.eye(2)
    with pytest.raises(ValueError, match=r"delta_nu must have length n=2"):
        build_fisher_bias(
            function=model,
            theta0=theta0,
            cov=cov,
            fisher_matrix=fisher_ok,
            delta_nu=np.zeros(3),
        )


def test_fisher_bias_accepts_2d_delta_row_major_consistency():
    """Test that build_fisher_bias accepts 2D delta_nu and matches flattened 1D input."""
    design_matrix = np.array([[1.0, 2.0],
                              [0.5, -1.0]], float)
    model = partial(_linear_model, design_matrix)
    cov = np.eye(2)
    theta0 = np.zeros(2)

    fisher = build_fisher_matrix(model, theta0, cov)

    delta_2d = np.array([[1.0, -0.5]], float)
    bias_a, dtheta_a = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta_2d,
    )

    delta_1d = build_delta_nu(
        cov=cov,
        data_biased=delta_2d,
        data_unbiased=np.zeros_like(delta_2d),
    ).ravel(order="C")
    bias_b, dtheta_b = build_fisher_bias(
        function=model,
        theta0=theta0,
        cov=cov,
        fisher_matrix=fisher,
        delta_nu=delta_1d,
    )

    np.testing.assert_allclose(bias_a, bias_b)
    np.testing.assert_allclose(dtheta_a, dtheta_b)


def test_build_fisher_bias_raises_on_nans_in_delta():
    """If delta_nu contains NaNs, build_fisher_bias should raise FloatingPointError."""
    matrix = np.eye(2, dtype=float)
    model = partial(_linear_model, matrix)
    cov = np.eye(2, dtype=float)
    theta0 = np.zeros(2)

    fisher = build_fisher_matrix(model, theta0, cov)

    with pytest.raises(FloatingPointError, match="Non-finite values"):
        build_fisher_bias(
            function=model,
            theta0=theta0,
            cov=cov,
            fisher_matrix=fisher,
            delta_nu=np.array([np.nan, 0.0]),
        )


def test_build_delta_nu_validation_errors():
    """Tests that build_delta_nu raises on bad inputs."""
    cov = np.eye(2)

    with pytest.raises(ValueError):
        # incompatible lengths
        build_delta_nu(cov=cov, data_biased=np.array([1.0, 2.0]), data_unbiased=np.array([1.0]))

    with pytest.raises(FloatingPointError):
        # NaN in data_biased
        build_delta_nu(cov=cov, data_biased=np.array([np.nan, 1.0]), data_unbiased=np.array([0.0, 0.0]))


def test_build_delta_nu_1d_and_2d_row_major():
    """Tests that build_delta_nu returns correct shapes and values."""
    # 1D case
    cov_2 = np.eye(2)
    data_biased = np.array([3.0, -1.0], dtype=float)
    data_unbiased = np.array([2.5, -2.0], dtype=float)

    delta_1d = build_delta_nu(cov=cov_2, data_biased=data_biased, data_unbiased=data_unbiased)
    np.testing.assert_allclose(delta_1d, np.array([0.5, 1.0], dtype=float))
    assert delta_1d.shape == (cov_2.shape[0],)

    # 2D case
    cov_6 = np.eye(6)
    a2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b2 = np.array([[0, 1, 1], [1, 1, 1]], dtype=float)

    delta_2d = build_delta_nu(cov=cov_6, data_biased=a2, data_unbiased=b2)
    np.testing.assert_allclose(delta_2d, (a2 - b2).ravel(order="C"))
    assert delta_2d.ndim == 1
    assert delta_2d.size == 6

def test_inv_cov_behaves_like_inverse():
    """Tests that _inv_cov() returns a matrix behaving like the inverse."""
    cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    inv = invert_covariance(cov, rcond=1e-12)
    np.testing.assert_allclose(inv @ cov, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(cov @ inv, np.eye(2), atol=1e-12)
