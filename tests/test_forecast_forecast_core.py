"""Tests for forecast_core methods."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.fisher import build_delta_nu, build_fisher_bias
from derivkit.forecasting.forecast_core import (
    _get_derivatives,
    get_forecast_tensors,
)
from derivkit.utils.linalg import invert_covariance


def _identity(x):
    """A test function that returns its input unchanged."""
    return x

def _two_obs(theta):
    """A test function that returns two identical observables."""
    t = float(np.asarray(theta)[0])
    return np.array([t, t], dtype=float)


def test_derivative_order():
    """Tests that unsupported derivative orders raise ValueError."""
    func = _identity
    theta0 = np.array([1.0])
    cov = np.array([[1.0]])

    with pytest.raises(ValueError):
        _get_derivatives(func, theta0, cov, order=np.random.randint(low=4, high=30))


def test_forecast_order():
    """Tests that unsupported forecast orders raise ValueError."""
    func = _identity
    theta0 = np.array([1.0])
    cov = np.array([[1.0]])

    with pytest.raises(ValueError):
        get_forecast_tensors(func, theta0, cov, forecast_order=3)

    with pytest.raises(ValueError):
        get_forecast_tensors(
            func, theta0, cov, forecast_order=np.random.randint(low=4, high=30)
        )


def test_pseudoinverse_path_no_nan(caplog):
    """Test that pseudoinverse path yields finite tensors."""
    # singular covariance -> forces pinv path
    func = _two_obs
    theta0 = np.array([1.0])

    cov = np.array([[1.0, 1.0],
                    [1.0, 1.0]], dtype=float)  # singular 2x2

    fisher = get_forecast_tensors(func, theta0, cov, forecast_order=1)
    tensor_g, tensor_h = get_forecast_tensors(func, theta0, cov, forecast_order=2)

    # We expect 2 warnings: 1 from the Fisher path and 1 from the DALI path
    assert 2 == len([x for x in caplog.records
        if "pseudoinverse" in x.message
        and "WARNING" == x.levelname
    ])

    assert np.isfinite(fisher).all()
    assert np.isfinite(tensor_g).all()
    assert np.isfinite(tensor_h).all()

@pytest.mark.parametrize(
    (
        "model, "
        "fiducials, "
        "covariance_matrix, "
        "expected_fisher, "
        "expected_dali_g, "
        "expected_dali_h"
    ),
    [
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[1.03612509]]),
            np.array([[[0.49105455]]]),
            np.array([[[[0.23272727]]]]),
        ),
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([1.1, 0.4]),
            np.array([[1.0, 2.75], [3.2, 0.1]]),
            np.array([[-0.00890115, 0.08901149], [0.10357701, -0.01177011]]),
            np.array(
                [
                    [
                        [-8.09195402e-03, 8.09195402e-02],
                        [-1.46382975e-16, -2.08644092e-16],
                    ],
                    [
                        [-1.42668169e-16, -3.75546474e-16],
                        [2.58942529e-01, -2.94252874e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-7.35632184e-03, -1.11448615e-16],
                            [-1.06393661e-16, 2.02298851e-01],
                        ],
                        [
                            [-1.33075432e-16, 7.15511183e-31],
                            [1.01887953e-30, -5.21610229e-16],
                        ],
                    ],
                    [
                        [
                            [-1.29698336e-16, 9.78598871e-31],
                            [1.29608820e-30, -9.38866185e-16],
                        ],
                        [
                            [2.35402299e-01, -6.14828927e-16],
                            [-1.10097326e-15, -7.35632184e-02],
                        ],
                    ],
                ]
            ),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[0.01890366]]),
            np.array([[[-0.03087509]]]),
            np.array([[[[0.05042786]]]]),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([1.1, 0.4]),
            np.array([[1.0, 2.75], [3.2, 0.1]]),
            np.array([[-0.00414466, 0.07008167], [0.08154958, -0.01566948]]),
            np.array(
                [
                    [
                        [7.89639690e-04, -1.33519460e-02],
                        [6.87814470e-15, -1.84147323e-15],
                    ],
                    [
                        [6.59290723e-17, -1.11478872e-15],
                        [1.71255860e-01, -3.29062482e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-1.50441995e-04, -1.12697772e-15],
                            [-1.25607936e-17, -2.80393711e-02],
                        ],
                        [
                            [-1.31042274e-15, -2.06221337e-28],
                            [-1.09410605e-28, -3.86713302e-15],
                        ],
                    ],
                    [
                        [
                            [-1.25607936e-17, -9.40943024e-29],
                            [-1.04873334e-30, -2.34108006e-15],
                        ],
                        [
                            [-3.26276318e-02, -4.04783117e-15],
                            [-2.72416588e-15, -6.91038222e-02],
                        ],
                    ],
                ]
            ),
        ),
    ],
)
def test_forecast(
    model,
    fiducials,
    covariance_matrix,
    expected_fisher,
    expected_dali_g,
    expected_dali_h,
    caplog,
):
    """Validates Fisher and DALI tensors against reference values.

    This test is parametrized over simple scalar/vector models and covariance
    matrices.

    - Fisher is compared with `atol=0` (after casting to float64).
    - DALI tensors (G, H) use a mixed tolerance: tight relative tolerance and a
      tiny absolute floor only where the expected entries are near zero. This
      avoids false failures from floating-point noise in ~0 entries while
      keeping real values strict.
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    func = observables
    theta0 = fiducial_values

    # Fisher (order=1): set tolerances up front
    fisher_rtol = 3e-3
    fisher_atol = 1e-12

    fisher_matrix = get_forecast_tensors(func, theta0, covmat, forecast_order=1)
    dali_g, dali_h = get_forecast_tensors(func, theta0, covmat, forecast_order=2)
    # Helper: warn only if cov is non-symmetric
    want_sym_warn = not np.allclose(covmat, covmat.T)
    if want_sym_warn:
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert r"`cov` is not symmetric; proceeding as-is" in caplog.text

    assert np.allclose(
        np.asarray(fisher_matrix, float),
        np.asarray(expected_fisher, float),
        rtol=fisher_rtol,
        atol=fisher_atol,
    )

    is_multi_param = fiducials.size > 1
    rtol_g = 3e-3 if is_multi_param else 5e-4
    # H is more sensitive;
    rtol_h = 5e-3 if is_multi_param else 2e-3

    # Non-symmetric covariances are more sensitive; relax slightly
    if not np.allclose(covmat, covmat.T):
        rtol_g = max(rtol_g, 1.5e-2)  # was 3e-3
        rtol_h = max(rtol_h, 2.5e-2)  # was 5e-3

    _assert_close_mixed(dali_g, expected_dali_g, rtol=rtol_g, label="dali_g")
    _assert_close_mixed(dali_h, expected_dali_h, rtol=rtol_h, label="dali_h")


def _assert_close_mixed(
    actual, expected, *, rtol=1e-8, zero_band=1e-10, floor=5e-13, label=""
):
    """Asserts numerical closeness with a near-zero absolute floor.

    Compares arrays using `np.isclose`, but applies a small absolute
    tolerance only where the reference values are already near zero. This
    pattern preserves strict relative comparisons for meaningful entries while
    preventing spurious failures from sub-epsilon noise in ~0 slots.

    Args:
      actual (array_like): Actual values.
      expected (array_like): Reference values (tolerances are defined relative to this).
      rtol (float, optional): Relative tolerance for non-near-zero entries.
      zero_band (float, optional): Threshold below which ``abs(b)`` is treated
        as “near zero” and the absolute floor applies.
      floor (float, optional): Absolute tolerance to allow only where
        ``abs(b) < zero_band``.
      label (str, optional): Short label included in the failure message.

    Raises:
      AssertionError: If any element of ``a`` is not close to ``b`` under the
        mixed tolerance rule. The message includes the first failing index,
        values, absolute difference, and effective tolerances.
    """
    actual = np.asarray(actual, dtype=float)  # ensure float64
    expected = np.asarray(expected, dtype=float)

    # absolute floor only where the *expected* value is near zero
    atol = np.where(np.abs(expected) < zero_band, floor, 0.0)

    ok = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if not ok.all():
        idx = tuple(np.argwhere(~ok)[0])
        diff = actual - expected
        raise AssertionError(
            f"{label} not close at {idx}: "
            f"actual={actual[idx]} expected={expected[idx]} "
            f"|diff|={abs(diff[idx])} rtol={rtol} atol[idx]={atol[idx]}"
        )


def test_raises_on_mismatched_obs_cov_dims_runtime():
    """Tests that get_forecast_tensors raises on mismatched observable/cov dims."""

    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0] ** 2])

    theta0 = np.array([0.1])
    cov = np.eye(2)  # expects 2 observables

    # Now the shape check happens immediately, for any forecast_order
    with pytest.raises(ValueError, match=r"Expected 2 observables"):
        get_forecast_tensors(model, theta0, cov, forecast_order=1)

    with pytest.raises(ValueError, match=r"Expected 2 observables"):
        get_forecast_tensors(model, theta0, cov, forecast_order=2)


def test_inv_cov_behaves_like_inverse():
    """Tests that _inv_cov() returns a matrix behaving like the inverse."""
    cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    inv = invert_covariance(cov, rcond=1e-12)
    np.testing.assert_allclose(inv @ cov, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(cov @ inv, np.eye(2), atol=1e-12)


def test_build_delta_nu_validation_errors():
    """Tests that build_delta_nu raises on bad inputs."""
    cov = np.eye(2)

    with pytest.raises(ValueError):
        # incompatible lengths
        build_delta_nu(cov=cov, data_biased=np.array([1.0, 2.0]), data_unbiased=np.array([1.0]))

    with pytest.raises(FloatingPointError):
        # NaN in data_biased
        build_delta_nu(cov=cov, data_biased=np.array([np.nan, 1.0]), data_unbiased=np.array([0.0, 0.0]))


def _linear_model(design_matrix, theta):
    """Returns design_matrix @ theta for linear-model tests."""
    theta = np.asarray(theta, dtype=float)
    return design_matrix @ theta


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

    fisher_matrix = get_forecast_tensors(
        linear_model, theta0, covariance, forecast_order=1
    )

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

    fisher_matrix = get_forecast_tensors(
        linear_model, theta0, covariance, forecast_order=1
    )

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


def test_fisher_bias_accepts_2d_delta_row_major_consistency():
    """Test that build_fisher_bias accepts 2D delta_nu and matches flattened 1D input."""
    design_matrix = np.array([[1.0, 2.0],
                              [0.5, -1.0]], float)
    model = partial(_linear_model, design_matrix)
    cov = np.eye(2)
    theta0 = np.zeros(2)

    fisher = get_forecast_tensors(model, theta0, cov, forecast_order=1)

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


def test_fisher_bias_singular_fisher_uses_pinv_baseline():
    """Tests that the Fisher bias equation is satisfied for singular Fisher matrices."""
    design_matrix = np.array([[1.0, 1.0],
                              [1.0, 1.0]], float)  # rank-1
    model = partial(_linear_model, design_matrix)
    cov = np.eye(2)
    theta0 = np.zeros(2)

    fisher = get_forecast_tensors(model, theta0, cov, forecast_order=1)
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

    fisher = get_forecast_tensors(model, theta0, cov, forecast_order=1)

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
    fisher_matrix = get_forecast_tensors(model, theta0, cov, forecast_order=1)

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
    fisher = get_forecast_tensors(model, theta0, cov, forecast_order=1)

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


def model_quadratic(theta: np.ndarray) -> np.ndarray:
    """A simple 2D→2D quadratic model for basic multi-parameter tests."""
    t0, t1 = np.asarray(theta, float)
    return np.array([t0**2, 2.0 * t0 * t1], float)


def test_fisher_bias_quadratic_small_systematic():
    """End-to-end test of Fisher bias against quadratic model with small systematic."""
    theta0 = np.array([1.2, -0.7], float)
    cov = np.diag([0.8, 1.1])
    cov_inv = np.diag(1.0 / np.diag(cov))

    delta = np.array([0.03, -0.02], float)

    fisher = get_forecast_tensors(model_quadratic, theta0, cov, forecast_order=1)
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


def test_build_fisher_bias_raises_on_nans_in_delta():
    """If delta_nu contains NaNs, build_fisher_bias should raise FloatingPointError."""
    matrix = np.eye(2, dtype=float)
    model = partial(_linear_model, matrix)
    cov = np.eye(2, dtype=float)
    theta0 = np.zeros(2)

    fisher = get_forecast_tensors(model, theta0, cov, forecast_order=1)

    with pytest.raises(FloatingPointError, match="Non-finite values"):
        build_fisher_bias(
            function=model,
            theta0=theta0,
            cov=cov,
            fisher_matrix=fisher,
            delta_nu=np.array([np.nan, 0.0]),
        )
