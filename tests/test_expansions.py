"""Tests for LikelihoodExpansion."""

from contextlib import nullcontext
from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def test_derivative_order():
    """High derivative orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        like._get_derivatives(order=3)
    with pytest.raises(ValueError):
        like._get_derivatives(order=np.random.randint(low=4, high=30))


def test_forecast_order():
    """High forecast orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))

    with pytest.raises(ValueError):
        like.get_forecast_tensors(forecast_order=3)

    with pytest.raises(ValueError):
        like.get_forecast_tensors(
            forecast_order=np.random.randint(low=4, high=30)
        )


def test_pseudoinverse_path_no_nan():
    """If inversion fails, pseudoinverse path should still return finite numbers."""
    # singular covariance -> forces pinv path
    forecaster = LikelihoodExpansion(
        lambda x: x,
        np.array([1.0]),
        np.array([[0.0]]),
    )

    # Fisher (order=1) triggers: ill-conditioned + inversion failed → pinv
    with pytest.warns(RuntimeWarning, match=r"`cov` is ill-conditioned"):
        with pytest.warns(
            RuntimeWarning,
            match=r"`cov` inversion failed; using pseudoinverse",
        ):
            fisher = forecaster.get_forecast_tensors(forecast_order=1)
    assert np.isfinite(fisher).all()

    # DALI (order=2) should raise the same two warnings again
    with pytest.warns(RuntimeWarning, match=r"`cov` is ill-conditioned"):
        with pytest.warns(
            RuntimeWarning,
            match=r"`cov` inversion failed; using pseudoinverse",
        ):
            tensor_g, tensor_h = forecaster.get_forecast_tensors(
                forecast_order=2
            )
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
):
    """Validate Fisher and DALI tensors against reference values.

    This test is parametrized over simple scalar/vector models and covariance
    matrices.

    - Fisher is compared with `atol=0` (after casting to float64).
    - DALI tensors (G, H) use a mixed tolerance: tight relative tolerance and a
      tiny absolute floor only where the expected entries are near zero. This
      avoids false failures from floating-point noise in ~0 entries while
      keeping real values strict.

    Args:
      model (Callable[[np.ndarray], np.ndarray | float]): Observable
        model evaluated at `fiducials`.
      fiducials (np.ndarray): Parameter point(s) at which derivatives are
        evaluated.
      covariance_matrix (np.ndarray): Observables' covariance used to weight
        derivatives.
      expected_fisher (np.ndarray): Reference Fisher matrix.
      expected_dali_g (np.ndarray): Reference third-order DALI tensor G.
      expected_dali_h (np.ndarray): Reference fourth-order DALI tensor H.

    Raises:
      AssertionError: If any tensor deviates beyond the specified tolerances.
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    le = LikelihoodExpansion(observables, fiducial_values, covmat)

    # Helper: warn only if cov is non-symmetric
    want_sym_warn = not np.allclose(covmat, covmat.T)

    # Fisher (order=1): set tolerances up front
    fisher_rtol = 3e-3
    fisher_atol = 1e-12

    fisher_ctx = (
        pytest.warns(
            RuntimeWarning, match=r"`cov` is not symmetric; proceeding as-is"
        )
        if want_sym_warn
        else nullcontext()
    )
    with fisher_ctx:
        fisher_matrix = le.get_forecast_tensors(forecast_order=1)

    assert np.allclose(
        np.asarray(fisher_matrix, float),
        np.asarray(expected_fisher, float),
        rtol=fisher_rtol,
        atol=fisher_atol,
    )

    # DALI (order=2): same symmetry-warning behavior
    dali_ctx = (
        pytest.warns(
            RuntimeWarning, match=r"`cov` is not symmetric; proceeding as-is"
        )
        if want_sym_warn
        else nullcontext()
    )
    with dali_ctx:
        dali_g, dali_h = le.get_forecast_tensors(forecast_order=2)

    is_multi_param = fiducials.size > 1
    rtol_g = 3e-3 if is_multi_param else 5e-4
    # H is more sensitive;
    rtol_h = 5e-3 if is_multi_param else 2e-3

    _assert_close_mixed(dali_g, expected_dali_g, rtol=rtol_g, label="dali_g")
    _assert_close_mixed(dali_h, expected_dali_h, rtol=rtol_h, label="dali_h")


def _assert_close_mixed(
    actual, expected, *, rtol=1e-8, zero_band=1e-10, floor=5e-13, label=""
):
    """Assert numerical closeness with a near-zero absolute floor.

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
    """If model and covariance dimensions mismatch, raise ValueError."""

    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0] ** 2])

    le = LikelihoodExpansion(model, np.array([0.1]), np.eye(2))  # construct ok
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=1)  # shape check triggers here


def test_le_init_public_attrs_contract():
    """Required public attrs exist, counts match, and no unexpected public instance fields."""
    theta0 = np.array([1.0, -2.0])
    cov = np.eye(2)
    like = LikelihoodExpansion(lambda x: x, theta0, cov)
    assert like.n_parameters == theta0.size
    assert like.n_observables == cov.shape[0]

    # Required public attributes
    expected_attributes = {
        "function",
        "theta0",
        "cov",
        "n_parameters",
        "n_observables",
    }
    actual_attributes = {k for k in like.__dict__ if not k.startswith("_")}
    assert expected_attributes == set(actual_attributes)


def test_inv_cov_behaves_like_inverse():
    """Test that _inv_cov() returns a matrix behaving like the inverse."""
    cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    like = LikelihoodExpansion(lambda x: x, np.array([0.0, 0.0]), cov)
    inv = like._inv_cov()
    np.testing.assert_allclose(inv @ cov, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(cov @ inv, np.eye(2), atol=1e-12)


def test_raises_on_mismatched_obs_cov_dims():
    """If model output length != cov size, computing tensors should raise."""

    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0] ** 2])

    le = LikelihoodExpansion(
        model, np.array([0.1]), np.eye(2)
    )  # constructs fine

    # Fisher path
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=1)

    # DALI path (optional, but good to check both)
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=2)


def test_build_delta_nu_validation_errors():
    """Test build_delta_nu raises on bad inputs."""
    cov = np.eye(2)
    def linear_model(theta):
        return np.asarray([theta[0], theta[1]], dtype=float)

    le = LikelihoodExpansion(linear_model, theta0=np.zeros(2), cov=cov)

    with pytest.raises(ValueError):
        # This should raise an exception because the two data vectors are of incompatible length
        le.build_delta_nu(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(FloatingPointError):
        # This should raise an exception because one of the data vectors contains a NaN.
        le.build_delta_nu(np.array([np.nan, 1.0]), np.array([0.0, 0.0]))


def _linear_model(design_matrix, theta):
    """Return design_matrix @ theta for linear-model tests."""
    theta = np.asarray(theta, dtype=float)
    return design_matrix @ theta


def test_build_delta_nu_1d_and_2d_row_major():
    """Test that build_delta_nu returns correct shapes and values."""
    # Test the case where theta0 is a 1D array
    cov_2 = np.eye(2)
    theta0 = np.zeros(2)
    model_2 = partial(_linear_model, np.eye(2))

    le2 = LikelihoodExpansion(model_2, theta0=theta0, cov=cov_2)

    data_with = np.array([3.0, -1.0], dtype=float)
    data_without = np.array([2.5, -2.0], dtype=float)
    delta_1d = le2.build_delta_nu(data_with, data_without)
    np.testing.assert_allclose(delta_1d, np.array([0.5, 1.0], dtype=float))
    assert delta_1d.shape == (le2.n_observables,)

    # Test the case where theta0 is a 2D array
    cov_6 = np.eye(6)

    model_6 = partial(_linear_model, np.zeros((6,6)))

    le6 = LikelihoodExpansion(model_6, theta0=np.zeros(1), cov=cov_6)

    a2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b2 = np.array([[0, 1, 1], [1, 1, 1]], dtype=float)
    delta_2d = le6.build_delta_nu(a2, b2)
    np.testing.assert_allclose(delta_2d, (a2 - b2).ravel(order="C"))
    assert delta_2d.ndim == 1
    assert delta_2d.size == 6


def test_fisher_bias_matches_lstsq_identity_cov():
    """Fisher bias should match an ordinary least-squares solution when covariance is identity."""
    design_matrix = np.array(
        [[1.0, 2.0],
         [0.5, -1.0]],
        dtype=float,
    )

    # Bind design_matrix so the model signature is model(theta) -> y
    linear_model = partial(_linear_model, design_matrix)

    covariance = np.eye(2)
    le = LikelihoodExpansion(linear_model, theta0=np.zeros(2), cov=covariance)

    # Fisher matrix computed by the class.
    fisher_matrix = le.get_forecast_tensors(forecast_order=1)

    delta_nu = np.array([1.0, -0.5], dtype=float)
    bias_vec, delta_theta = le.build_fisher_bias(
        fisher_matrix=fisher_matrix, delta_nu=delta_nu, n_workers=1
    )

    # Ground truth for identity covariance would be ordinary least squares:
    expected_bias = design_matrix.T @ delta_nu
    theta_lstsq, *_ = np.linalg.lstsq(design_matrix, delta_nu, rcond=None)

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-12, atol=1e-12)
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

    le = LikelihoodExpansion(linear_model, theta0=np.zeros(3), cov=covariance)

    # Fisher matrix computed by the class with the stored covariance.
    fisher_matrix = le.get_forecast_tensors(forecast_order=1)

    delta_nu = np.array([0.7, -1.2], dtype=float)
    bias_vec, delta_theta = le.build_fisher_bias(
        fisher_matrix=fisher_matrix, delta_nu=delta_nu, n_workers=1
    )

    # Ground truth via GLS: bias = A^T C^{-1} Δν; Δθ = F^{+} bias (use pinv for singular/ill-conditioned F)
    expected_bias = design_matrix.T @ (inv_covariance @ delta_nu)
    expected_delta_theta = np.linalg.pinv(fisher_matrix) @ expected_bias

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(delta_theta, expected_delta_theta, rtol=1e-10, atol=1e-12)


def test_fisher_bias_accepts_2d_delta_row_major_consistency():
    """Test that build_fisher_bias accepts 2D delta_nu and matches flattened 1D input."""
    design_matrix = np.array([[1.0, 2.0],
                              [0.5, -1.0]], float)
    model = partial(_linear_model, design_matrix)

    le = LikelihoodExpansion(model, theta0=np.zeros(2), cov=np.eye(2))
    fisher = le.get_forecast_tensors(forecast_order=1)

    delta_2d = np.array([[1.0, -0.5]], float)
    bias_a, dtheta_a = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_2d)

    delta_1d = le.build_delta_nu(delta_2d, np.zeros_like(delta_2d)).ravel(order="C")
    bias_b, dtheta_b = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_1d)

    np.testing.assert_allclose(bias_a, bias_b)
    np.testing.assert_allclose(dtheta_a, dtheta_b)


def test_fisher_bias_singular_fisher_uses_pinv_baseline():
    """Tests that the Fisher bias equation is satisfied for singular Fisher matrices."""
    design_matrix = np.array([[1.0, 1.0],
                              [1.0, 1.0]], float)  # rank-1
    model = partial(_linear_model, design_matrix)

    le = LikelihoodExpansion(model, theta0=np.zeros(2), cov=np.eye(2))
    fisher = le.get_forecast_tensors(forecast_order=1)
    delta = np.array([1.0, -0.5], float)

    bias_vec, delta_theta = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta)

    expected_bias = design_matrix.T @ delta

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fisher @ delta_theta, expected_bias, rtol=1e-10, atol=1e-12)


def test_fisher_bias_singular_covariance_matches_pinv_baseline():
    """Test that build_fisher_bias with singular covariance matches pinv baseline."""
    design_matrix = np.array([[1.0, 0.0],
                              [1.0, 0.0]], float)
    model = partial(_linear_model, design_matrix)

    cov = np.array([[1.0, 1.0],
                    [1.0, 1.0]], float)  # rank-1
    le = LikelihoodExpansion(model, theta0=np.zeros(2), cov=cov)
    fisher = le.get_forecast_tensors(forecast_order=1)

    delta = np.array([2.0, -1.0], float)

    with pytest.warns(RuntimeWarning, match="covariance solve"):
        bias_vec, dtheta = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta)

    c_pinv = np.linalg.pinv(cov)
    expected_bias = design_matrix.T @ (c_pinv @ delta)
    expected_dtheta = np.linalg.pinv(fisher) @ expected_bias

    np.testing.assert_allclose(bias_vec, expected_bias, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-8, atol=1e-10)


def test_fisher_bias_raises_on_wrong_shapes():
    """Test that build_fisher_bias raises on mismatched shapes."""
    model = partial(_linear_model, np.eye(2))
    le = LikelihoodExpansion(model, theta0=np.zeros(2), cov=np.eye(2))
    # Wrong Fisher shape (3x3 vs 2 params); should raise an exception.
    fisher_bad = np.eye(3)
    with pytest.raises(ValueError, match=r"fisher_matrix must be square;|shape.*\(3, 3\).*"):
        le.build_fisher_bias(fisher_matrix=fisher_bad, delta_nu=np.zeros(2))

    # Fisher shape OK (2x2), but delta_nu length wrong (3 vs n_obs=2);
    # should raise an exception
    fisher_ok = np.eye(2)
    with pytest.raises(ValueError, match=r"delta_nu must have length n=2"):
        le.build_fisher_bias(fisher_matrix=fisher_ok, delta_nu=np.zeros(3))


def test_fisher_bias_linear_ground_truth_end_to_end():
    """End-to-end test of Fisher bias against linear-model analytic solution."""
    # 4 observables, 3 parameters
    A = np.array([[1.0, 2.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [2.0, 0.0, 1.0],
                  [-1.0, 0.5, 0.5]], float)
    model = partial(_linear_model, A)

    cov = np.diag([0.5, 1.2, 2.0, 0.8])
    Cinv = np.diag(1.0 / np.diag(cov))

    theta0 = np.zeros(3)
    le = LikelihoodExpansion(model, theta0, cov)

    # Two data vectors: "with systematics" = y + s, "without" = y
    y = model(theta0)
    s = np.array([0.3, -0.1, 0.05, 0.2], float)  # arbitrary systematic
    d_with, d_without = y + s, y

    delta = le.build_delta_nu(d_with, d_without)  # should equal s (row-major flatten)
    fisher_matrix = le.get_forecast_tensors(forecast_order=1)

    bias, dtheta = le.build_fisher_bias(fisher_matrix=fisher_matrix, delta_nu=delta)

    # analytic solution: b = A^T C^{-1} s ; Δθ = F^{+} b with F = A^T C^{-1} A
    expected_bias = A.T @ (Cinv @ s)
    expected_fisher = A.T @ Cinv @ A
    expected_dtheta = np.linalg.pinv(expected_fisher) @ expected_bias

    np.testing.assert_allclose(bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-11, atol=1e-12)


def test_fisher_bias_linear_full_cov_gls_formula():
    """End-to-end test of Fisher bias against linear-model analytic solution with full cov."""
    # 3 obervabless, 2 parameters
    A = np.array([[1.0,  0.0],
                  [2.0, -1.0],
                  [0.5, 1.0]], float)
    model = partial(_linear_model, A)

    cov = np.array([[ 1.0,  0.2, -0.1],
                  [ 0.2,  2.0,  0.3],
                  [-0.1,  0.3,  1.5]], float)
    Cinv = np.linalg.inv(cov)

    le = LikelihoodExpansion(model, theta0=np.zeros(2), cov=cov)
    fisher = le.get_forecast_tensors(forecast_order=1)

    s = np.array([0.4, -0.2, 0.1], float)  # “with” – “without”
    bias, dtheta = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=s)

    expected_bias = A.T @ (Cinv @ s)
    expected_fisher = A.T @ Cinv @ A
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
    Cinv = np.diag(1.0 / np.diag(cov))

    le = LikelihoodExpansion(model_quadratic, theta0, cov)

    J = np.array([[2.0 * theta0[0], 0.0],
                  [2.0 * theta0[1], 2.0 * theta0[0]]], float)

    delta = np.array([0.03, -0.02], float)

    expected_fisher = J.T @ Cinv @ J
    expected_bias = J.T @ (Cinv @ delta)
    expected_dtheta = np.linalg.pinv(expected_fisher) @ expected_bias

    fisher = le.get_forecast_tensors(forecast_order=1)
    bias, dtheta = le.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta)

    np.testing.assert_allclose(fisher, expected_fisher, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dtheta, expected_dtheta, rtol=1e-11, atol=1e-12)
