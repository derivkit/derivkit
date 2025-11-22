"""Tests for Fisher-bias estimation in LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)
    return np.zeros(3, dtype=float)


def two_obs_model(theta):
    """Model that always returns a 2-element observable vector."""
    _ = np.atleast_1d(theta)
    return np.zeros(2, dtype=float)


J_GLOBAL = None
J_CALL_INFO: dict = {}
SOLVE_CALLS: list[dict] = []


def fake_build_jacobian(function, theta0, *args, **kwargs):
    """Mimics build_jacobian by returning a global J_GLOBAL and recording call info."""
    global J_CALL_INFO
    J_CALL_INFO = {
        "function": function,
        "theta0": np.asarray(theta0, dtype=float),
        "args": args,
        "kwargs": kwargs,
    }
    return J_GLOBAL


def fake_solve_or_pinv(a, b, rcond=1e-12, assume_symmetric=True, warn_context=None):
    """Mimics solve_or_pinv by recording call info and performing genuine solve/pinv."""
    global SOLVE_CALLS
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    SOLVE_CALLS.append(
        {
            "a": a_arr,
            "b": b_arr,
            "rcond": rcond,
            "assume_symmetric": assume_symmetric,
            "warn_context": warn_context,
        }
    )

    # Use a genuine solve so we can check numerical values
    try:
        x = np.linalg.solve(a_arr, b_arr)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(a_arr, rcond=rcond) @ b_arr
    return x


def test_build_fisher_bias_diagonal_covariance(monkeypatch):
    """Tests that diagonal covariance uses optimized path without solve_or_pinv for cov."""
    theta0 = np.array([0.1, -0.2])
    # 3 observables, diagonal positive covariance
    cov = np.diag([2.0, 1.0, 0.5])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])

    global J_GLOBAL, J_CALL_INFO, SOLVE_CALLS
    J_GLOBAL = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )  # shape (n_obs=3, n_params=2)
    J_CALL_INFO = {}
    SOLVE_CALLS = []

    delta_nu = np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.solve_or_pinv",
        fake_solve_or_pinv,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        n_workers=3,
        method="finite",
        step_size=1e-2,
    )

    # Expected computations: diagonal cov branch => cinv_delta = delta_nu / diag
    diag = np.diag(cov)
    cinv_delta = delta_nu / diag
    expected_bias = J_GLOBAL.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    # build_jacobian delegation checks
    assert J_CALL_INFO["function"] is three_obs_model
    np.testing.assert_allclose(J_CALL_INFO["theta0"], theta0)
    assert J_CALL_INFO["kwargs"]["method"] == "finite"
    assert J_CALL_INFO["kwargs"]["n_workers"] == 3
    assert J_CALL_INFO["kwargs"]["step_size"] == 1e-2

    # Diagonal positive cov => no solve_or_pinv for C, only for Fisher solve
    assert len(SOLVE_CALLS) == 1
    np.testing.assert_allclose(SOLVE_CALLS[0]["a"], fisher)
    np.testing.assert_allclose(SOLVE_CALLS[0]["b"], expected_bias)
    assert SOLVE_CALLS[0]["warn_context"] == "Fisher solve"


def test_build_fisher_bias_uses_solver_for_nondiagonal_cov(monkeypatch):
    """Tests that non-diagonal covariance uses solver path."""
    theta0 = np.array([0.0, 0.5])
    cov = np.array(
        [
            [2.0, 0.3, 0.0],
            [0.3, 1.5, 0.1],
            [0.0, 0.1, 0.8],
        ]
    )
    fisher = np.array([[1.0, 0.2], [0.2, 1.3]])

    global J_GLOBAL, J_CALL_INFO, SOLVE_CALLS
    J_GLOBAL = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ]
    )
    J_CALL_INFO = {}
    SOLVE_CALLS = []

    delta_nu = np.array([0.5, -1.0, 2.0])

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.solve_or_pinv",
        fake_solve_or_pinv,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        n_workers=1,
        method="adaptive",
    )

    # Expected using the same linear algebra path as the implementation
    cinv_delta = np.linalg.solve(cov, delta_nu)
    expected_bias = J_GLOBAL.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    # We should have two solver calls: one for cov, one for Fisher
    assert len(SOLVE_CALLS) == 2
    assert {c["warn_context"] for c in SOLVE_CALLS} == {
        "covariance solve",
        "Fisher solve",
    }


def test_build_fisher_bias_rejects_non_square_fisher():
    """Tests that non-square Fisher matrix raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(2)
    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = np.ones((2, 3))
    delta_nu = np.zeros(2)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_wrong_jacobian_shape(monkeypatch):
    """Tests that wrong Jacobian shape raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(3)  # n_obs = 3, n_params = 2

    global J_GLOBAL, J_CALL_INFO
    # Wrong shape: (n_obs+1, n_params)
    J_GLOBAL = np.ones((4, 2))
    J_CALL_INFO = {}

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.zeros(3)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_cov_shape_mismatch(monkeypatch):
    """Tests that covariance shape mismatch raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(2)  # initial n_obs=2

    global J_GLOBAL, J_CALL_INFO
    J_GLOBAL = np.zeros((2, 1))  # J has 2 rows, 1 param
    J_CALL_INFO = {}

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)
    # Manually break consistency after init
    lx.cov = np.eye(3)

    fisher = np.eye(1)
    delta_nu = np.zeros(2)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_fisher_shape_mismatch(monkeypatch):
    """Tests that Fisher shape mismatch raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(3)  # n_obs = 3

    global J_GLOBAL, J_CALL_INFO
    # J: 3 observables, 2 params
    J_GLOBAL = np.zeros((3, 2))
    J_CALL_INFO = {}

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(3)  # should be (2,2), so this mismatches J_GLOBAL
    delta_nu = np.zeros(3)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_wrong_delta_nu_length(monkeypatch):
    """Tests that delta_nu with wrong length raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)  # n_obs = 3

    global J_GLOBAL
    J_GLOBAL = np.zeros((3, 2))
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.zeros(2)  # wrong length

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_nonfinite_delta_nu(monkeypatch):
    """Tests that non-finite delta_nu entries raise FloatingPointError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    global J_GLOBAL
    J_GLOBAL = np.zeros((3, 2))
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.array([0.0, np.nan, 1.0])

    with pytest.raises(FloatingPointError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)

def test_build_fisher_bias_accepts_2d_delta_nu_and_flattens(monkeypatch):
    """Tests that 2D delta_nu is flattened correctly."""
    # One parameter, four observables
    theta0 = np.array([0.0])
    cov = np.eye(4)

    global J_GLOBAL
    J_GLOBAL = np.ones((4, 1))
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = np.array([[2.0]])
    delta_nu_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Should not raise, just smoke-test flattening path
    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu_2d,
    )
    assert bias_vec.shape == (1,)
    assert delta_theta.shape == (1,)


@pytest.mark.parametrize("method", ["adaptive", "finite", "local_polyfit"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_build_fisher_bias_forwards_derivative_kwargs(
    monkeypatch,
    method,
    extrapolation,
    stencil,
):
    """Tests that derivative kwargs are forwarded correctly."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    global J_GLOBAL, J_CALL_INFO, SOLVE_CALLS
    J_GLOBAL = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    J_CALL_INFO = {}
    SOLVE_CALLS = []

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        fake_build_jacobian,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.solve_or_pinv",
        fake_solve_or_pinv,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.array([1.0, 2.0, 3.0])

    if method == "local_polyfit":
        # polyfit path: no extrapolation/stencil, but degree/window/trim_fraction
        bias_vec, delta_theta = lx.build_fisher_bias(
            fisher_matrix=fisher,
            delta_nu=delta_nu,
            n_workers=5,
            method="local_polyfit",
            degree=5,
            window=4,
            trim_fraction=0.2,
        )
    else:
        # finite / adaptive path: use extrapolation + stencil
        bias_vec, delta_theta = lx.build_fisher_bias(
            fisher_matrix=fisher,
            delta_nu=delta_nu,
            n_workers=5,
            method=method,
            extrapolation=extrapolation,
            stencil=stencil,
        )

    # Numeric expectations (cov = I, so cinv_delta = delta_nu)
    cinv_delta = delta_nu
    expected_bias = J_GLOBAL.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    # Delegation checks
    kwargs = J_CALL_INFO["kwargs"]
    assert kwargs["n_workers"] == 5

    if method == "local_polyfit":
        assert kwargs["method"] == "local_polyfit"
        assert kwargs["degree"] == 5
        assert kwargs["window"] == 4
        assert kwargs["trim_fraction"] == 0.2
        # optional: make sure FD-only kwargs are *not* present
        assert "extrapolation" not in kwargs
        assert "stencil" not in kwargs
    else:
        assert kwargs["method"] == method
        assert kwargs["extrapolation"] == extrapolation
        assert kwargs["stencil"] == stencil


def test_build_delta_nu_1d_ok():
    """Tests that build_delta_nu works for 1D inputs."""
    theta0 = np.array([0.0])
    cov = np.eye(4)  # n_observables = 4

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    data_with = np.array([1.0, 2.0, 3.0, 4.0])
    data_without = np.array([0.5, 1.0, 1.5, 2.0])

    delta = lx.build_delta_nu(data_with=data_with, data_without=data_without)

    expected = data_with - data_without
    assert delta.shape == (4,)
    np.testing.assert_allclose(delta, expected)


def test_build_delta_nu_2d_flattens_row_major():
    """Tests that build_delta_nu works for 2D inputs and flattens in row-major order."""
    theta0 = np.array([0.0])
    cov = np.eye(4)  # n_observables = 4

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
    """Tests that inputs with wrong length vs n_observables raise ValueError."""
    theta0 = np.array([0.0])
    cov = np.eye(5)  # n_observables = 5

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    # 2x2 so length 4, but n_observables = 5
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))

    with pytest.raises(ValueError):
        lx.build_delta_nu(data_with=a, data_without=b)


def test_build_delta_nu_rejects_nonfinite_values():
    """Tests that non-finite entries in inputs raise FloatingPointError."""
    theta0 = np.array([0.0])
    cov = np.eye(3)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.array([0.0, np.nan, 1.0])
    b = np.zeros(3)

    with pytest.raises(FloatingPointError):
        lx.build_delta_nu(data_with=a, data_without=b)
