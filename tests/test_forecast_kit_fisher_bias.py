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


@pytest.fixture
def fisher_bias_mocks(monkeypatch):
    """Provides fake build_jacobian + solve_or_pinv with per-test state."""

    class FisherBiasMocks:
        """Holds state and fakes for Fisher-bias tests."""
        def __init__(self):
            """Initialises with empty state."""
            self.jac = None
            self.jac_call_info: dict | None = None
            self.solve_calls: list[dict] = []

        def set_jacobian(self, jac):
            """Sets the Jacobian matrix to be returned by the fake."""
            self.jac = np.asarray(jac, dtype=float)
            self.jac_call_info = None
            self.solve_calls = []

        def fake_build_jacobian(self, function, theta0, *args, **kwargs):
            """Mimics build_jacobian by returning self.J and recording call info."""
            self.jac_call_info = {
                "function": function,
                "theta0": np.asarray(theta0, dtype=float),
                "args": args,
                "kwargs": kwargs,
            }
            return self.jac

        def fake_solve_or_pinv(
            self,
            a,
            b,
            rcond: float = 1e-12,
            assume_symmetric: bool = True,
            warn_context: str | None = None,
        ):
            """Mimics solve_or_pinv by recording call info and performing genuine solve/pinv."""
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)

            call = {
                "a": a_arr,
                "b": b_arr,
                "rcond": rcond,
                "assume_symmetric": assume_symmetric,
                "warn_context": warn_context,
            }
            self.solve_calls.append(call)

            try:
                x = np.linalg.solve(a_arr, b_arr)
            except np.linalg.LinAlgError:
                x = np.linalg.pinv(a_arr, rcond=rcond) @ b_arr
            return x

    mocks = FisherBiasMocks()

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.build_jacobian",
        mocks.fake_build_jacobian,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.solve_or_pinv",
        mocks.fake_solve_or_pinv,
        raising=True,
    )

    return mocks


def test_build_fisher_bias_diagonal_covariance(fisher_bias_mocks):
    """Tests that diagonal covariance uses the optimized path."""
    theta0 = np.array([0.1, -0.2])
    cov = np.diag([2.0, 1.0, 0.5])
    fisher = np.array([[2.0, 0.1], [0.1, 1.5]])

    fisher_bias_mocks.set_jacobian(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )

    delta_nu = np.array([1.0, 2.0, 3.0])

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        n_workers=3,
        method="finite",
        step_size=1e-2,
    )

    diag = np.diag(cov)
    cinv_delta = delta_nu / diag
    expected_bias = fisher_bias_mocks.jac.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    jac_info = fisher_bias_mocks.jac_call_info
    assert jac_info["function"] is three_obs_model
    np.testing.assert_allclose(jac_info["theta0"], theta0)
    assert jac_info["kwargs"]["method"] == "finite"
    assert jac_info["kwargs"]["n_workers"] == 3
    assert jac_info["kwargs"]["step_size"] == 1e-2

    solve_calls = fisher_bias_mocks.solve_calls
    assert len(solve_calls) == 1
    np.testing.assert_allclose(solve_calls[0]["a"], fisher)
    np.testing.assert_allclose(solve_calls[0]["b"], expected_bias)
    assert solve_calls[0]["warn_context"] == "Fisher solve"


def test_build_fisher_bias_uses_solver_for_nondiagonal_cov(fisher_bias_mocks):
    """Tests that non-diagonal covariance uses the general solver path."""
    theta0 = np.array([0.0, 0.5])
    cov = np.array(
        [
            [2.0, 0.3, 0.0],
            [0.3, 1.5, 0.1],
            [0.0, 0.1, 0.8],
        ]
    )
    fisher = np.array([[1.0, 0.2], [0.2, 1.3]])

    fisher_bias_mocks.set_jacobian(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
            ]
        )
    )

    delta_nu = np.array([0.5, -1.0, 2.0])

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    bias_vec, delta_theta = lx.build_fisher_bias(
        fisher_matrix=fisher,
        delta_nu=delta_nu,
        n_workers=1,
        method="adaptive",
    )

    cinv_delta = np.linalg.solve(cov, delta_nu)
    expected_bias = fisher_bias_mocks.jac.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    solve_calls = fisher_bias_mocks.solve_calls
    assert len(solve_calls) == 2
    assert {c["warn_context"] for c in solve_calls} == {
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


def test_build_fisher_bias_rejects_wrong_jacobian_shape(fisher_bias_mocks):
    """Tests that wrong Jacobian shape raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(3)

    fisher_bias_mocks.set_jacobian(np.ones((4, 2)))

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.zeros(3)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_cov_shape_mismatch(fisher_bias_mocks):
    """Tests that covariance shape mismatch raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(2)

    fisher_bias_mocks.set_jacobian(np.zeros((2, 1)))

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)
    lx.cov = np.eye(3)

    fisher = np.eye(1)
    delta_nu = np.zeros(2)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_fisher_shape_mismatch(fisher_bias_mocks):
    """Tests that Fisher shape mismatch raises ValueError."""
    theta0 = np.array([0.0, 1.0])
    cov = np.eye(3)

    fisher_bias_mocks.set_jacobian(np.zeros((3, 2)))

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(3)
    delta_nu = np.zeros(3)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_wrong_delta_nu_length(fisher_bias_mocks):
    """Tests that wrong-length delta_nu raises ValueError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    fisher_bias_mocks.set_jacobian(np.zeros((3, 2)))

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.zeros(2)

    with pytest.raises(ValueError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_rejects_nonfinite_delta_nu(fisher_bias_mocks):
    """Tests that non-finite delta_nu raises FloatingPointError."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    fisher_bias_mocks.set_jacobian(np.zeros((3, 2)))

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.array([0.0, np.nan, 1.0])

    with pytest.raises(FloatingPointError):
        lx.build_fisher_bias(fisher_matrix=fisher, delta_nu=delta_nu)


def test_build_fisher_bias_accepts_2d_delta_nu_and_flattens(fisher_bias_mocks):
    """Tests that 2D delta_nu is flattened correctly."""
    theta0 = np.array([0.0])
    cov = np.eye(4)

    fisher_bias_mocks.set_jacobian(np.ones((4, 1)))

    lx = LikelihoodExpansion(function=two_obs_model, theta0=theta0, cov=cov)

    fisher = np.array([[2.0]])
    delta_nu_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

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
    fisher_bias_mocks,
    method,
    extrapolation,
    stencil,
):
    """Tests that derivative kwargs are passed correctly."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    fisher_bias_mocks.set_jacobian(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    fisher = np.eye(2)
    delta_nu = np.array([1.0, 2.0, 3.0])

    if method == "local_polyfit":
        bias_vec, delta_theta = lx.build_fisher_bias(
            fisher_matrix=fisher,
            delta_nu=delta_nu,
            n_workers=5,
            method="local_polyfit",
            degree=5,
        )
    else:
        bias_vec, delta_theta = lx.build_fisher_bias(
            fisher_matrix=fisher,
            delta_nu=delta_nu,
            n_workers=5,
            method=method,
            extrapolation=extrapolation,
            stencil=stencil,
        )

    cinv_delta = delta_nu
    expected_bias = fisher_bias_mocks.jac.T @ cinv_delta
    expected_delta_theta = np.linalg.solve(fisher, expected_bias)

    np.testing.assert_allclose(bias_vec, expected_bias)
    np.testing.assert_allclose(delta_theta, expected_delta_theta)

    # Delegation checks
    jac_info = fisher_bias_mocks.jac_call_info
    kwargs = jac_info["kwargs"]
    assert kwargs["n_workers"] == 5

    if method == "local_polyfit":
        assert kwargs["method"] == "local_polyfit"
        assert kwargs["degree"] == 5
        assert "extrapolation" not in kwargs
        assert "stencil" not in kwargs
    else:
        assert kwargs["method"] == method
        assert kwargs["extrapolation"] == extrapolation
        assert kwargs["stencil"] == stencil


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
    """Tests that build_delta_nu flattens 2D inputs in row-major order."""
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
    """Tests that wrong-length inputs raise ValueError."""
    theta0 = np.array([0.0])
    cov = np.eye(5)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))

    with pytest.raises(ValueError):
        lx.build_delta_nu(data_with=a, data_without=b)


def test_build_delta_nu_rejects_nonfinite_values():
    """Tests that non-finite values raise FloatingPointError."""
    theta0 = np.array([0.0])
    cov = np.eye(3)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    a = np.array([0.0, np.nan, 1.0])
    b = np.zeros(3)

    with pytest.raises(FloatingPointError):
        lx.build_delta_nu(data_with=a, data_without=b)
