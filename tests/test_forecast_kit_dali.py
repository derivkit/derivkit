"""Tests for doublet-DALI tensors in LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)  # this is a phantom use of theta
    return np.zeros(3, dtype=float)


@pytest.fixture
def dali_mocks(monkeypatch):
    """Provides fake derivative + covariance inversion with per-test state."""

    class DALIMocks:
        """Holds state and fake methods for mocking LikelihoodExpansion behavior."""
        def __init__(self):
            """Initialises empty state."""
            self.d1 = None
            self.d2 = None
            self.invcov = None
            self.deriv_calls: list[dict] = []
            self.invcov_call_info: dict | None = None

        def set_state(self, d1, d2, invcov):
            """Sets the internal state for the fakes."""
            self.d1 = np.asarray(d1, dtype=float)
            self.d2 = np.asarray(d2, dtype=float)
            self.invcov = np.asarray(invcov, dtype=float)
            self.deriv_calls = []
            self.invcov_call_info = None

        def fake_get_derivatives(self, *args, **kwargs):
            """Mimics LikelihoodExpansion._get_derivatives, returns d1 or d2 based on 'order'."""
            self.deriv_calls.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            order = kwargs.get("order")
            if order == 1:
                return self.d1
            if order == 2:
                return self.d2
            raise ValueError(f"Unexpected order in fake_get_derivatives: {order}")

        def fake_invert_covariance(self, cov, warn_prefix=None):
            """Mimics invert_covariance, returns invcov and records call info."""
            cov_arr = np.asarray(cov, dtype=float)
            self.invcov_call_info = {
                "cov": cov_arr,
                "warn_prefix": warn_prefix,
            }
            return self.invcov

    mocks = DALIMocks()

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.invert_covariance",
        mocks.fake_invert_covariance,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.LikelihoodExpansion._get_derivatives",
        mocks.fake_get_derivatives,
        raising=True,
    )

    return mocks


def test_build_dali_matches_reference_values():
    """Tests that _build_dali matches precomputed reference tensors."""

    d1 = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
        ]
    )

    d2 = np.array(
        [
            [
                [1.0, 0.0, 2.0],
                [0.5, 1.5, -0.5],
            ],
            [
                [0.2, -0.1, 0.0],
                [1.0, 0.5, 0.3],
            ],
        ]
    )

    cov = np.diag([2.0, 1.0, 0.5])
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(function=three_obs_model,
                             theta0=np.array([0.0, 0.0]),
                             cov=np.eye(3))

    g, h = lx._build_dali(d1, d2, invcov)

    g_expected = np.array(
        [
            [[12.5, 16.25],
             [0.25, -5.375]],
            [[-0.1, 0.15],
             [3.3, 2.15]],
        ]
    )

    h_expected = np.array(
        [
            [
                [[8.5, -1.75],
                 [0.1, 1.7]],
                [[-1.75, 2.875],
                 [-0.1, 0.7]],
            ],
            [
                [[0.1, -0.1],
                 [0.03, 0.05]],
                [[1.7, 0.7],
                 [0.05, 0.93]],
            ],
        ]
    )

    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    np.testing.assert_allclose(g, g_expected)
    np.testing.assert_allclose(h, h_expected)


def test_build_dali_symmetry_with_symmetric_inputs():
    """Tests that _build_dali produces symmetric tensors when given symmetric inputs."""
    rng = np.random.default_rng(123)

    n_params = 3
    n_obs = 4

    d2_raw = rng.normal(size=(n_params, n_params, n_obs))
    d2 = 0.5 * (d2_raw + np.swapaxes(d2_raw, 0, 1))

    d1 = rng.normal(size=(n_params, n_obs))

    a = rng.normal(size=(n_obs, n_obs))
    cov = a @ a.T + np.eye(n_obs)
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(
        function=three_obs_model,
        theta0=np.zeros(n_params),
        cov=np.eye(n_obs),
    )

    g, h = lx._build_dali(d1, d2, invcov)

    np.testing.assert_allclose(h, np.swapaxes(h, 0, 1))
    np.testing.assert_allclose(h, np.swapaxes(h, 2, 3))
    np.testing.assert_allclose(g, np.swapaxes(g, 0, 1))


def test_get_forecast_tensors_order2_builds_dali(dali_mocks):
    """Tests that get_forecast_tensors with order=2 calls _build_dali with correct inputs."""
    theta0 = np.array([0.1, -0.2])
    cov = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.0, 0.2, 0.8],
        ]
    )

    d1 = np.array(
        [
            [1.0, 0.5, -0.2],
            [0.0, 1.0, 0.3],
        ]
    )
    d2 = np.array(
        [
            [
                [0.1, 0.0, 0.2],
                [0.0, 0.2, -0.1],
            ],
            [
                [0.3, -0.2, 0.0],
                [0.4, 0.1, 0.1],
            ],
        ]
    )
    invcov = np.linalg.inv(cov)

    dali_mocks.set_state(d1, d2, invcov)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    g, h = lx.get_forecast_tensors(
        forecast_order=2,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    expected_g, expected_h = lx._build_dali(dali_mocks.d1, dali_mocks.d2, dali_mocks.invcov)

    assert g.shape == expected_g.shape == (2, 2, 2)
    assert h.shape == expected_h.shape == (2, 2, 2, 2)

    np.testing.assert_allclose(g, expected_g)
    np.testing.assert_allclose(h, expected_h)

    invcov_info = dali_mocks.invcov_call_info
    np.testing.assert_allclose(invcov_info["cov"], cov)
    assert invcov_info["warn_prefix"] == "LikelihoodExpansion"

    assert len(dali_mocks.deriv_calls) == 2
    orders = {call["kwargs"]["order"] for call in dali_mocks.deriv_calls}
    assert orders == {1, 2}

    for call in dali_mocks.deriv_calls:
        kwargs = call["kwargs"]
        assert kwargs["method"] == "adaptive"
        assert kwargs["n_workers"] == 4
        assert kwargs["step_size"] == 1e-3


@pytest.mark.parametrize("method", ["adaptive", "finite", "local_polyfit"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_get_forecast_tensors_order2_forwards_derivative_kwargs(
    dali_mocks,
    method,
    extrapolation,
    stencil,
):
    """Tests that get_forecast_tensors forwards derivative kwargs correctly."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    d1 = np.zeros((2, 3))
    d2 = np.zeros((2, 2, 3))
    invcov = np.eye(3)

    dali_mocks.set_state(d1, d2, invcov)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    g, h = lx.get_forecast_tensors(
        forecast_order=2,
        method=method,
        n_workers=5,
        extrapolation=extrapolation,
        stencil=stencil,
    )

    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    assert len(dali_mocks.deriv_calls) == 2
    for call in dali_mocks.deriv_calls:
        kwargs = call["kwargs"]
        assert kwargs["n_workers"] == 5
        assert kwargs["order"] in (1, 2)

        if method == "local_polyfit":
            assert kwargs["method"] == "local_polyfit"
        else:
            assert kwargs["method"] == method
            assert kwargs["extrapolation"] == extrapolation
            assert kwargs["stencil"] == stencil


def test_build_dali_p_equals_one_scalar_case():
    """Tests that _build_dali works correctly for the scalar P=1 case."""
    n_params = 1
    n_obs = 2

    d1 = np.array([[2.0, 3.0]])
    d2 = np.array([[[5.0, -1.0]]])

    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(
        function=three_obs_model,
        theta0=np.zeros(n_params),
        cov=np.eye(n_obs),
    )

    g, h = lx._build_dali(d1, d2, invcov)

    g_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d1[0])
    h_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d2[0, 0])

    assert g.shape == (1, 1, 1)
    assert h.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(g[0, 0, 0], g_manual)
    np.testing.assert_allclose(h[0, 0, 0, 0], h_manual)
