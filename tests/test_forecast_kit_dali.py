"""Tests for doublet-DALI tensors in LikelihoodExpansion."""

import pytest

import numpy as np

from derivkit.forecasting.expansions import LikelihoodExpansion


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)
    return np.zeros(3, dtype=float)


D1_GLOBAL = None
D2_GLOBAL = None
DERIV_CALLS: list[dict] = []
INVCOV_GLOBAL = None
INVCOV_CALL_INFO: dict = {}


def fake_get_derivatives(*args, **kwargs):
    """Mimics LikelihoodExpansion._get_derivatives, returning D1 or D2 based on 'order'."""
    global DERIV_CALLS
    DERIV_CALLS.append(
        {
            "args": args,
            "kwargs": kwargs,
        }
    )
    order = kwargs.get("order")
    if order == 1:
        return D1_GLOBAL
    if order == 2:
        return D2_GLOBAL
    raise ValueError(f"Unexpected order in fake_get_derivatives: {order}")


def fake_invert_covariance(cov, warn_prefix=None):
    """Mimics invert_covariance by returning a global INVCOV_GLOBAL and recording call info."""
    global INVCOV_CALL_INFO
    cov_arr = np.asarray(cov, dtype=float)
    INVCOV_CALL_INFO = {
        "cov": cov_arr,
        "warn_prefix": warn_prefix,
    }
    return INVCOV_GLOBAL


def test_build_dali_matches_einsum():
    """Tests that _build_dali produces the expected results via einsum."""
    # P = 2 parameters, N = 3 observables
    d1 = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 4.0],
        ]
    )  # (P, N) = (2, 3)

    d2 = np.array(
        [
            # a = 0
            [
                [1.0, 0.0, 2.0],  # b = 0
                [0.5, 1.5, -0.5],  # b = 1
            ],
            # a = 1
            [
                [0.2, -0.1, 0.0],
                [1.0, 0.5, 0.3],
            ],
        ]
    )  # (P, P, N) = (2, 2, 3)

    cov = np.diag([2.0, 1.0, 0.5])
    invcov = np.linalg.inv(cov)

    theta0 = np.array([0.0, 0.0])
    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=np.eye(3))

    g, h = lx._build_dali(d1, d2, invcov)

    # Manual definitions via einsum, matching the implementation
    g_manual = np.einsum("abi,ij,cj->abc", d2, invcov, d1)
    h_manual = np.einsum("abi,ij,cdj->abcd", d2, invcov, d2)

    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    np.testing.assert_allclose(g, g_manual)
    np.testing.assert_allclose(h, h_manual)


def test_build_dali_symmetry_with_symmetric_inputs():
    """Tests that _build_dali outputs have correct symmetries when given symmetric inputs."""
    rng = np.random.default_rng(123)

    n_params = 3  # parameters
    n_obs = 4  # observables

    # Symmetric second derivative tensor in first two indices
    d2_raw = rng.normal(size=(n_params, n_params, n_obs))
    d2 = 0.5 * (d2_raw + np.swapaxes(d2_raw, 0, 1))

    # Some first derivatives
    d1 = rng.normal(size=(n_params, n_obs))

    # Symmetric positive-definite covariance
    a = rng.normal(size=(n_obs, n_obs))
    cov = a @ a.T + np.eye(n_obs)
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(function=three_obs_model, theta0=np.zeros(n_params), cov=np.eye(n_obs))

    g, h = lx._build_dali(d1, d2, invcov)

    # Symmetry checks
    np.testing.assert_allclose(h, np.swapaxes(h, 0, 1))  # swap a <-> b
    np.testing.assert_allclose(h, np.swapaxes(h, 2, 3))  # swap c <-> d

    # G should be symmetric in the first two indices as well
    np.testing.assert_allclose(g, np.swapaxes(g, 0, 1))


def test_get_forecast_tensors_order2_builds_dali(monkeypatch):
    """Tests that get_forecast_tensors with order=2 builds DALI tensors correctly."""
    theta0 = np.array([0.1, -0.2])
    cov = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.0, 0.2, 0.8],
        ]
    )

    global D1_GLOBAL, D2_GLOBAL, DERIV_CALLS, INVCOV_GLOBAL, INVCOV_CALL_INFO
    DERIV_CALLS = []
    INVCOV_CALL_INFO = {}

    # P = 2, N = 3
    D1_GLOBAL = np.array(
        [
            [1.0, 0.5, -0.2],
            [0.0, 1.0, 0.3],
        ]
    )
    D2_GLOBAL = np.array(
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

    INVCOV_GLOBAL = np.linalg.inv(cov)

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.invert_covariance",
        fake_invert_covariance,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.LikelihoodExpansion._get_derivatives",
        fake_get_derivatives,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    g, h = lx.get_forecast_tensors(
        forecast_order=2,
        method="adaptive",
        n_workers=4,
        step_size=1e-3,
    )

    # Shape + numeric check vs _build_dali with our globals
    expected_g, expected_h = lx._build_dali(D1_GLOBAL, D2_GLOBAL, INVCOV_GLOBAL)

    assert g.shape == expected_g.shape == (2, 2, 2)
    assert h.shape == expected_h.shape == (2, 2, 2, 2)

    np.testing.assert_allclose(g, expected_g)
    np.testing.assert_allclose(h, expected_h)

    # invert_covariance got the stored cov and correct prefix
    np.testing.assert_allclose(INVCOV_CALL_INFO["cov"], cov)
    assert INVCOV_CALL_INFO["warn_prefix"] == "LikelihoodExpansion"

    # _get_derivatives was called twice: once for order=1 and once for order=2
    assert len(DERIV_CALLS) == 2
    orders = [call["kwargs"]["order"] for call in DERIV_CALLS]
    assert set(orders) == {1, 2}

    # Check kwargs forwarding for at least one of the calls
    for call in DERIV_CALLS:
        assert call["kwargs"]["method"] == "adaptive"
        assert call["kwargs"]["n_workers"] == 4
        assert call["kwargs"]["step_size"] == 1e-3


@pytest.mark.parametrize("method", ["adaptive", "finite", "local_polyfit"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_get_forecast_tensors_order2_forwards_derivative_kwargs(
    monkeypatch,
    method,
    extrapolation,
    stencil,
):
    """All derivative-related kwargs should be forwarded to _get_derivatives for D=2."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    global D1_GLOBAL, D2_GLOBAL, DERIV_CALLS, INVCOV_GLOBAL, INVCOV_CALL_INFO
    D1_GLOBAL = np.zeros((2, 3))
    D2_GLOBAL = np.zeros((2, 2, 3))
    DERIV_CALLS = []
    INVCOV_CALL_INFO = {}
    INVCOV_GLOBAL = np.eye(3)

    monkeypatch.setattr(
        "derivkit.forecasting.expansions.invert_covariance",
        fake_invert_covariance,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecasting.expansions.LikelihoodExpansion._get_derivatives",
        fake_get_derivatives,
        raising=True,
    )

    lx = LikelihoodExpansion(function=three_obs_model, theta0=theta0, cov=cov)

    g, h = lx.get_forecast_tensors(
        forecast_order=2,
        method=method,
        n_workers=5,
        extrapolation=extrapolation,
        stencil=stencil,
    )

    # Just check shapes to ensure the code path ran
    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    # _get_derivatives should have been called for both orders with the same kwargs
    assert len(DERIV_CALLS) == 2
    for call in DERIV_CALLS:
        kwargs = call["kwargs"]
        assert kwargs["method"] == method
        assert kwargs["n_workers"] == 5
        assert kwargs["extrapolation"] == extrapolation
        assert kwargs["stencil"] == stencil  # <-- use the parametrized value
        assert kwargs["order"] in (1, 2)


def test_build_dali_p_equals_one_scalar_case():
    """Tests that _build_dali works correctly for the scalar P=1 case."""
    # One parameter, two observables
    n_params = 1
    n_obs = 2

    d1 = np.array([[2.0, 3.0]])  # shape (1, 2)
    d2 = np.array([[[5.0, -1.0]]])  # shape (1, 1, 2)

    cov = np.array([[2.0, 0.5],
                    [0.5, 1.0]])
    invcov = np.linalg.inv(cov)

    lx = LikelihoodExpansion(function=three_obs_model,
                             theta0=np.zeros(n_params),
                             cov=np.eye(n_obs))

    g, h = lx._build_dali(d1, d2, invcov)

    # Manual scalar versions:
    # G_000 = sum_ij d2[0,0,i] invcov[i,j] d1[0,j]
    g_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d1[0])
    # H_0000 = sum_ij d2[0,0,i] invcov[i,j] d2[0,0,j]
    h_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d2[0, 0])

    assert g.shape == (1, 1, 1)
    assert h.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(g[0, 0, 0], g_manual)
    np.testing.assert_allclose(h[0, 0, 0, 0], h_manual)
