"""Tests for derivkit.forecasting.dali."""

import numpy as np
import pytest

from derivkit.forecasting.dali import build_dali


def three_obs_model(theta):
    """Model that always returns a 3-element observable vector."""
    _ = np.atleast_1d(theta)  # phantom use of theta
    return np.zeros(3, dtype=float)


@pytest.fixture
def dali_mocks(monkeypatch):
    """Provides fake forecast tensors with per-test state."""

    class DALIMocks:
        """Holds state and fake method for mocking get_forecast_tensors behavior."""

        def __init__(self):
            """Initialises empty state."""
            self.g = None
            self.h = None
            self.calls: list[dict] = []

        def set_state(self, g, h):
            """Sets the internal state for the fake get_forecast_tensors."""
            self.g = np.asarray(g, dtype=float)
            self.h = np.asarray(h, dtype=float)

        def fake_get_forecast_tensors(
            self,
            function,
            theta0,
            cov,
            *,
            forecast_order,
            method,
            n_workers,
            **dk_kwargs,
        ):
            """Mimics get_forecast_tensors, returns G and H."""
            self.calls.append(
                {
                    "function": function,
                    "theta0": np.asarray(theta0, dtype=float),
                    "cov": np.asarray(cov, dtype=float),
                    "forecast_order": forecast_order,
                    "method": method,
                    "n_workers": n_workers,
                    "dk_kwargs": dk_kwargs,
                }
            )
            return self.g, self.h

    mocks = DALIMocks()

    monkeypatch.setattr(
        "derivkit.forecasting.dali.get_forecast_tensors",
        mocks.fake_get_forecast_tensors,
        raising=True,
    )

    return mocks


def test_build_dali_matches_reference_values(dali_mocks):
    """Tests that build_dali returns expected DALI tensors for known reference values."""
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

    dali_mocks.set_state(g_expected, h_expected)

    theta0 = np.array([0.0, 0.0])
    cov = np.diag([2.0, 1.0, 0.5])

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method="adaptive",
        n_workers=2,
    )

    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(g, g_expected)
    np.testing.assert_allclose(h, h_expected)

    assert len(dali_mocks.calls) == 1
    call = dali_mocks.calls[0]
    assert call["function"] is three_obs_model
    np.testing.assert_allclose(call["theta0"], theta0)
    np.testing.assert_allclose(call["cov"], cov)
    assert call["forecast_order"] == 2
    assert call["method"] == "adaptive"
    assert call["n_workers"] == 2


def test_build_dali_symmetry_with_symmetric_inputs(dali_mocks):
    """Tests that build_dali returns symmetric DALI tensors when given symmetric inputs."""
    rng = np.random.default_rng(42)
    p = 3

    g_raw = rng.normal(size=(p, p, p))
    g_sym = 0.5 * (g_raw + np.swapaxes(g_raw, 0, 1))

    h_raw = rng.normal(size=(p, p, p, p))
    h_sym = 0.25 * (
        h_raw
        + np.swapaxes(h_raw, 0, 1)
        + np.swapaxes(h_raw, 2, 3)
        + np.swapaxes(np.swapaxes(h_raw, 0, 1), 2, 3)
    )

    dali_mocks.set_state(g_sym, h_sym)

    theta0 = np.zeros(p)
    cov = np.eye(4)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method="adaptive",
        n_workers=1,
    )

    assert g.shape == (p, p, p)
    assert h.shape == (p, p, p, p)

    np.testing.assert_allclose(g, np.swapaxes(g, 0, 1))
    np.testing.assert_allclose(h, np.swapaxes(h, 0, 1))
    np.testing.assert_allclose(h, np.swapaxes(h, 2, 3))


def test_build_dali_p_equals_one_scalar_case(dali_mocks):
    """Tests that build_dali handles the scalar reference value case correctly (shapes + scalar)."""
    n_params = 1

    d1 = np.array([[2.0, 3.0]])
    d2 = np.array([[[5.0, -1.0]]])

    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    invcov = np.linalg.inv(cov)

    g_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d1[0])
    h_manual = np.einsum("i,ij,j->", d2[0, 0], invcov, d2[0, 0])

    g_expected = np.array([[[g_manual]]])
    h_expected = np.array([[[[h_manual]]]])

    dali_mocks.set_state(g_expected, h_expected)

    theta0 = np.zeros(n_params)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method="adaptive",
        n_workers=1,
    )

    assert g.shape == (1, 1, 1)
    assert h.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(g[0, 0, 0], g_manual)
    np.testing.assert_allclose(h[0, 0, 0, 0], h_manual)


def test_build_dali_allows_method_none(dali_mocks):
    """Tests that build_dali allows method=None and forwards it correctly."""
    theta0 = np.zeros(2)
    cov = np.eye(3)

    g_expected = np.zeros((2, 2, 2))
    h_expected = np.zeros((2, 2, 2, 2))
    dali_mocks.set_state(g_expected, h_expected)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method=None,
        n_workers=1,
    )

    np.testing.assert_allclose(g, g_expected)
    np.testing.assert_allclose(h, h_expected)

    assert len(dali_mocks.calls) == 1
    call = dali_mocks.calls[0]
    assert call["forecast_order"] == 2
    assert call["method"] is None


@pytest.mark.parametrize("method", ["adaptive", "finite", "local_polyfit"])
@pytest.mark.parametrize("extrapolation", ["richardson", "ridders", "gauss_richardson"])
@pytest.mark.parametrize("stencil", [3, 5, 7, 9])
def test_build_dali_forwards_derivative_kwargs(
    dali_mocks,
    method,
    extrapolation,
    stencil,
):
    """Tests that derivative method and its kwargs are forwarded to the internal derivative routine."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    g_expected = np.zeros((2, 2, 2))
    h_expected = np.zeros((2, 2, 2, 2))
    dali_mocks.set_state(g_expected, h_expected)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method=method,
        n_workers=5,
        extrapolation=extrapolation,
        stencil=stencil,
    )

    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    assert len(dali_mocks.calls) == 1
    call = dali_mocks.calls[0]

    assert call["n_workers"] == 5
    assert call["method"] == method

    dk_kwargs = call["dk_kwargs"]
    assert dk_kwargs["extrapolation"] == extrapolation
    assert dk_kwargs["stencil"] == stencil


def test_build_dali_uses_default_method_and_workers(dali_mocks):
    """Tests that build_dali forwards its default method=None and n_workers=1."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    g_expected = np.zeros((2, 2, 2))
    h_expected = np.zeros((2, 2, 2, 2))
    dali_mocks.set_state(g_expected, h_expected)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
    )

    np.testing.assert_allclose(g, g_expected)
    np.testing.assert_allclose(h, h_expected)

    assert len(dali_mocks.calls) == 1
    call = dali_mocks.calls[0]
    assert call["method"] is None
    assert call["n_workers"] == 1


def test_build_dali_forwards_arbitrary_extra_kwargs(dali_mocks):
    """Tests that arbitrary extra kwargs are forwarded to the internal derivative routine."""
    theta0 = np.array([0.0, 0.0])
    cov = np.eye(3)

    g_expected = np.zeros((2, 2, 2))
    h_expected = np.zeros((2, 2, 2, 2))
    dali_mocks.set_state(g_expected, h_expected)

    g, h = build_dali(
        three_obs_model,
        theta0,
        cov,
        method="adaptive",
        n_workers=3,
        foo="bar",
        tol=1e-4,
    )

    np.testing.assert_allclose(g, g_expected)
    np.testing.assert_allclose(h, h_expected)

    assert len(dali_mocks.calls) == 1
    call = dali_mocks.calls[0]
    dk_kwargs = call["dk_kwargs"]
    assert dk_kwargs["foo"] == "bar"
    assert dk_kwargs["tol"] == 1e-4
