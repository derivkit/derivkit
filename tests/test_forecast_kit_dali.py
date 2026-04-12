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


@pytest.mark.parametrize("method", ["adaptive", "finite", "polyfit"])
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


def analytic_nonlinear_model(theta):
    """Smooth nonlinear 2-parameter model with nonzero mixed and second derivatives."""
    x, y = np.asarray(theta, dtype=float)

    return np.array(
        [
            np.exp(0.3 * x + 0.2 * y) + 0.1 * x * y,
            np.sin(0.5 * x - 0.4 * y) + 0.05 * x**2,
            x**2 + 0.7 * x * y + 0.2 * y**3,
            np.cos(0.2 * x * y) + 0.3 * y,
        ],
        dtype=float,
    )


def analytic_nonlinear_model_d1(theta):
    """Analytic first derivatives of the smooth nonlinear test model."""
    x, y = np.asarray(theta, dtype=float)

    e = np.exp(0.3 * x + 0.2 * y)
    sxy = np.sin(0.2 * x * y)

    return np.array(
        [
            [
                0.3 * e + 0.1 * y,
                0.5 * np.cos(0.5 * x - 0.4 * y) + 0.1 * x,
                2.0 * x + 0.7 * y,
                -0.2 * y * sxy,
            ],
            [
                0.2 * e + 0.1 * x,
                -0.4 * np.cos(0.5 * x - 0.4 * y),
                0.7 * x + 0.6 * y**2,
                -0.2 * x * sxy + 0.3,
            ],
        ],
        dtype=float,
    )


def analytic_nonlinear_model_d2(theta):
    """Analytic second derivatives of the smooth nonlinear test model."""
    x, y = np.asarray(theta, dtype=float)

    e = np.exp(0.3 * x + 0.2 * y)
    arg = 0.5 * x - 0.4 * y
    s = np.sin(arg)
    sxy = np.sin(0.2 * x * y)
    cxy = np.cos(0.2 * x * y)

    d2 = np.zeros((2, 2, 4), dtype=float)

    d2[0, 0, 0] = 0.09 * e
    d2[0, 1, 0] = 0.06 * e + 0.1
    d2[1, 0, 0] = d2[0, 1, 0]
    d2[1, 1, 0] = 0.04 * e

    d2[0, 0, 1] = -0.25 * s + 0.1
    d2[0, 1, 1] = 0.2 * s
    d2[1, 0, 1] = d2[0, 1, 1]
    d2[1, 1, 1] = -0.16 * s

    d2[0, 0, 2] = 2.0
    d2[0, 1, 2] = 0.7
    d2[1, 0, 2] = 0.7
    d2[1, 1, 2] = 1.2 * y

    d2[0, 0, 3] = -(0.2 * y) ** 2 * cxy
    d2[1, 1, 3] = -(0.2 * x) ** 2 * cxy
    d2[0, 1, 3] = -0.2 * sxy - 0.04 * x * y * cxy
    d2[1, 0, 3] = d2[0, 1, 3]

    return d2


def analytic_reference_dali(theta0, cov):
    """Builds exact DALI tensors from analytic first and second derivatives."""
    d1 = analytic_nonlinear_model_d1(theta0)
    d2 = analytic_nonlinear_model_d2(theta0)
    invcov = np.linalg.inv(np.asarray(cov, dtype=float))

    g = np.einsum("abi,ij,gj->abg", d2, invcov, d1)
    h = np.einsum("abi,ij,gdj->abgd", d2, invcov, d2)

    return g, h


@pytest.mark.parametrize(
    ("method", "build_kwargs", "rtol", "atol"),
    [
        ("finite", {"extrapolation": None, "num_points": 5, "stepsize": 1e-4}, 2e-3, 2e-5),
        ("finite", {"extrapolation": "ridders", "num_points": 5, "stepsize": 1e-3}, 5e-4, 5e-6),
        ("finite", {"extrapolation": "richardson", "num_points": 5, "stepsize": 1e-3}, 1e-3, 1e-5),
        ("polyfit", {"degree": 4}, 1e-2, 5e-5),
        ("adaptive", {"n_points": 10, "spacing": "auto"}, 1e-2, 5e-5),
    ],
)
def test_build_dali_matches_analytic_reference_across_methods(
    method,
    build_kwargs,
    rtol,
    atol,
):
    """Tests that build_dali agrees with analytic DALI tensors for a nonlinear model across derivative methods."""
    theta0 = np.array([0.31, -0.27], dtype=float)
    cov = np.array(
        [
            [1.4, 0.2, 0.1, 0.0],
            [0.2, 1.1, 0.15, 0.05],
            [0.1, 0.15, 0.9, 0.12],
            [0.0, 0.05, 0.12, 1.2],
        ],
        dtype=float,
    )

    g_expected, h_expected = analytic_reference_dali(theta0, cov)

    dali = build_dali(
        analytic_nonlinear_model,
        theta0,
        cov,
        method=method,
        forecast_order=2,
        n_workers=1,
        **build_kwargs,
    )
    g, h = dali[2]

    assert g.shape == g_expected.shape
    assert h.shape == h_expected.shape

    np.testing.assert_allclose(g, g_expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h, h_expected, rtol=rtol, atol=atol)


def harder_nonlinear_model(theta):
    """More curved nonlinear model used to compare derivative methods against each other."""
    x, y = np.asarray(theta, dtype=float)

    return np.array(
        [
            np.exp((x - 0.7 * y) ** 2 + 0.15 * y),
            np.exp(0.3 * x) * (1.0 + 0.4 * y + 0.8 * y**2),
            np.sin(0.8 * x + 0.3 * y) + 0.2 * x**3 - 0.1 * x * y**2,
            np.cos(0.4 * x * y) + 0.05 * x**2 * y,
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "method,build_kwargs",
    [
        ("finite", {"extrapolation": None, "num_points": 5, "stepsize": 1e-4}),
        ("finite", {"extrapolation": "ridders", "num_points": 5, "stepsize": 1e-3}),
        ("finite", {"extrapolation": "richardson", "num_points": 5, "stepsize": 1e-3}),
        ("polyfit", {"degree": 4}),
        ("adaptive", {"n_points": 10, "spacing": "auto"}),
    ],
)
def test_build_dali_methods_remain_in_the_same_ballpark_on_harder_model(
    method,
    build_kwargs,
):
    """Tests that different derivative methods return DALI tensors in the same numerical ballpark on a harder nonlinear model."""
    theta0 = np.array([0.18, 0.06], dtype=float)
    cov = np.array(
        [
            [1.0, 0.3, 0.2, 0.0],
            [0.3, 1.2, 0.25, 0.1],
            [0.2, 0.25, 0.95, 0.2],
            [0.0, 0.1, 0.2, 1.1],
        ],
        dtype=float,
    )

    dali_ref = build_dali(
        harder_nonlinear_model,
        theta0,
        cov,
        method="finite",
        forecast_order=2,
        n_workers=1,
        extrapolation="ridders",
        num_points=5,
        stepsize=1e-3,
    )
    g_ref, h_ref = dali_ref[2]

    dali_test = build_dali(
        harder_nonlinear_model,
        theta0,
        cov,
        method=method,
        forecast_order=2,
        n_workers=1,
        **build_kwargs,
    )
    g_test, h_test = dali_test[2]

    g_ref_norm = np.linalg.norm(g_ref)
    h_ref_norm = np.linalg.norm(h_ref)

    g_diff = np.linalg.norm(g_test - g_ref)
    h_diff = np.linalg.norm(h_test - h_ref)

    assert g_diff <= 0.25 * max(g_ref_norm, 1e-12)
    assert h_diff <= 0.35 * max(h_ref_norm, 1e-12)


@pytest.mark.parametrize(
    "method,build_kwargs",
    [
        ("finite", {"extrapolation": None, "num_points": 5, "stepsize": 1e-4}),
        ("finite", {"extrapolation": "ridders", "num_points": 5, "stepsize": 1e-3}),
        ("finite", {"extrapolation": "richardson", "num_points": 5, "stepsize": 1e-3}),
        ("polyfit", {"degree": 4}),
        ("adaptive", {"n_points": 10, "spacing": "auto"}),
    ],
)
def test_build_dali_preserves_expected_tensor_symmetries_for_real_methods(
    method,
    build_kwargs,
):
    """Tests that real derivative backends preserve the expected DALI tensor symmetries."""
    theta0 = np.array([0.31, -0.27], dtype=float)
    cov = np.array(
        [
            [1.4, 0.2, 0.1, 0.0],
            [0.2, 1.1, 0.15, 0.05],
            [0.1, 0.15, 0.9, 0.12],
            [0.0, 0.05, 0.12, 1.2],
        ],
        dtype=float,
    )

    dali = build_dali(
        analytic_nonlinear_model,
        theta0,
        cov,
        method=method,
        forecast_order=2,
        n_workers=1,
        **build_kwargs,
    )
    g, h = dali[2]

    np.testing.assert_allclose(g, np.swapaxes(g, 0, 1), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h, np.swapaxes(h, 0, 1), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h, np.swapaxes(h, 2, 3), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "method,build_kwargs",
    [
        ("finite", {"extrapolation": None, "num_points": 5, "stepsize": 1e-4}),
        ("finite", {"extrapolation": "ridders", "num_points": 5, "stepsize": 1e-3}),
        ("finite", {"extrapolation": "richardson", "num_points": 5, "stepsize": 1e-3}),
        ("polyfit", {"degree": 4}),
        ("adaptive", {"n_points": 10, "spacing": "auto"}),
    ],
)
def test_build_dali_returns_small_tensors_for_linear_model_across_methods(
    method,
    build_kwargs,
):
    """Tests that build_dali returns numerically tiny tensors across derivative methods."""
    def linear_model(theta):
        """Strictly linear observable model with zero second derivatives."""
        x, y = np.asarray(theta, dtype=float)
        return np.array(
            [
                2.0 * x - 1.5 * y + 0.2,
                -0.3 * x + 4.1 * y,
                1.2 * x + 0.7 * y - 0.8,
                -2.5 * x + 0.5 * y + 3.0,
            ],
            dtype=float,
        )

    theta0 = np.array([0.2, -0.1], dtype=float)
    cov = np.array(
        [
            [1.2, 0.1, 0.0, 0.0],
            [0.1, 1.1, 0.05, 0.0],
            [0.0, 0.05, 0.9, 0.1],
            [0.0, 0.0, 0.1, 1.3],
        ],
        dtype=float,
    )

    dali = build_dali(
        linear_model,
        theta0,
        cov,
        method=method,
        forecast_order=2,
        n_workers=1,
        **build_kwargs,
    )
    g, h = dali[2]

    assert np.linalg.norm(g) < 3e-6
    assert np.linalg.norm(h) < 3e-6
