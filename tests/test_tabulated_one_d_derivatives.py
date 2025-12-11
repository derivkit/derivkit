"""Unit tests for derivatives on 1D tabulated models."""

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit
from derivkit.tabulated_model.one_d import Tabulated1DModel


def make_linear_scalar_model():
    """Creates a Tabulated1DModel for a scalar linear function."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y_tab = 3.0 * x_tab + 1.0
    return Tabulated1DModel(x_tab, y_tab)


def make_linear_vector_model():
    """Creates a Tabulated1DModel for a 2D vector function."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y1 = 3.0 * x_tab + 1.0
    y2 = -2.0 * x_tab + 0.5
    y_tab = np.column_stack([y1, y2])
    return Tabulated1DModel(x_tab, y_tab)


def make_linear_tensor_model():
    """Creates a Tabulated1DModel for a 2x2 tensor-valued linear function."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y11 = 2.0 * x_tab + 1.0
    y12 = -1.0 * x_tab + 0.0
    y21 = 0.5 * x_tab + 2.0
    y22 = -3.0 * x_tab + 4.0
    y_tab = np.array([[y11, y12], [y21, y22]]).transpose(2, 0, 1)
    return Tabulated1DModel(x_tab, y_tab)


# Expected derivative for the tensor case as a 2x2 matrix
_EXPECTED_TENSOR_MATRIX = np.array([[2.0, -1.0], [0.5, -3.0]])

_MODEL_CASES = [
    (
        "scalar",
        make_linear_scalar_model,
        True,
        3.0,
    ),
    (
        "vector",
        make_linear_vector_model,
        False,
        np.array([3.0, -2.0]),
    ),
    (
        "tensor",
        make_linear_tensor_model,
        False,
        _EXPECTED_TENSOR_MATRIX.ravel(order="C"),
    ),
]


@pytest.mark.parametrize("method", ["finite", "adaptive", "lp"])
@pytest.mark.parametrize("label, model_factory, is_scalar, expected", _MODEL_CASES)
def test_tabulated_linear_derivative_methods(label, model_factory, is_scalar, expected, method):
    """Tests that various derivative methods work on linear tabulated models."""
    model = model_factory()
    x0 = 0.3

    kit = DerivativeKit(model, x0)
    d1 = kit.differentiate(method=method, order=1)

    if is_scalar:
        assert np.isscalar(d1)
        assert float(d1) == pytest.approx(float(expected), rel=1e-6, abs=1e-8)
    else:
        assert isinstance(d1, np.ndarray)
        assert d1.shape == np.shape(expected)
        np.testing.assert_allclose(d1, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "extrapolation",
    [None, "richardson", "ridders", "gauss-richardson"],
)
@pytest.mark.parametrize("label, model_factory, is_scalar, expected", _MODEL_CASES)
def test_finite_derivative_extrapolation_linear(
    label,
    model_factory,
    is_scalar,
    expected,
    extrapolation,
):
    """Tests that finite difference with extrapolation works on linear tabulated models."""
    model = model_factory()
    x0 = -0.8

    kit = DerivativeKit(model, x0)

    dk_kwargs = {"order": 1}
    if extrapolation is not None:
        dk_kwargs["extrapolation"] = extrapolation

    d1 = kit.differentiate(method="finite", **dk_kwargs)

    if is_scalar:
        assert np.isscalar(d1)
        assert float(d1) == pytest.approx(float(expected), rel=1e-6, abs=1e-8)
    else:
        assert isinstance(d1, np.ndarray)
        assert d1.shape == np.shape(expected)
        np.testing.assert_allclose(d1, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["finite", "adaptive", "lp"])
@pytest.mark.parametrize("label, model_factory, is_scalar, expected", _MODEL_CASES)
def test_derivative_kit_tabulated_vs_callable(label, model_factory, is_scalar, expected, method):
    """Tests that DerivativeKit(tab_x, tab_y) matches DerivativeKit(function=model)."""
    model = model_factory()
    x0 = 0.3

    # via callable model
    kit_callable = DerivativeKit(function=model, x0=x0)
    d1_callable = kit_callable.differentiate(method=method, order=1)

    # reconstruct the original y_tab from the model internals
    x_tab = model.x
    if model._out_shape == ():
        y_tab = model.y_flat[:, 0]
    else:
        y_tab = model.y_flat.reshape(len(x_tab), *model._out_shape)

    # via tabulated mode
    kit_tab = DerivativeKit(tab_x=x_tab, tab_y=y_tab, x0=x0)
    d1_tab = kit_tab.differentiate(method=method, order=1)

    np.testing.assert_allclose(d1_tab, d1_callable, rtol=1e-12, atol=0.0)
