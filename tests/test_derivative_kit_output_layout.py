"""Unit tests for DerivativeKit output layout for scalar, vector, and tensor functions."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit
from derivkit.finite.core import single_finite_step


def scalar_func(x: float) -> float:
    """A scalar linear function: f(x) = 3x + 1  → f'(x) = 3."""
    return 3.0 * x + 1.0


def vector_func(x: float) -> np.ndarray:
    """A 2D vector-valued linear function: f'(x) = [1, 2]."""
    return np.array([x, 2.0 * x])


def tensor_func(x: float) -> np.ndarray:
    """A tensor-valued linear function (2x2 matrix)."""
    return np.array([[x, 2.0 * x], [3.0 * x, 4.0 * x]])


# expected derivative values
EXPECTED_SCALAR = 3.0
EXPECTED_VECTOR = np.array([1.0, 2.0])
EXPECTED_TENSOR_MATRIX = np.array([[1.0, 2.0], [3.0, 4.0]])
EXPECTED_TENSOR_FLAT = EXPECTED_TENSOR_MATRIX.ravel(order="C")


_MODEL_CASES = [
    ("scalar", scalar_func, True, EXPECTED_SCALAR),
    ("vector", vector_func, False, EXPECTED_VECTOR),
    ("tensor", tensor_func, False, EXPECTED_TENSOR_FLAT),
]


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
@pytest.mark.parametrize("label, func, is_scalar, expected", _MODEL_CASES)
def test_derivativekit_layout_single_x0(method, label, func, is_scalar, expected):
    """Tests that DerivativeKit returns correct layout for single x0."""
    x0 = 0.3

    dk = DerivativeKit(func, x0)
    d1 = dk.differentiate(method=method, order=1)

    if is_scalar:
        assert np.isscalar(d1)
        assert float(d1) == pytest.approx(float(expected), rel=1e-8, abs=1e-10)
    else:
        assert isinstance(d1, np.ndarray)
        assert d1.shape == np.shape(expected)
        np.testing.assert_allclose(d1, expected, rtol=1e-8, atol=1e-10)


def test_derivativekit_layout_array_x0_tensor():
    """Tests that DerivativeKit returns correct layout for array x0 with tensor output."""
    xs = np.array([-1.0, 0.0, 0.5])
    dk = DerivativeKit(tensor_func, xs)

    d1_adaptive = dk.differentiate(method="adaptive", order=1)
    d1_finite = dk.differentiate(method="finite", order=1)
    d1_lp = dk.differentiate(method="lp", order=1)

    for d1 in (d1_adaptive, d1_finite, d1_lp):
        assert d1.shape == (xs.size, EXPECTED_TENSOR_FLAT.size)
        for row in d1:
            np.testing.assert_allclose(row, EXPECTED_TENSOR_FLAT, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "extrapolation",
    [None, "richardson", "ridders", "gauss-richardson"],
)
@pytest.mark.parametrize("label, func, is_scalar, expected", _MODEL_CASES)
def test_derivativekit_finite_extrapolation_layout_all(
    extrapolation,
    label,
    func,
    is_scalar,
    expected,
):
    """Tests that DerivativeKit finite method with extrapolation returns correct layout."""
    x0 = 0.7
    dk = DerivativeKit(func, x0)

    dk_kwargs = {"order": 1}
    if extrapolation is not None:
        dk_kwargs["extrapolation"] = extrapolation

    d1 = dk.differentiate(method="finite", **dk_kwargs)

    if is_scalar:
        assert np.isscalar(d1)
        assert float(d1) == pytest.approx(float(expected), rel=1e-8, abs=1e-10)
    else:
        assert isinstance(d1, np.ndarray)
        assert d1.shape == np.shape(expected)
        np.testing.assert_allclose(d1, expected, rtol=1e-8, atol=1e-10)


def test_single_finite_step_tensor_layout():
    """Tests that tensor-valued outputs are flattened correctly in C order."""
    x0 = 0.3
    order = 1
    stepsize = 1e-2
    num_points = 5
    n_workers = 1

    deriv = single_finite_step(
        tensor_func,
        x0=x0,
        order=order,
        stepsize=stepsize,
        num_points=num_points,
        n_workers=n_workers,
    )

    assert isinstance(deriv, np.ndarray)
    assert deriv.shape == EXPECTED_TENSOR_FLAT.shape
    np.testing.assert_allclose(deriv, EXPECTED_TENSOR_FLAT, rtol=1e-6, atol=1e-8)
