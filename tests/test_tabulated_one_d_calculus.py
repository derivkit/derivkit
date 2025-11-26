"""Unit tests for calculus with 1D tabulated models."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.calculus_kit import CalculusKit
from derivkit.tabulated_model.one_d import Tabulated1DModel


def _make_tabulated_linear_scalar():
    """A linear tabulated model: f(x) = 3x + 1."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y_tab = 3.0 * x_tab + 1.0
    return Tabulated1DModel(x_tab, y_tab)


def _make_tabulated_linear_vector():
    """A vector-valued linear tabulated model."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y1 = x_tab
    y2 = 2.0 * x_tab
    y_tab = np.column_stack([y1, y2])
    return Tabulated1DModel(x_tab, y_tab)


def scalar_observable(theta: np.ndarray) -> float:
    """A tabulated scalar observable: observable = tabulated_scalar(theta[0])."""
    model = _make_tabulated_linear_scalar()
    return float(model(theta[0]))


def vector_observable(theta: np.ndarray) -> np.ndarray:
    """A tabulated vector observable: observable = tabulated_vector(theta[0])."""
    model = _make_tabulated_linear_vector()
    return np.asarray(model(theta[0]), dtype=float)


def _make_tabulated_quadratic(a: float = 2.0):
    """A quadratic tabulated model: f(x) = 0.5 a x^2."""
    x_tab = np.linspace(-2.0, 2.0, 101)
    y_tab = 0.5 * a * x_tab**2
    return Tabulated1DModel(x_tab, y_tab)


def _make_tabulated_identity():
    """A tabulated identity model: f(x) = x."""
    x_tab = np.linspace(-2.0, 2.0, 101)
    y_tab = x_tab
    return Tabulated1DModel(x_tab, y_tab)


def quadratic_scalar(theta: np.ndarray, a: float = 2.0, b: float = 1.5) -> float:
    """A scalar observable with quadratic tabulated dependence on θ0."""
    model = _make_tabulated_identity()
    x0 = float(model(theta[0]))
    term1 = 0.5 * a * x0**2
    term2 = b * theta[1]
    return term1 + term2


def _make_tabulated_linear_tensor():
    """A tabulated 2x2 tensor model."""
    x_tab = np.linspace(-2.0, 2.0, 41)
    y11 = 1.0 * x_tab
    y12 = 2.0 * x_tab
    y21 = 3.0 * x_tab
    y22 = 4.0 * x_tab
    y_tab = np.array([[y11, y12], [y21, y22]]).transpose(2, 0, 1)
    return Tabulated1DModel(x_tab, y_tab)


def tensor_observable(theta: np.ndarray) -> np.ndarray:
    """A tabulated tensor observable: observable = tabulated_tensor(theta[0])."""
    model = _make_tabulated_linear_tensor()
    mat = model(theta[0])
    return np.asarray(mat, dtype=float).ravel(order="C")


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
def test_gradient_tabulated_scalar(method: str):
    """Tests that gradient of tabulated scalar function is correct."""
    theta0 = np.array([0.3, -0.7])

    calc = CalculusKit(scalar_observable, theta0)
    grad = calc.gradient(method=method)

    grad = np.asarray(grad, dtype=float).ravel(order="C")

    assert grad.shape == (2,)
    np.testing.assert_allclose(grad, [3.0, 0.0], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
def test_jacobian_tabulated_vector(method: str):
    """Tests that Jacobian of tabulated vector function is correct."""
    theta0 = np.array([0.3, -0.7])

    calc = CalculusKit(vector_observable, theta0)
    jac = calc.jacobian(method=method)

    assert jac.shape == (2, 2)
    expected = np.array([[1.0, 0.0],
                         [2.0, 0.0]])
    np.testing.assert_allclose(jac, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
def test_hessian_tabulated_quadratic(method: str):
    """Tests that Hessian of tabulated quadratic function is correct."""
    theta0 = np.array([0.7, -1.2])
    a = 2.0
    b = 1.5

    calc = CalculusKit(lambda t: quadratic_scalar(t, a=a, b=b), theta0)
    hess = calc.hessian(method=method)

    assert hess.shape == (2, 2)
    expected = np.array([[a, 0.0],
                         [0.0, 0.0]])
    np.testing.assert_allclose(hess, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("method", ["adaptive", "finite", "lp"])
def test_jacobian_tabulated_tensor(method: str):
    """Tests that Jacobian of tabulated tensor function is correct."""
    theta0 = np.array([0.3, -0.7])

    calc = CalculusKit(tensor_observable, theta0)
    jac = calc.jacobian(method=method)

    assert jac.shape == (4, 2)

    expected_dtheta0 = np.array([[1.0, 2.0], [3.0, 4.0]]).ravel(order="C")
    expected = np.column_stack([expected_dtheta0, np.zeros_like(expected_dtheta0)])

    np.testing.assert_allclose(jac, expected, rtol=1e-6, atol=1e-8)
