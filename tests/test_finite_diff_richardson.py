"""Tests for Richardson extrapolation support in FiniteDifferenceDerivative."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit
from derivkit.finite.batch_eval import eval_points
from derivkit.finite.finite_difference import FiniteDifferenceDerivative
from derivkit.utils.extrapolation import richardson_extrapolate


def _rel_err(a, b) -> float:
    """Relative error metric between a and b (max over all components)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / denom))


def _central_diff_scalar(f, x0: float, h: float) -> float:
    """Simple 3-point central difference for a scalar function."""
    return float((f(x0 + h) - f(x0 - h)) / (2.0 * h))


def _central_diff_sin_exp_vec(x0: float, h: float) -> np.ndarray:
    """3-point central difference for [sin(x), exp(x)]."""
    f_plus = np.array([np.sin(x0 + h), np.exp(x0 + h)], dtype=float)
    f_minus = np.array([np.sin(x0 - h), np.exp(x0 - h)], dtype=float)
    return (f_plus - f_minus) / (2.0 * h)


def _poly3(x: float) -> float:
    """Test cubic: f(x) = x^3 - 2x + 0.5."""
    return x**3 - 2.0 * x + 0.5


def _poly3_second_derivative(x: float) -> float:
    """Analytic second derivative of _poly3: f''(x) = 6x."""
    return 6.0 * x


def _vec_sin_cos(x: float) -> np.ndarray:
    """Vector-valued test function: [sin(x), cos(x)]."""
    return np.array([np.sin(x), np.cos(x)], dtype=float)


def _vec_sin_cos_derivative(x: float) -> np.ndarray:
    """Analytic derivative of _vec_sin_cos: [cos(x), -sin(x)]."""
    return np.array([np.cos(x), -np.sin(x)], dtype=float)


def test_richardson_extrapolate_requires_two_values():
    """Tests that at least two values are needed for Richardson extrapolation."""
    with pytest.raises(ValueError):
        richardson_extrapolate([1.0], p=2, r=2.0)


def test_richardson_extrapolate_improves_scalar_central_diff():
    """Tests that Richardson extrapolation improves central-difference estimate for sin'."""
    x0 = 0.3
    exact = np.cos(x0)

    h = 1e-1
    d1 = _central_diff_scalar(np.sin, x0, h)
    d2 = _central_diff_scalar(np.sin, x0, h / 2.0)

    improved = richardson_extrapolate([d1, d2], p=2, r=2.0)

    err1 = abs(d1 - exact)
    err2 = abs(d2 - exact)
    err_imp = abs(improved - exact)

    assert err_imp <= max(err1, err2)
    assert err_imp < 1e-6


def test_richardson_extrapolate_vector_works_componentwise():
    """Tests that Richardson extrapolation works for vector-valued functions."""
    x0 = 0.1
    exact = np.array([np.cos(x0), np.exp(x0)], dtype=float)

    h = 5e-2
    d1 = _central_diff_sin_exp_vec(x0, h)
    d2 = _central_diff_sin_exp_vec(x0, h / 2.0)
    d3 = _central_diff_sin_exp_vec(x0, h / 4.0)

    improved = richardson_extrapolate([d1, d2, d3], p=2, r=2.0)

    assert improved.shape == (2,)
    assert _rel_err(improved, exact) < 1e-6


@pytest.mark.parametrize("x0", [0.0, 0.3, -0.7])
@pytest.mark.parametrize("num_points", [5, 7, 9])  # combos with truncation order defined
def test_finite_richardson_fixed_levels_matches_analytic_sin(x0, num_points):
    """Tests that finite difference + Richardson matches analytic derivative of sin."""
    exact = np.cos(x0)

    d = FiniteDifferenceDerivative(function=np.sin, x0=x0)
    est = d.differentiate(
        order=1,
        stepsize=1e-2,
        num_points=num_points,
        extrapolation="richardson",
        levels=4,
    )

    assert isinstance(est, float)
    assert _rel_err(est, exact) < 1e-6


@pytest.mark.parametrize("x0", [0.0, 0.2])
def test_finite_richardson_fixed_levels_poly_second_derivative(x0):
    """Checks that (5-point, order=2) + Richardson uses the right truncation order."""
    exact = _poly3_second_derivative(x0)

    d = FiniteDifferenceDerivative(function=_poly3, x0=x0)
    est = d.differentiate(
        order=2,
        stepsize=5e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )

    assert isinstance(est, float)
    assert _rel_err(est, exact) < 1e-6


def test_finite_richardson_adaptive_close_to_fixed():
    """Tests that adaptive levels matches fixed-levels Richardson result."""
    x0 = 0.3
    d = FiniteDifferenceDerivative(function=np.sin, x0=x0)

    est_fixed = d.differentiate(
        order=1,
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )

    est_adapt = d.differentiate(
        order=1,
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=None,
    )

    assert isinstance(est_adapt, float)
    assert _rel_err(est_adapt, est_fixed) < 1e-8


def test_finite_richardson_vector_output():
    """Tests that finite difference + Richardson works for vector-valued functions."""
    x0 = 0.1
    exact = _vec_sin_cos_derivative(x0)

    d = FiniteDifferenceDerivative(function=_vec_sin_cos, x0=x0)
    est = d.differentiate(
        order=1,
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )

    assert isinstance(est, np.ndarray)
    assert est.shape == (2,)
    assert _rel_err(est, exact) < 1e-6


def test_finite_richardson_levels_must_be_ge_2():
    """Tests that levels < 2 raises ValueError."""
    d = FiniteDifferenceDerivative(function=np.sin, x0=0.0)

    with pytest.raises(ValueError):
        d.differentiate(
            order=1,
            stepsize=1e-2,
            num_points=5,
            extrapolation="richardson",
            levels=1,
        )


def test_finite_richardson_unsupported_combo_raises():
    """Tests that unsupported (num_points, order) combo raises ValueError."""
    d = FiniteDifferenceDerivative(function=np.sin, x0=0.0)

    # 3-point stencil for order 2 is not in _TRUNCATION_ORDER
    with pytest.raises(ValueError):
        d.differentiate(
            order=2,
            stepsize=1e-2,
            num_points=3,
            extrapolation="richardson",
            levels=3,
        )


def _poly3_third_derivative(_x: float = 0.0) -> float:
    """Analytic third derivative of _poly3: constant 6."""
    return 6.0


def test_finite_richardson_third_derivative_poly():
    """Tests that finite-difference + Richardson works for third derivative of cubic."""
    x0 = 0.7
    exact = _poly3_third_derivative(x0)

    d = FiniteDifferenceDerivative(function=_poly3, x0=x0)
    est = d.differentiate(
        order=3,
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )

    assert abs(est - exact) < 1e-6


@pytest.mark.parametrize("num_points,order", [
    (3, 1),
    (5, 1), (5, 2), (5, 3), (5, 4),
    (7, 1), (7, 2),
    (9, 1), (9, 2),
])
def test_finite_richardson_sin_all_supported_combos(num_points, order):
    """Tests that finite difference + Richardson works for all supported (num_points, order)."""
    f = np.sin
    x0 = 0.2
    # analytic derivatives cycle
    if order == 1:
        exact = np.cos(x0)
    elif order == 2:
        exact = -np.sin(x0)
    elif order == 3:
        exact = -np.cos(x0)
    else:
        exact = np.sin(x0)

    d = FiniteDifferenceDerivative(function=f, x0=x0)
    est = d.differentiate(
        order=order,
        stepsize=5e-2,
        num_points=num_points,
        extrapolation="richardson",
        levels=4,
    )

    assert _rel_err(est, exact) < 1e-5


def test_richardson_extrapolate_constant_sequence():
    """Tests that Richardson extrapolation of constant sequence returns the same value."""
    vals = [1.23, 1.23, 1.23]
    out = richardson_extrapolate(vals, p=2, r=2.0)
    assert out == pytest.approx(1.23)


def test_richardson_extrapolate_preserves_shape():
    """Tests that Richardson extrapolation preserves array shapes."""
    a = np.ones((3, 2))
    b = np.ones((3, 2)) * 1.1
    out = richardson_extrapolate([a, b], p=2, r=2.0)
    assert out.shape == (3, 2)


def _square(x: float) -> float:
    """Simple function to square a scalar."""
    return x * x


def test_eval_points_scalar_no_workers():
    """Tests that eval_points works with n_workers=1 for scalar function."""
    xs = np.array([0.0, 1.0, 2.0])
    out = eval_points(_square, xs, n_workers=1)
    assert np.allclose(out, xs**2)


def test_eval_points_scalar_some_workers():
    """Tests that eval_points works with n_workers>1 for scalar function."""
    xs = np.linspace(-1.0, 1.0, 5)
    out = eval_points(_square, xs, n_workers=4)
    assert np.allclose(out, xs**2)


def test_derivative_kit_uses_finite_richardson():
    """Tests that DerivativeKit dispatches to finite difference + Richardson correctly."""
    dk = DerivativeKit(function=np.sin, x0=0.3)
    est = dk.differentiate(
        method="finite",
        order=1,
        stepsize=1e-2,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )
    assert _rel_err(est, np.cos(0.3)) < 1e-6


def _tensor_quadratic(x: np.ndarray) -> float:
    """Scalar test function on tensor input: f(x) = sum(x^2)."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def test_finite_richardson_tensor_input_directional_derivative():
    """Tests Richardson with tensor x0: derivative along scalar step direction."""
    x0 = np.array([[0.1, -0.2],
                   [0.3,  0.4]], dtype=float)

    # FiniteDifferenceDerivative perturbs with a scalar h: x -> x + h.
    # So this is the directional derivative along an all-ones tensor:
    # df/dε|_{ε=0} f(x0 + ε 1) = 2 * sum(x0).
    exact = 2.0 * float(np.sum(x0))

    d = FiniteDifferenceDerivative(function=_tensor_quadratic, x0=x0)
    est = d.differentiate(
        order=1,
        stepsize=1e-4,
        num_points=5,
        extrapolation="richardson",
        levels=4,
    )

    assert np.isscalar(est)
    assert abs(est - exact) < 1e-8
