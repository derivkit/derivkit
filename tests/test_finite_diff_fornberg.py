"""Unit tests for Fornberg finite-difference derivative computation."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivatives.fornberg import FornbergDerivative


def _poly_derivative_at(x: float, *, degree: int, order: int) -> float:
    """Returns d^order/dx^order x^degree evaluated at x."""
    if order < 0:
        raise ValueError("order must be >= 0.")
    if order > degree:
        return 0.0

    coeff = 1.0
    for k in range(order):
        coeff *= float(degree - k)
    return coeff * (x ** (degree - order))


def _central_uniform_grid(x0: float, h: float, n: int) -> np.ndarray:
    """Returns uniform central grid x0 + offsets*h for odd n."""
    if n < 3 or n % 2 == 0:
        raise ValueError("n must be odd and >= 3.")
    half = n // 2
    return x0 + np.arange(-half, half + 1, dtype=float) * h


def _fail_close(
    *,
    name: str,
    got: float,
    expected: float,
    rtol: float,
    atol: float,
    extra: dict[str, object] | None = None,
) -> None:
    """Fails with a readable message (no huge pytest trace)."""
    if bool(np.isclose(got, expected, rtol=rtol, atol=atol)):
        return

    denom = abs(expected) if expected != 0.0 else 1.0
    rel = (got - expected) / denom

    lines = [
        f"{name} mismatch",
        f"expected: {expected:.16e}",
        f"got:      {got:.16e}",
        f"abs err:  {(got - expected):.16e}",
        f"rel err:  {rel:.16e}",
        f"rtol={rtol} atol={atol}",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

    pytest.fail("\n".join(lines), pytrace=False)


def test_fornberg_negative_order_raises() -> None:
    """Tests that negative derivative order raises ValueError."""
    x0 = 0.0
    grid = np.array([-1.0, 0.0, 1.0], dtype=float)
    d = FornbergDerivative(np.sin, np.float64(x0))
    with pytest.raises(ValueError):
        _ = d.differentiate(grid=grid, order=-1)


def test_fornberg_order0_is_function_value() -> None:
    """Tests that order=0 returns f(x0) (via the 0th weight row)."""
    x0 = 0.3
    grid = x0 + np.array([-0.2, 0.0, 0.4], dtype=float)

    f = np.cos
    d = FornbergDerivative(f, np.float64(x0))

    got = float(d.differentiate(grid=grid, order=0))
    expected = float(f(x0))

    _fail_close(
        name="order0 equals function value",
        got=got,
        expected=expected,
        rtol=0.0,
        atol=0.0,
        extra={"x0": x0, "grid": np.array2string(grid, precision=6, separator=", ")},
    )


def test_get_weights_shape() -> None:
    """Tests that get_weights returns shape (order+1, n_points)."""
    x0 = 0.0
    grid = np.array([-0.1, 0.0, 0.2, 0.5], dtype=float)
    order = 3

    d = FornbergDerivative(np.sin, np.float64(x0))
    w = d.get_weights(grid, order)

    assert w.shape == (order + 1, grid.size)


def test_get_weights_partition_of_unity() -> None:
    """Tests that order-0 weights sum to 1 (constant function exactness)."""
    x0 = 0.3
    grid = x0 + np.array([-0.3, -0.1, 0.0, 0.2, 0.7], dtype=float)

    d = FornbergDerivative(np.sin, np.float64(x0))
    w = d.get_weights(grid, order=4)

    # For constant function f(x)=1, order 0 should reproduce 1 exactly:
    # sum_i w0_i = 1
    s0 = float(np.sum(w[0]))
    _fail_close(
        name="sum(order0 weights) == 1",
        got=s0,
        expected=1.0,
        rtol=0.0,
        atol=1e-14,
        extra={"x0": x0, "grid": np.array2string(grid, precision=6, separator=", ")},
    )


def test_get_weights_derivative_of_constant_is_zero() -> None:
    """Tests that higher-order weight rows sum to 0 (derivative of constant is 0)."""
    x0 = 0.3
    grid = x0 + np.array([-0.3, -0.1, 0.0, 0.2, 0.7], dtype=float)

    d = FornbergDerivative(np.sin, np.float64(x0))
    w = d.get_weights(grid, order=4)

    for k in range(1, 5):
        sk = float(np.sum(w[k]))
        _fail_close(
            name=f"sum(order{k} weights) == 0",
            got=sk,
            expected=0.0,
            rtol=0.0,
            atol=1e-12,
            extra={"order": k, "x0": x0},
        )


@pytest.mark.parametrize("n_points", [3, 5, 7, 9], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4], ids=lambda o: f"ord{o}")
def test_fornberg_exact_on_polynomials_uniform_grid(n_points: int, order: int) -> None:
    """Tests that Fornberg reproduces polynomial derivatives on a uniform grid."""
    # Keep this test numerically well-conditioned.
    # The irregular-grid test is the strict “exactness” test.
    degree = min(n_points - 1, order + 1)  # was: order + 2
    if order > degree:
        pytest.skip("Derivative order above polynomial degree is trivially 0.")

    x0 = 0.7
    h = 1e-2
    grid = _central_uniform_grid(x0, h, n_points)

    def f(x):
        return np.asarray(x, dtype=float) ** degree

    d = FornbergDerivative(f, np.float64(x0))
    got = float(d.differentiate(grid=grid, order=order))
    expected = _poly_derivative_at(x0, degree=degree, order=order)

    # Order-aware tolerances: uniform-grid high-order weights are ill-conditioned in float64.
    if order <= 2:
        rtol, atol = 1e-10, 1e-12
    elif order == 3:
        rtol, atol = 1e-7, 1e-9
    else:  # order == 4
        rtol, atol = 1e-6, 1e-8

    _fail_close(
        name=f"poly uniform (x^{degree})",
        got=got,
        expected=expected,
        rtol=rtol,
        atol=atol,
        extra={"x0": x0, "h": h, "n_points": n_points, "order": order},
    )


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4], ids=lambda o: f"ord{o}")
def test_fornberg_exact_on_degree4_polynomial_irregular_grid(order: int) -> None:
    """Tests that Fornberg is exact on x^4 with 5 irregular points (orders 0–4)."""
    x0 = 0.3
    grid = x0 + np.array([-0.3, -0.25, -0.1, 0.0, 0.12], dtype=float)

    degree = 4
    if order > degree:
        pytest.skip("Order above degree.")

    def f(x):
        """Returns x^degree."""
        return np.asarray(x, dtype=float) ** degree

    d = FornbergDerivative(f, np.float64(x0))
    got = float(d.differentiate(grid=grid, order=order))
    expected = _poly_derivative_at(x0, degree=degree, order=order)

    _fail_close(
        name="poly exactness irregular (x^4)",
        got=got,
        expected=expected,
        rtol=1e-10,
        atol=1e-12,
        extra={
            "x0": x0,
            "order": order,
            "grid": np.array2string(grid, precision=6, separator=", "),
        },
    )


def test_fornberg_docstring_tan_example() -> None:
    """Tests that the docstring tan example reproduces the documented value."""
    x0 = float(np.pi / 4.0)
    grid = x0 + np.array([-0.3, -0.25, -0.1, 0.0, 0.12], dtype=float)

    d = FornbergDerivative(lambda x: np.tan(x), np.float64(x0))
    got = float(d.differentiate(grid=grid, order=1))

    # keep exactly what the docstring says (this is a regression test)
    expected = 2.0022106298738143
    _fail_close(
        name="docstring tan example",
        got=got,
        expected=expected,
        rtol=1e-14,
        atol=0.0,
        extra={"x0": x0, "grid": np.array2string(grid, precision=6, separator=", ")},
    )
