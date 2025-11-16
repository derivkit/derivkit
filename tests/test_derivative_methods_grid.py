"""Tests for various derivative estimation methods on polynomial functions."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def poly_function(x: float, power: int) -> float:
    """Return x**power."""
    return x**power


def poly_derivative_value(power: int, order: int, x0: float) -> float:
    """k-th derivative of x^p at x0, evaluated analytically."""
    if power < order:
        return 0.0

    coeff = 1
    for k in range(power - order + 1, power + 1):
        coeff *= k
    return coeff * (x0 ** (power - order))


def near(a: float, b: float, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """Relative+absolute closeness check."""
    return np.isclose(a, b, rtol=rtol, atol=atol)


# method name + kwargs fed straight into DerivativeKit.differentiate
METHOD_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    # finite, no extrapolation
    ("finite",
        {"stepsize": 1.0e-4,
         "num_points": 5,
         "extrapolation": None,},
    ),
    # finite + Richardson
    (
        "finite",
        {
            "stepsize": 1.0e-4,
            "num_points": 5,
            "extrapolation": "richardson",
            "levels": 3,
        },
    ),
    # finite + Ridders
    (
        "finite",
        {
            "stepsize": 1.0e-4,
            "num_points": 5,
            "extrapolation": "ridders",
            "levels": 3,
        },
    ),
    # local polynomial (tune kwargs to whatever LP expects)
    (
        "local_polynomial",
        {
            "degree": 5,
        },
    ),
    # adaptive fit (your default engine)
    (
        "adaptive",
        {
            # usually it has its own defaults; add kwargs if you want
        },
    ),
]


@pytest.mark.parametrize(
    "power, order, x0",
    [
        (3, 1, 0.2),
        (3, 2, 0.2),
        (4, 1, -0.3),
        (4, 2, -0.3),
        (4, 3, 0.7),
    ],
)
@pytest.mark.parametrize("method, method_kwargs", METHOD_CONFIGS)
def test_methods_on_polynomials(
    power: int,
    order: int,
    x0: float,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """All engines should be accurate on x^p test functions."""
    # No nested function, no lambda:
    def_fn = partial(poly_function, power=power)

    dk = DerivativeKit(def_fn, x0)

    est = dk.differentiate(method=method, order=order, **method_kwargs)
    truth = poly_derivative_value(power, order, x0)

    if order == 4:
        atol, rtol = 5e-2, 5e-2
    elif order == 3:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 1e-8, 1e-8

    assert near(est, truth, rtol=rtol, atol=atol), (
        f"method={method}, kwargs={method_kwargs}, "
        f"power={power}, order={order}, x0={x0}, est={est}, truth={truth}"
    )


@pytest.mark.parametrize("x0", [0.1, 0.5, 1.0])
def test_methods_mutually_consistent(x0: float) -> None:
    """All methods should give similar answers on sin."""
    f = np.sin
    order = 1

    dk = DerivativeKit(f, x0)

    estimates: list[tuple[str, float]] = []

    for method, method_kwargs in METHOD_CONFIGS:
        est = dk.differentiate(method=method, order=order, **method_kwargs)
        estimates.append((method, est))

    ref_method, ref_value = estimates[0]
    for method, value in estimates[1:]:
        assert near(value, ref_value, rtol=1e-5, atol=1e-7), (
            f"method {method} disagrees with {ref_method}: "
            f"{value} vs {ref_value}"
        )
