"""Tests for FiniteDifferenceDerivative validation of inputs and correctness."""

import re

import numpy as np
import pytest

from derivkit.finite.finite_difference import FiniteDifferenceDerivative


def f_poly(power):
    """Returns a polynomial function of given power."""
    return lambda x: x**power

def near(a, b, rtol=1e-8, atol=1e-10):
    """Checks if two floats are close within given tolerances."""
    return np.isclose(a, b, rtol=rtol, atol=atol)

ISSUE_HINT_RE = re.compile(r"issue\s*#?\s*202", flags=re.I)


@pytest.mark.parametrize(
    "num_points, order",
    [
        # currently unsupported by this build. See
        # https://github.com/derivkit/derivkit/issues/202
        (3, 2), (3, 3), (3, 4),
        (7, 3), (7, 4),
        (9, 3), (9, 4),
    ],
)


def test_unsupported_combo_raises_with_issue_hint(num_points, order, tmp_path):
    """Unsupported (num_points, order) combos raise ValueError with hint to issue #202."""
    d = FiniteDifferenceDerivative(function=f_poly(3), x0=0.1, log_file=str(tmp_path / "fd.log"))
    with pytest.raises(ValueError) as ei:
        d.differentiate(order=order, stepsize=1e-3, num_points=num_points)
    msg = str(ei.value)
    # Human-friendly hint is present
    assert "Not implemented yet" in msg
    assert ISSUE_HINT_RE.search(msg), "Error message should reference issue #202"
    # Logged to file too
    log_text = (tmp_path / "fd.log").read_text()
    assert "Not implemented yet" in log_text
    assert ISSUE_HINT_RE.search(log_text)


def test_invalid_stencil_size_raises():
    """Unsupported stencil size raises ValueError."""
    d = FiniteDifferenceDerivative(function=f_poly(2), x0=0.0)
    with pytest.raises(ValueError) as ei:
        d.differentiate(order=1, stepsize=1e-3, num_points=4)  # invalid stencil size
    assert "Unsupported stencil size" in str(ei.value)


def test_invalid_order_raises():
    """Unsupported derivative order raises ValueError."""
    d = FiniteDifferenceDerivative(function=f_poly(2), x0=0.0)
    with pytest.raises(ValueError) as ei:
        d.differentiate(order=0, stepsize=1e-3, num_points=5)  # invalid order
    assert "Unsupported derivative order" in str(ei.value)


@pytest.mark.parametrize(
    "num_points, order, power, x0, h",
    [
        # 3-point: order 1
        (3, 1, 1, 1.234, 1e-4),  # d/dx x = 1

        # 5-point: orders 1..4
        (5, 1, 1, 0.37, 1e-4),   # slope of x is 1
        (5, 2, 2, -0.3, 1e-4),  # d2/dx2 x^2 = 2
        (5, 3, 3, 0.1, 5e-5),  # d3/dx3 x^3 = 6
        (5, 4, 4, 0.3, 1e-3),  # d4/dx4 x^4 = 24

        # 7-point: orders 1..2
        (7, 1, 1, 1.234, 1e-4),  # d/dx x = 1
        (7, 2, 2, 0.5, 1e-4),  # d2/dx2 x^2 = 2

        # 9-point: orders 1..2
        (9, 1, 1, -0.8, 1e-4),  # d/dx x = 1
        (9, 2, 2, -0.5, 1e-4),  # d2/dx2 x^2 = 2
    ],
)


def test_supported_combo_smoke_and_exact_on_polynomials(num_points, order, power, x0, h):
    """Supported (num_points, order) combos compute correct derivative on x^p."""
    d = FiniteDifferenceDerivative(function=f_poly(power), x0=x0)
    est = d.differentiate(order=order, stepsize=h, num_points=num_points)

    # exact derivative for x^p at arbitrary x0:
    if power < order:
        truth = 0.0
    else:
        coeff = 1
        for k in range(power - order + 1, power + 1):
            coeff *= k
        truth = coeff * (x0 ** (power - order))

    # tolerances: tight for low orders, looser for 4th (bigger h is already used)
    if order == 4:
        atol, rtol = 5e-2, 5e-2
    elif order == 3:
        atol, rtol = 5e-6, 5e-6
    else:
        atol, rtol = 1e-8, 1e-8

    assert near(est, truth, rtol=rtol, atol=atol), (
        f"Mismatch (n={num_points}, order={order}, power={power}): est={est}, truth={truth}"
    )
