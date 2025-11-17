"""Tests for derivkit.utils.numerics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.numerics import (
    central_difference_error_estimate,
    relative_error,
)


@pytest.mark.parametrize(
    "order, expected",
    [
        (1, 1 / 6),
        (2, 1 / 12),
        (3, 1 / 20),
        (4, 1 / 30),
    ],
)
def test_central_difference_error_estimate_matches_old(order, expected):
    """Tests that central_difference_error_estimate matches expected values."""
    h = 0.1
    assert_allclose(
        central_difference_error_estimate(h, order),
        expected * h**2,
    )


@pytest.mark.parametrize("bad_order", [0, -1])
def test_central_difference_error_estimate_raises_on_nonpositive_order(bad_order):
    """Tests that non-positive derivative orders should raise a ValueError."""
    with pytest.raises(ValueError):
        central_difference_error_estimate(0.1, bad_order)


@pytest.mark.parametrize("high_order", [5, 999])
def test_central_difference_error_estimate_warns_on_orders_above_supported_range(high_order):
    """Tests that orders > 4 emit a warning but still scale like h^2."""
    h1 = 0.1
    h2 = 0.05  # half of h1

    with pytest.warns(UserWarning):
        e1 = central_difference_error_estimate(h1, high_order)
    with pytest.warns(UserWarning):
        e2 = central_difference_error_estimate(h2, high_order)

    # Both estimates finite and non-negative
    assert np.isfinite(e1)
    assert np.isfinite(e2)
    assert e1 >= 0.0
    assert e2 >= 0.0

    # For fixed order, estimate ∝ h^2
    assert_allclose(e2 / e1, (h2 / h1) ** 2, rtol=1e-12, atol=0.0)

def test_central_difference_error_estimate_scales_like_h_squared_for_supported_order():
    """Tests that error estimate should scale as h^2 for a valid order."""
    order = 2
    h1 = 0.1
    h2 = 0.05

    e1 = central_difference_error_estimate(h1, order)
    e2 = central_difference_error_estimate(h2, order)

    assert_allclose(e2 / e1, (h2 / h1) ** 2, rtol=1e-12, atol=0.0)
    assert e1 >= 0.0
    assert e2 >= 0.0


def test_relative_error_zero_when_equal():
    """Tests that relative_error is zero when inputs are identical."""
    a = np.array([1.0, -2.0, 5.0])
    b = np.array([1.0, -2.0, 5.0])
    assert relative_error(a, b) == 0.0


def test_relative_error_symmetric():
    """Tests that relative_error(a, b) == relative_error(b, a)."""
    a = np.array([1.0, 10.0])
    b = np.array([1.1, 10.0])
    err_ab = relative_error(a, b)
    err_ba = relative_error(b, a)
    assert_allclose(err_ab, err_ba)


def test_relative_error_bounded_and_reasonable():
    """Tests that relative_error produces reasonable bounded results."""
    a = np.array([0.0, 100.0])
    b = np.array([1.0, 100.0])
    err = relative_error(a, b)
    # First component contributes ~1 / max(1,1) = 1, second is 0
    assert 0.9 <= err <= 1.1
