"""Unit tests for adaptive.transforms."""

import math

import numpy as np
import pytest

from derivkit.adaptive.transforms import (
    pullback_sqrt_at_zero,
    signed_log_derivatives_to_x,
    signed_log_forward,
    signed_log_to_physical,
    sqrt_domain_forward,
    sqrt_to_physical,
)


@pytest.mark.parametrize("x0, expected_sgn", [(2.5, 1.0), (-3.0, -1.0)])
def test_signed_log_forward_basic(x0, expected_sgn):
    """Maps x to (log(abs(x)), sign) for valid finite nonzero inputs."""
    q0, sgn = signed_log_forward(x0)
    assert pytest.approx(q0) == math.log(abs(x0))
    assert sgn == expected_sgn


@pytest.mark.parametrize("bad_x0", [0.0, float("nan"), float("inf"), -float("inf")])
def test_signed_log_forward_errors(bad_x0):
    """Raises ValueError on zero or non-finite inputs."""
    with pytest.raises(ValueError):
        signed_log_forward(bad_x0)


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_signed_log_to_physical_roundtrip(sgn):
    """Round-trips q to x with fixed sign and matches sgn*exp(q)."""
    # q -> x = sgn * exp(q)
    q = np.array([-2.0, 0.0, 1.5])
    x = signed_log_to_physical(q, sgn)
    np.testing.assert_allclose(x, sgn * np.exp(q), rtol=0, atol=0)

    # forward then back (except sign, which is fixed)
    # pick a nonzero x0 compatible with sign
    x0 = (1.7 if sgn > 0 else -2.3)
    q0, s = signed_log_forward(x0)
    assert s == sgn
    x_back = signed_log_to_physical(np.array([q0]), s)
    np.testing.assert_allclose(x_back, np.array([x0]))


def test_signed_log_to_physical_nonfinite_q_raises():
    """Raises ValueError when q contains non-finite values."""
    with pytest.raises(ValueError):
        signed_log_to_physical(np.array([0.0, np.nan]), 1.0)


def test_pullback_signed_log_order1_and_2():
    """Applies pullback through signed-log for first and second derivatives."""
    x0 = 0.7
    dfdq = np.array([2.0, -1.0])
    d2fdq2 = np.array([5.0, 3.0])

    # order 1: d/dx = (df/dq) * dq/dx, with q=log|x| => dq/dx = 1/x
    d1 = signed_log_derivatives_to_x(1, x0, dfdq)
    np.testing.assert_allclose(d1, dfdq / x0)

    # order 2: given implementation (d2fdq2 - dfdq) / x^2
    d2 = signed_log_derivatives_to_x(2, x0, dfdq, d2fdq2)
    np.testing.assert_allclose(d2, (d2fdq2 - dfdq) / (x0 ** 2))

    # order 2 missing d2fdq2 -> error
    with pytest.raises(ValueError):
        signed_log_derivatives_to_x(2, x0, dfdq, None)

    # x0 == 0 -> error
    with pytest.raises(ValueError):
        signed_log_derivatives_to_x(1, 0.0, dfdq)


@pytest.mark.parametrize(
    "x0, sign, expected_u, expected_s",
    [
        (4.0, None, 2.0, 1.0),
        (-9.0, None, 3.0, -1.0),
        (0.0, +1.0, 0.0, 1.0),
        (0.0, -1.0, 0.0, -1.0),
    ],
)
def test_sqrt_domain_forward_basic(x0, sign, expected_u, expected_s):
    """Maps x to (u, sign) with u≥0 for the sqrt-domain reparameterization."""
    u0, s = sqrt_domain_forward(x0, sign)
    assert s == expected_s
    assert pytest.approx(u0) == expected_u


def test_sqrt_domain_forward_errors():
    """Raises ValueError on ambiguous or inconsistent sqrt-domain inputs."""
    # x0 == 0 and sign None -> ambiguous
    with pytest.raises(ValueError):
        sqrt_domain_forward(0.0, None)
    # inconsistent sign with nonzero x0
    with pytest.raises(ValueError):
        sqrt_domain_forward(1.0, -1.0)
    with pytest.raises(ValueError):
        sqrt_domain_forward(-1.0, +1.0)
    # non-finite x0
    with pytest.raises(ValueError):
        sqrt_domain_forward(float("nan"), 1.0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sqrt_to_physical_mapping(sign):
    """Maps u back to x via x = sign * u^2."""
    u = np.array([0.0, 1.0, 2.0, 3.0])
    x = sqrt_to_physical(u, sign)
    np.testing.assert_allclose(x, sign * (u ** 2))


def test_sqrt_to_physical_nonfinite_u_raises():
    """Raises ValueError when u contains non-finite values."""
    with pytest.raises(ValueError):
        sqrt_to_physical(np.array([1.0, np.inf]), 1.0)


def test_pullback_sqrt_at_zero_orders_and_errors():
    """Computes pullbacks at x=0 for sqrt map and validates required inputs."""
    # order 1: g''(0) / (2*s)
    for s in (1.0, -1.0):
        g2 = np.array([2.0, -4.0])
        d1 = pullback_sqrt_at_zero(1, s, g2=g2)
        np.testing.assert_allclose(d1, g2 / (2.0 * s))

    # order 2: g''''(0) / (12*s^2)
    for s in (1.0, -1.0):
        g4 = np.array([12.0, -24.0])
        d2 = pullback_sqrt_at_zero(2, s, g4=g4)
        np.testing.assert_allclose(d2, g4 / (12.0 * s * s))

    # missing g2/g4 errors
    with pytest.raises(ValueError):
        pullback_sqrt_at_zero(1, 1.0, g2=None)
    with pytest.raises(ValueError):
        pullback_sqrt_at_zero(2, 1.0, g4=None)
