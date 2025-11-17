"""Tests for derivkit.utils.extrapolation."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.extrapolation import (
    gauss_richardson_extrapolate,
    richardson_extrapolate,
    ridders_extrapolate,
)


def _make_base_values(true_val: float, h0: float, p: int, n: int, r: float = 2.0):
    """Generates synthetic approximations T + c * h^p."""
    c = 2.0
    return [
        true_val + c * (h0 / (r**j)) ** p
        for j in range(n)
    ]


def test_richardson_extrapolate_scalar_recovers_true_value():
    """Tests that richardson_extrapolate should cancel the leading error term."""
    true_val = 3.14
    p = 2
    h0 = 0.1
    n = 3
    base_values = _make_base_values(true_val, h0, p, n, r=2.0)

    out = richardson_extrapolate(base_values, p=p, r=2.0)

    assert isinstance(out, float)
    assert_allclose(out, true_val, rtol=1e-12, atol=1e-12)


def test_richardson_extrapolate_vector_recovers_true_value_componentwise():
    """Tests that richardson_extrapolate should work on vector inputs."""
    true_vec = np.array([1.0, -2.0, 0.5])
    p = 2
    h0 = 0.05
    n = 4
    c = 1.5

    base_values = [
        true_vec + c * (h0 / (2.0**j)) ** p for j in range(n)
    ]

    out = richardson_extrapolate(base_values, p=p, r=2.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == true_vec.shape
    assert_allclose(out, true_vec, rtol=1e-12, atol=1e-12)


def test_richardson_extrapolate_raises_on_too_few_values():
    """Tests that richardson_extrapolate should require at least two base values."""
    with pytest.raises(ValueError):
        richardson_extrapolate([1.0], p=2, r=2.0)


def test_ridders_extrapolate_scalar_basic():
    """Tests that ridders_extrapolate returns best value close to true and small error."""
    true_val = 0.7
    p = 2
    h0 = 0.1
    n = 4
    base_values = _make_base_values(true_val, h0, p, n, r=2.0)

    best, err = ridders_extrapolate(base_values, r=2.0, p=p)

    assert isinstance(best, float)
    assert isinstance(err, float)
    assert_allclose(best, true_val, rtol=1e-12, atol=1e-12)
    first_err_scale = abs(base_values[0] - true_val)
    assert err < first_err_scale


def test_ridders_extrapolate_vector_shape_and_type():
    """Tests that ridders_extrapolate should preserve vector shape and return scalar error."""
    true_vec = np.array([1.0, 0.0, -1.0])
    p = 2
    h0 = 0.05
    n = 3
    c = 0.8

    base_values = [
        true_vec + c * (h0 / (2.0**j)) ** p for j in range(n)
    ]

    best, err = ridders_extrapolate(base_values, r=2.0, p=p)

    assert isinstance(best, np.ndarray)
    assert best.shape == true_vec.shape
    assert isinstance(err, float)
    assert_allclose(best, true_vec, rtol=1e-12, atol=1e-12)


def test_ridders_extrapolate_raises_on_too_few_values():
    """Tests that ridders_extrapolate should require at least two base values."""
    with pytest.raises(ValueError):
        ridders_extrapolate([1.0])


def _recording_extrapolator(values, *, p, r, calls):
    """Extrapolator that records how it's being called."""
    _ = r  # keep argument for API compatibility, avoid "unused" warning
    calls.append((len(values), p))
    return values[-1]


def test_ridders_extrapolate_uses_custom_extrapolator():
    """Custom extrapolator should be invoked when passed in."""
    calls: list[tuple[int, float]] = []

    base_values = [1.0, 2.0, 3.0, 4.0]

    dummy = partial(_recording_extrapolator, calls=calls)

    best, err = ridders_extrapolate(
        base_values,
        r=2.0,
        p=3,
        extrapolator=dummy,
    )

    assert len(calls) >= 1
    assert all(passed_p == 3 for _, passed_p in calls)
    assert best in base_values
    assert isinstance(err, float)


def test_gauss_richardson_extrapolate_scalar_basic():
    """Tests that Gauss–Richardson should recover a scalar true value to good accuracy."""
    true_val = 1.234
    p = 2
    # a few step sizes, roughly geometric
    h_values = np.array([0.2, 0.1, 0.05, 0.025])
    c = 1.5

    base_values = [true_val + c * (h**p) for h in h_values]

    mean, err = gauss_richardson_extrapolate(
        base_values,
        h_values=h_values,
        p=p,
    )

    assert isinstance(mean, float)
    assert isinstance(err, float)

    # GP-based GRE won't be machine-precision exact, but should be very accurate
    assert_allclose(mean, true_val, rtol=1e-3, atol=1e-5)

    # Error estimate should be smaller than the leading raw error scale
    first_err_scale = abs(base_values[0] - true_val)
    assert err < first_err_scale


def test_gauss_richardson_extrapolate_vector_shape_and_accuracy():
    """Tests that Gauss–Richardson should handle vector inputs and return scalar error."""
    true_vec = np.array([1.0, -2.0, 0.5])
    p = 3
    h_values = np.array([0.2, 0.1, 0.05])
    c_vec = np.array([0.8, -0.4, 1.2])

    base_values = [
        true_vec + c_vec * (h**p) for h in h_values
    ]

    mean, err = gauss_richardson_extrapolate(
        base_values,
        h_values=h_values,
        p=p,
    )

    assert isinstance(mean, np.ndarray)
    assert mean.shape == true_vec.shape
    assert isinstance(err, float)

    # Again, allow modest tolerance since this is a GP-based extrapolation
    assert_allclose(mean, true_vec, rtol=1e-3, atol=1e-5)

    raw_scale = np.max(np.abs(base_values[0] - true_vec))
    assert 0.0 <= err < raw_scale


def test_gauss_richardson_extrapolate_raises_on_mismatched_lengths():
    """Tests that GRE should enforce matching lengths of base_values and h_values."""
    base_values = [1.0, 1.1]
    h_values = [0.1]  # wrong length

    with pytest.raises(ValueError):
        gauss_richardson_extrapolate(base_values, h_values=h_values, p=2)


def test_gauss_richardson_extrapolate_raises_on_nonpositive_h():
    """Tests that GRE should reject non-positive step sizes."""
    base_values = [1.0, 1.1]
    h_values = [0.1, 0.0]  # includes non-positive

    with pytest.raises(ValueError):
        gauss_richardson_extrapolate(base_values, h_values=h_values, p=2)
