"""Unit tests for resolve_spacing step-size logic."""

import numpy as np
import pytest

from derivkit.derivatives.adaptive.spacing import resolve_spacing


def test_numeric_ok():
    """Numeric spacing values are returned as-is if positive."""
    assert resolve_spacing(0.1, 0.0, None) == 0.1


def test_numeric_bad():
    """Non-positive or non-finite numeric spacing raises ValueError."""
    with pytest.raises(ValueError):
        resolve_spacing(0.0, 0.0, None)
    with pytest.raises(ValueError):
        resolve_spacing(float("nan"), 1.0, None)


@pytest.mark.parametrize("x0", [0.0, 1.0, -3.5])
def test_auto_uses_floor_and_scales(x0):
    """Test 'auto' spacing behavior."""
    h = resolve_spacing("auto", x0, None)
    assert np.isfinite(h) and h > 0
    # with floor=1e-3, when x0==0 we should get the floor
    if x0 == 0.0:
        assert h == 1e-3


def test_auto_with_base_abs_overrides_floor():
    """Test 'auto' spacing with custom base_abs."""
    assert resolve_spacing("auto", 0.0, 1e-2) == 1e-2


def test_percent_scales_with_x0_and_uses_floor_when_x0_zero():
    """Test percentage spacing behavior."""
    assert resolve_spacing("2%", 10.0, None) == 0.2
    assert resolve_spacing("2%", 0.0, None) == 1e-3
    assert resolve_spacing("2%", 0.0, 5e-3) == 5e-3


def test_percent_bad():
    """Malformed or non-positive percentage spacing raises ValueError."""
    with pytest.raises(ValueError):
        resolve_spacing("-5%", 1.0, None)
    with pytest.raises(ValueError):
        resolve_spacing("foo%", 1.0, None)
