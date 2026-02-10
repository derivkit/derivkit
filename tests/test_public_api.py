"""Unit tests for public API."""

from __future__ import annotations

import derivkit
from derivkit import CalculusKit, DerivativeKit, ForecastKit, LikelihoodKit


def test_kits_importable_from_top_level():
    """Test that all public kits can be imported from top level."""
    # Import happens at module scope; this test just ensures the module loads.
    assert ForecastKit is not None
    assert DerivativeKit is not None
    assert CalculusKit is not None
    assert LikelihoodKit is not None


def test_public_all_contains_kits():
    """Test that __all__ contains the expected public kits."""
    expected = {"CalculusKit", "DerivativeKit", "ForecastKit", "LikelihoodKit"}
    assert expected.issubset(set(derivkit.__all__))


def test_kits_report_top_level_module():
    """Test that kits present as coming from the top-level package."""
    for cls in derivkit.SUPPORTED_KITS:
        assert cls.__module__ == "derivkit"
