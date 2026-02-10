"""Unit tests for public API."""

from __future__ import annotations

import derivkit
from derivkit import CalculusKit, DerivativeKit, ForecastKit, LikelihoodKit


def test_public_all_contains_kits():
    """Test that __all__ contains the expected public kits."""
    expected = {"CalculusKit", "DerivativeKit", "ForecastKit", "LikelihoodKit"}
    assert derivkit.SUPPORTED_KITS.issubset(set(derivkit.__all__))


def test_kits_report_top_level_module():
    """Test that kits present as coming from the top-level package."""
    for cls in derivkit.SUPPORTED_KITS:
        assert cls.__module__ == "derivkit"
