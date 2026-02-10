"""Provides all derivkit methods."""

from importlib.metadata import PackageNotFoundError, version

from derivkit.calculus_kit import CalculusKit
from derivkit.derivative_kit import DerivativeKit
from derivkit.forecast_kit import ForecastKit
from derivkit.likelihood_kit import LikelihoodKit

try:
    __version__ = version("derivkit")
except PackageNotFoundError:
    pass

SUPPORTED_KITS = (CalculusKit, DerivativeKit, ForecastKit, LikelihoodKit)
for kit in SUPPORTED_KITS:
    kit.__module__ = "derivkit"


__all__ = [
    "DerivativeKit",
    "ForecastKit",
    "CalculusKit",
    "LikelihoodKit"
]
