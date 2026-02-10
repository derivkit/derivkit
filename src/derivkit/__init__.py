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

for _cls in (CalculusKit, DerivativeKit, ForecastKit, LikelihoodKit):
    _cls.__module__ = "derivkit"


__all__ = [
    "DerivativeKit",
    "ForecastKit",
    "CalculusKit",
    "LikelihoodKit"
]
