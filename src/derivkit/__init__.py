"""Provides all derivkit methods."""

from importlib.metadata import PackageNotFoundError, version

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.calculus_kit import CalculusKit
from derivkit.derivative_kit import DerivativeKit, register_method
from derivkit.finite.finite_difference import FiniteDifferenceDerivative
from derivkit.forecast_kit import ForecastKit
from derivkit.forecasting.expansions import LikelihoodExpansion

try:
    __version__ = version("derivkit")
except PackageNotFoundError:
    pass

DerivativeKit.__module__ = "derivkit.derivative_kit"

__all__ = [
    "AdaptiveFitDerivative",
    "DerivativeKit",
    "FiniteDifferenceDerivative",
    "ForecastKit",
    "LikelihoodExpansion",
    "CalculusKit",
    "register_method",
]
