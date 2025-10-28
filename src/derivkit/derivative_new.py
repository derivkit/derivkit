"""Provides the DerivativeAPI class.

The class unifies :class:`AdaptiveFitDerivative` and
:class:`FiniteDifferenceDerivative` under a single interface. The user must
specify the function to differentiate, the central value at which the
derivative should be evaluated, and the desired method. More details about
available options can be found in the documentation of the methods.

Typical usage example:

>>> import numpy as np
>>> from derivative_new import DerivativeProcedure, DerivativeAPI
>>> derivative = DerivativeAPI(function=lambda x: np.cos(x), x0=1)
>>> result = derivative.differentiate(
...     method=DerivativeProcedure.ADAPTIVE,
...     order=1
... )

``result`` is the derivative of ``function`` evaluated at value 1 using the
adaptive-fit method.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable, Literal

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.finite.finite_difference import FiniteDifferenceDerivative


class DerivativeProcedure(Enum):
    ADAPTIVE = auto()
    FINITE = auto()

class DerivativeAPI:
    """Provides a wrapper for DerivKit's derivative procedures.

    Attributes:
        function: The function to be differentiated.
        x0: The central value at which the derivatives are evaluated.
    """


    def __init__(self, function: Callable, x0: float):
        """Initialize the wrapper with a target function and expansion point.

        Args:
          function: Callable of one float returning either a float (scalar observable)
            or a 1D array-like (multi-observable). This is the function whose
            derivatives will be estimated.
          x0: Central value at which derivatives are evaluated.
        """
        self.function = function
        self.x0 = x0

    def differentiate(self, proc: DerivativeProcedure, *args, **kwargs) -> float:
        """Computes the derivative of the function.

        The function must be called with the derivative procedure and any
        additional arguments to the derivative method of the procedure. This
        method passes all arguments except ``proc`` directly to the wrapped
        method without any checking. See the documentation of the wrapped
        procedures for more details.

        Currently supported procedures are listed in ``DerivativeProcedure``.

        Args:
            proc: The derivative procedure that should be used.
            args: Any arguments that should be passed to the wrapped procedure.
            kwargs: Any keyword arguments that should be passed to the wrapped
                procedure.

        Returns:
            The derivative of the function evaluated at the central value.
        """
        match proc:
            case DerivativeProcedure.ADAPTIVE:
                f = AdaptiveFitDerivative(self.function, self.x0)
            case DerivativeProcedure.FINITE:
                f = FiniteDifferenceDerivative(self.function, self.x0)
            case _:
                raise ValueError(f"Unexpected derivative procedure: {proc}")

        return f.differentiate(*args, **kwargs)
