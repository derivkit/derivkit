"""Provides the CalculusKit class.

A wrapper around the calculus helpers that exposes the gradient, Jacobian, and Hessian functions.

Typical usage examples:

>>> import numpy as np
>>> from derivkit.calculus_kit import CalculusKit
>>>
>>> def sin_function(x):
...     # scalar-valued function: f(x) = sin(x[0])
...     return np.sin(x[0])
>>>
>>> def identity_function(x):
...     # vector-valued function: f(x) = x
...     return np.asarray(x, dtype=float)
>>>
>>> calc = CalculusKit(sin_function, x0=np.array([0.5]))
>>> grad = calc.gradient()
>>> hess = calc.hessian()
>>>
>>> jac = CalculusKit(identity_function, x0=np.array([1.0, 2.0])).jacobian()
"""

from collections.abc import Callable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus import build_gradient, build_hessian, build_jacobian
from derivkit.calculus.hessian import build_hessian_diag


class CalculusKit:
    """Provides access to gradient, Jacobian, and Hessian tensors."""

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], float | NDArray[np.floating]],
        x0: Sequence[float] | np.ndarray,
    ):
        """Initialises class with function and expansion point.

        Args:
            function: The function to be differentiated. Accepts a 1D array-like. Must return
                      either a scalar (for gradient/Hessian) or a 1D array (for Jacobian).
            x0: Point at which to evaluate derivatives (shape (P,)).
        """
        self.function = function
        self.x0 = np.asarray(x0, dtype=float)

    def gradient(self, *args, **kwargs) -> NDArray[np.floating]:
        """Returns the gradient of a scalar-valued function."""
        return build_gradient(self.function, self.x0, *args, **kwargs)

    def jacobian(self, *args, **kwargs) -> NDArray[np.floating]:
        """Returns the Jacobian of a vector-valued function."""
        return build_jacobian(self.function, self.x0, *args, **kwargs)

    def hessian(self, *args, **kwargs) -> NDArray[np.floating]:
        """Returns the Hessian of a scalar-valued function."""
        return build_hessian(self.function, self.x0, *args, **kwargs)

    def hessian_diag(self, *args, **kwargs) -> NDArray[np.floating]:
        """Returns the diagonal of the Hessian of a scalar-valued function."""
        return build_hessian_diag(self.function, self.x0, *args, **kwargs)
