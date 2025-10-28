"""Provides the CalculusKit class.

A light wrapper around the calculus helpers that exposes a simple API
for gradient, Jacobian, and Hessian computations.

Typical usage examples:

>>> import numpy as np
>>> from derivkit.calculus_kit import CalculusKit  # noqa: F401
>>>
>>> def sin_function(x):
...     # scalar-valued function: f(θ) = sin(θ0)
...     return np.sin(x[0])
>>>
>>> def identity_function(x):
...     # vector-valued function: f(θ) = θ
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

from .calculus import build_gradient, build_hessian, build_jacobian


class CalculusKit:
    """Provides access to gradient, Jacobian, and Hessian tensors."""

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], float | NDArray[np.floating]],
        x0: Sequence[float] | np.ndarray,
    ):
        """Initialise with function and expansion point.

        Args:
            function: Maps parameters -> observable(s). Accepts a 1D array-like of length P.
                      Returns either a scalar (for gradient/Hessian of scalar f)
                      or a 1D array (for Jacobian).
            x0: Point at which to evaluate derivatives (shape (P,)).
        """
        self.function = function
        self.x0 = np.asarray(x0, dtype=float)

    def gradient(self, *, n_workers: int = 1) -> NDArray[np.floating]:
        """Returns the gradient of a scalar-valued function."""
        return build_gradient(self.function, self.x0, n_workers=n_workers)

    def jacobian(self, *, n_workers: int = 1) -> NDArray[np.floating]:
        """Returns the Jacobian of a vector-valued function."""
        return build_jacobian(self.function, self.x0, n_workers=n_workers)

    def hessian(self, *, n_workers: int = 1) -> NDArray[np.floating]:
        """Returns the Hessian of a scalar-valued function."""
        return build_hessian(self.function, self.x0, n_workers=n_workers)
