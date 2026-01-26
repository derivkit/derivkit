"""Implementation of Fornberg's algorithm for numerical derivatives.

The algorithm was publised by Fornberg in:
Bengt Fornberg, *Calculation of Weights in Finite Difference Formulas*,
SIAM Review, vol. 40, No. 3, pp. 685–691, September 1998

Typical usage example:

>>> import numpy as np
>>> from derivkit.derivatives.fornberg import FornbergDerivative
>>> x0 = np.pi/4
>>> grid = x0 + np.array([-0.3, -0.25, -0.1, 0, 0.12])
>>> fornberg = FornbergDerivative(lambda x: np.tan(x), x0)
>>> bool(np.isclose(
...     fornberg.differentiate(grid=grid, order=1),
...     2.0022106298738143,
...     rtol=1e-14,
...     atol=0.0,
... ))
True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class FornbergDerivative:
    """Supplies the Fornberg derivative.

    Attributes:
      function: the function to be differentiated.
      x0: the evaluation point for the derivative.
    """

    def __init__(
        self,
        function: Callable,
        x0: np.float64,
    ) -> None:
        """Initialises the class.

        Args:
          function: the function to be differentiated.
          x0: the evaluation point for the derivative.
        """
        self.function = function
        self.x0 = x0


    def differentiate(
        self,
        *,
        grid: NDArray[np.float64],
        order: int = 1,
    ) -> np.float64:
        """Constructs the derivative of a given order of a function at a point.

        Currently only the derivative of order `order` is returned.
        An array of orders of 0 to `order` can be constructed through
        `get_weights`. See section 3 of (Fornberg 1998) for more details.

        Args:
          grid: a series of points around the evaluation point.
          order: the order of the derivative.

        Returns:
            The derivative evaluated at the given point.
        """
        if order < 0:
            raise ValueError(
                "the maximum derivative order must be at least 0 "
                f" (the function itself), but is {order}."
            )

        y = self.function(grid)
        weights = self.get_weights(grid, order)[-1]
        result = np.dot(y, weights)

        return result

    def get_weights(
        self,
        grid: NDArray[np.float64],
        order: int = 1,
    ) -> np.ndarray:
        """Constructs the weights needed for derivatives up to a given order.

        Args:
            grid: a series of points around the evaluation point.
            order: the order of the derivative.

        Returns:
            A 2D array of numbers representing the derivative weights.
            The 0th axis corresponds to the order of the derivative, with the
            0th row corresponding with the function itself.
        """
        c1 = 1.0
        c4 = grid[0] - self.x0
        weights = np.zeros((grid.size, order+1), dtype=np.float64)
        weights[0, 0] = 1.0

        for i in range(1, grid.size):
            mn = min(i, order)
            c2 = 1.0
            c5 = c4
            c4 = grid[i] - self.x0

            for j in range(i):
                c3 = grid[i] - grid[j]
                c2 *= c3

                if i == j + 1:
                    for k in range(mn, 0, -1):
                        weights[i, k] = c1 * (
                            k * weights[i-1, k-1]
                            - c5 * weights[i-1, k]
                        ) / c2
                    weights[i, 0] = -c1 * c5 * weights[i-1, 0] / c2

                for k in range(mn, 0, -1):
                    weights[j, k] = (c4 * weights[j, k] - k * weights[j, k-1]) / c3
                weights[j, 0] = c4 * weights[j, 0] / c3

            c1 = c2

        return weights.T
