"""Implementation of Fornberg's algorithm for numerical derivatives.

The algorithm was publised by Fornberg in:
Bengt Fornberg, *Calculation of Weights in Finite Difference Formulas*,
SIAM Review, vol. 40, No. 3, pp. 685–691, September 1998
"""

from __future__ import annotations
from collections.abc import Callable

import numpy as np

class FornbergDerivative:
    """Supplies the Fornberg derivative.
    
    Attributes:
      func: the function to be differentiated.
      x0: the evaluation point for the derivative.
      grid: a series of points around the evaluation point.
      order: the order of the derivative.
    """

    def __init__(
        self,
        function: Callable,
        x0: np.float64,
        grid: np.array,
        order: int = 1,
    ) -> None:
        """Initialises the class."""
        self.func = function
        self.x0 = x0
        self.grid = grid
        self.order = order

        if self.order < 0:
            raise ValueError(
                "the maximum derivative order must be at least 0 "
                f" (the function itself), but is {self.order}."
            )


    def get_derivative(self, *, n_workers: int = 1) -> np.float64:
        """Constructs the derivative of of ``func`` at ``x0``.
        
        Args:
          n_workers: the number of multiprocessing workers. Not currently used.

        Returns:
            The derivative evaluated at the given point.
        """
        y = self.func(self.grid)
        weights = self.get_weights()[-1]
        result = np.dot(y, weights**self.order)

        return result

    def get_weights(self, n_workers: int = 1) -> np.ndarray:
        """Constructs the weights needed for the derivative.

        Args:
            n_workers: the number of multiprocessing workers. Not currently used.

        Returns:
            A 2D array of numbers representing the derivative weights.
            The row corresponds to the order of the derivative, with the
            0th row corresponding with the function itself.
        """
        c1 = 1.0
        c4 = self.grid[0] - self.x0
        weights = np.zeros((self.grid.size, self.order+1))
        weights[0, 0] = 1.0
        for i in range(1, self.grid.size):
            mn = min(i, self.order)
            c2 = 1.0
            c5 = c4
            c4 = self.grid[i] - self.x0
            for j in range(i):
                c3 = self.grid[i] - self.grid[j]
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
