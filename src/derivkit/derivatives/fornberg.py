"""Implementation of Fornberg's algorithm for numerical derivatives.

The algorithm was publised by Fornberg in:
Bengt Fornberg, *Calculation of Weights in Finite Difference Formulas*,
SIAM Review, vol. 40, No. 3, pp. 685–691, September 1998

Examples:
=========

Calculating the derivative at a single value::
>>> import numpy as np
>>> from derivkit.derivatives.fornberg import FornbergDerivative
>>> x0 = np.pi/4
>>> grid = np.array([-0.3, -0.25, -0.1, 0, 0.12])
>>> fornberg = FornbergDerivative(lambda x: np.tan(x), x0)
>>> bool(np.isclose(
...     fornberg.differentiate(grid=grid, order=1),
...     2.0022106298738143,
...     rtol=1e-14,
...     atol=0.0,
... ))
True

Calculating the derivative at an array of values using uniform offsets::
>>> import numpy as np
>>> from derivkit.derivatives.fornberg import FornbergDerivative
>>> x0 = np.array([
...     [[1, 2],
...      [3, 4]],
...     [[5, 6],
...      [7,8]]
... ])
>>> grid = np.array([-0.34, -0.02, 0.1, 0.34, 0.98])
>>> fornberg = FornbergDerivative(lambda x: np.cos(x), x0)
>>> np.allclose(
...     fornberg.differentiate(grid=grid, order=1),
...     -np.sin(x0),
...     rtol=1e-4,
...     atol=0.0,
... )
True

Calculating the derivative at an array of values using 5 unique offsets for
each evaluation point::
>>> import numpy as np
>>> from derivkit.derivatives.fornberg import FornbergDerivative
>>> x0 = np.array([2, 7, 10, -np.pi])
>>> grid = np.array([
...     [-0.34, -0.02, 0.1, 0.34, 0.98],
...     [-0.4,  -0.2, -0.1, 0.14, 0.68],
...     [-0.5,  -0.12, 0.15, 0.64, 0.78],
...     [-0.1,   0,    0.06, 0.24, 0.8]
... ]).T
>>> fornberg = FornbergDerivative(lambda x: np.cos(x), x0)
>>> np.allclose(
...     fornberg.differentiate(grid=grid, order=1),
...     -np.sin(x0),
...     rtol=1e-4,
...     atol=1e-4,
... )
True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class FornbergDerivative:
    """Supplies the Fornberg derivative.

    The Fornberg derivative relies on the interpolation of function values
    by Lagrange polynomials. For more information see Bengt Fornberg,
    *Calculation of Weights in Finite Difference Formulas*, SIAM Review,
    vol. 40, No. 3, pp. 685–691, September 1998.

    Attributes:
        function: the function to be differentiated. Must accept a single float
            and return a float.
        x0: the evaluation points for the derivative. Must be a float or a
            structure castable to a Numpy array. If it castable to a Numpy array
            the function and its derivative will be vectorised over the array.
    """

    def __init__(
        self,
        function: Callable,
        x0: np.float64 | NDArray[np.floating],
    ) -> None:
        """Initialises the class.

        Args:
            function: the function to be differentiated. Must accept a single
                float and return a float.
            x0: the evaluation points for the derivative. Must be a float or a
                structure castable to a Numpy array. If it castable to a Numpy
                array the function and its derivative will be vectorised over
                the array.
        """
        self.function = function

        temp_array = np.asarray(x0)
        self.original_shape = temp_array.shape
        self.x0 = np.ravel(temp_array)


    def differentiate(
        self,
        *,
        grid: NDArray[np.float64],
        order: int = 1,
    ) -> NDArray[np.float64]:
        """Constructs the derivative of a given order of a function at a point.

        The derivative is constructed by recursively differentiating the
        Lagrange polynomials that approximate the function around the
        evaluation points.

        Because the derivative of a given order is constructed from the
        derivatives of lower orders (this is part of the recursion) the
        method is capable of returning all derivatives up to the specified
        order. However, currently only the highest derivative is returned.

        See section 3 of (Fornberg 1998) for more details.

        Args:
            grid: an array of offsets relative to the evaluation points. These
                points specify the Lagrange interpolation of the function,
                and are used to calculate the derivatives. The following
                forms are supported:

                * If ``grid`` is a 1D array it is assumed that it contains
                  linear offsets from the evaluation points. These
                  are added uniformly to :data:`FornbergDerivative.x0`.

                * If ``grid`` is an ND array it is assumed that it contains
                  unique linear offsets for each evaluation point. In this
                  case the first axis of the grid must correspond to the
                  offsets while the remaining axes must be equal to the shape
                  of :data:`FornbergDerivative.x0`. In brief, in this case the
                  grid must be of shape ``(n, *x0.shape)`` for some integer
                  ``n``.

                  The advantage of this is that, in this case, the offsets can
                  be tuned for each evaluation point, although each point must
                  still be given the same number of offsets.

            order: the order of the derivative. Must be a non-negative number.
                The case of ``order==0`` corresponds to the Lagrange
                interpolation of the function.

        Returns:
            The derivative of :data:`FornbergDerivative.function` evaluated at
                :data:`FornbergDerivative.x0`.

        Raises:
            ValueError: if ``order`` is smaller than ``0``.
        """
        if order < 0:
            raise ValueError(
                "the maximum derivative order must be at least 0 "
                f" (the function itself), but is {order}."
            )

        if grid.ndim == 1:
            input_grid = self.x0 + grid[:, np.newaxis]
        else:
            input_grid = self.x0 + grid.reshape(grid.shape[0], -1)

        y = self.function(input_grid)
        weights = np.zeros((*input_grid.shape, order+1), dtype=np.float64)

        # Numpy passes around references to the array data so the weights
        # are updated in-place. No assignment is necessary.
        self._get_weights(weights, input_grid, order)
        # np.dot contracts the last axis of its first argument with the
        # second to last index of its second argument. The axes are permuted
        # to ensure that the function values are contracted with the
        # coefficients corresponding with the ``order``-th order derivative.
        # TODO: See if this can be made more transparant.
        derivatives = np.dot(
            y.T,
            np.swapaxes(weights, 0, 1)
        )[np.arange(self.x0.size), np.arange(self.x0.size), -1]

        return derivatives.reshape(self.original_shape)

    def _get_weights(
        self,
        weights: NDArray[np.float64],
        grid: NDArray[np.float64],
        order: int = 1,
    ) -> None:
        """Constructs the weights needed for derivatives up to a given order.

        Args:
            weights: the coefficients for the differentiated Lagrange
                polynomials. The coefficients are updated in place.
            grid: a series of offsets around the evaluation points.
            order: the order of the derivative.
        """
        c1 = 1.0
        c4 = grid[0, :] - self.x0
        weights[0, ..., 0] = 1.0

        for i in range(1, grid.shape[0]):
            mn = min(i, order)
            c2 = 1.0
            c5 = c4
            c4 = grid[i, :] - self.x0

            for j in range(i):
                c3 = grid[i, ...] - grid[j, ...]
                c2 *= c3

                if i == j + 1:
                    for k in range(mn, 0, -1):
                        weights[i, ..., k] = c1 * (
                            k * weights[i-1, ..., k-1]
                            - c5 * weights[i-1, ..., k]
                        ) / c2
                    weights[i, ..., 0] = -c1 * c5 * weights[i-1, ..., 0] / c2

                for k in range(mn, 0, -1):
                    weights[j, ..., k] = (c4 * weights[j, ..., k] - k * weights[j, ..., k-1]) / c3
                weights[j, ..., 0] = c4 * weights[j, ..., 0] / c3

            c1 = c2
