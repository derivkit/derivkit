"""Provides the FiniteDifferenceDerivative class.

The user must specify the function to differentiate and the central value
at which the derivative should be evaluated. More details about available
options can be found in the documentation of the methods.

Typical usage example:

>>>  derivative = FiniteDifferenceDerivative(
>>>    function,
>>>    1
>>>  ).differentiate(order=2)

derivative is the second order derivative of function at value 1.
"""

import numpy as np
from multiprocess import Pool


class FiniteDifferenceDerivative:
    """Computes numerical derivatives using central finite difference stencils.

    This class supports the calculation of first to fourth-order derivatives
    for scalar or vector-valued functions. It uses high-accuracy central
    difference formulas with configurable stencil sizes (3-, 5-, 7-, or 9-point).

    For scalar-valued functions, a single float is returned. For vector-valued
    functions, the derivative is computed component-wise and returned as a
    NumPy array.

    Attributes:
        function: callable
            The function to differentiate. Must accept a single float and return either
            a float or a 1D array-like object.
        x0 : float
            The point at which the derivative is evaluated.
        log_file : str, optional
            Path to a file where debug information may be logged.
        debug : bool, optional
            If True, debug information will be printed or logged.

    Supported Stencil and Derivative Combinations
    ---------------------------------------------
    - 3-point: first-order only
    - 5-point: first to fourth-order
    - 7-point: first and second-order
    - 9-point: first and second-order

    Examples:
    ---------
    >>> f = lambda x: x**3
    >>> d = FiniteDifferenceDerivative(function=f, x0=2.0)
    >>> d.differentiate(order=2)
    """

    def __init__(self, function, x0, log_file=None, debug=False):
        """Initialises the class based on function and central value.

        Arguments:
            function (callable): The function to differentiate. Must accept a
                single float and return either a float or a 1D array-like object.
            x0 (float): The point at which the derivative is evaluated.
            log_file (str, optional): Path to a file where debug information may
                be logged.
            debug (bool, optional): If True, debug information will be printed or
                logged.
        """
        self.function = function
        self.x0 = x0
        self.debug = debug
        self.log_file = log_file

    def differentiate(
        self, order=1, stepsize=0.01, num_points=5, n_workers=1
    ):
        """Computes the derivative using a central finite difference scheme.

        Supports 3-, 5-, 7-, or 9-point central difference stencils for
        derivative orders 1 through 4 (depending on the stencil size).
        Derivatives are computed for scalar or vector-valued functions.

        Args:
            order (int, optional): The order of the derivative to
                compute. Must be supported by the chosen stencil size.
                Default is 1.
            stepsize (float, optional): Step size (h) used to evaluate the
                function around the central value. Default is 0.01.
            num_points (int, optional): Number of points in the finite
                difference stencil. Must be one of [3, 5, 7, 9]. Default is 5.
            n_workers (int, optional): Number of worker to use in
                multiprocessing. Default is 1 (no multiprocessing).

        Returns:
            float or np.ndarray: The estimated derivative. Returns a float for
                scalar-valued functions, or a NumPy array for vector-valued
                functions.

        Raises:
            ValueError: If the combination of num_points and order
                is not supported.

        Notes:
            The available (num_points, order) combinations are:
                - 3: order 1
                - 5: orders 1, 2, 3, 4
                - 7: orders 1, 2
                - 9: orders 1, 2
        """
        offsets, coeffs_table = self.get_finite_difference_tables(stepsize)

        if num_points not in offsets:
            raise ValueError(
                f"Unsupported stencil size: {num_points}. Must be one of [3, 5, 7, 9]."
            )

        key = (num_points, order)
        if key not in coeffs_table:
            raise ValueError(
                f"Unsupported combination: stencil={num_points}, order={order}."
            )

        stencil = np.array(
            [self.x0 + i * stepsize for i in offsets[num_points]]
        )

        if n_workers > 1:
            n_workers = np.min((n_workers, len(stencil)))
            with Pool(n_workers) as pool:
                values = np.array(pool.map(self.function, stencil))
        else:
            values = np.array([self.function(x) for x in stencil])
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        derivs = np.dot(values.T, coeffs_table[key])
        # return 1D array for multi-output, Python scalar for single-output
        return derivs.ravel() if derivs.size > 1 else derivs.item()

    def get_finite_difference_tables(self, stepsize):
        """Returns offset patterns and coefficient tables.

        Args:
            stepsize (float): Stepsize for finite difference calculation.

        Returns:
            (dict, dict): A tuple of two dictionaries. The first maps from
                stencil size to symmetric offsets. The second mapps from
                (stencil_size, order) to coefficient arrays.
        """
        offsets = {
            3: [-1, 0, 1],
            5: [-2, -1, 0, 1, 2],
            7: [-3, -2, -1, 0, 1, 2, 3],
            9: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        }

        coeffs_table = {
            (3, 1): np.array([-0.5, 0, 0.5]) / stepsize,
            (5, 1): np.array([1, -8, 0, 8, -1]) / (12 * stepsize),
            (5, 2): np.array([-1, 16, -30, 16, -1]) / (12 * stepsize**2),
            (5, 3): np.array([-1, 2, 0, -2, 1]) / (2 * stepsize**3),
            (5, 4): np.array([1, -4, 6, -4, 1]) / (stepsize**4),
            (7, 1): np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * stepsize),
            (7, 2): np.array([2, -27, 270, -490, 270, -27, 2])
            / (180 * stepsize**2),
            (9, 1): np.array([3, -32, 168, -672, 0, 672, -168, 32, -3])
            / (840 * stepsize),
            (9, 2): np.array(
                [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]
            )
            / (5040 * stepsize**2),
        }

        return offsets, coeffs_table
