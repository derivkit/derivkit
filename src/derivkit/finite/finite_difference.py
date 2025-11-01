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

import sys
from collections.abc import Callable

import numpy as np
from multiprocess import Pool

_SUPPORTED_BY_STENCIL: dict[int, set[int]] = {
    3: {1},
    5: {1, 2, 3, 4},
    7: {1, 2},
    9: {1, 2},
}


class FiniteDifferenceDerivative:
    """Computes numerical derivatives using central finite difference stencils.

    This class supports the calculation of first to fourth-order derivatives
    for scalar or vector-valued functions. It uses high-accuracy central
    difference formulas with configurable stencil sizes (3-, 5-, 7-, or 9-point).

    For scalar-valued functions, a single float is returned. For vector-valued
    functions, the derivative is computed component-wise and returned as a
    NumPy array.

    Attributes:
        function: The function to differentiate. Must accept a single
            float and return eithera float or a 1D array-like object.
        x0: The point at which the derivative is evaluated.
        log_file: Path to a file where debug information may be logged.
        debug: If True, debug information will be printed or logged.

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

    def __init__(self,
        function: Callable,
        x0: float,
        log_file: str = None,
        debug: bool =False
    ) -> None:
        """Initialises the class based on function and central value.

        Arguments:
            function: The function to differentiate. Must accept a single
                float and return either a float or a 1D array-like object.
            x0: The point at which the derivative is evaluated.
            log_file: Path to a file where debug information may
                be logged.
            debug: If True, debug information will be printed or
                logged.
        """
        self.function = function
        self.x0 = x0
        self.debug = debug
        self.log_file = log_file

    def differentiate(self,
                      order: int = 1,
                      stepsize: float = 0.01,
                      num_points: int = 5,
                      n_workers: int = 1
        ) -> np.ndarray:
        """Computes the derivative using a central finite difference scheme.

        Supports 3-, 5-, 7-, or 9-point central difference stencils for
        derivative orders 1 through 4 (depending on the stencil size).
        Derivatives are computed for scalar or vector-valued functions.

        Args:
            order: The order of the derivative to
                compute. Must be supported by the chosen stencil size.
                Default is 1.
            stepsize: Step size (h) used to evaluate the
                function around the central value. Default is 0.01.
            num_points: Number of points in the finite
                difference stencil. Must be one of [3, 5, 7, 9]. Default is 5.
            n_workers: Number of workers to use in
                multiprocessing. Default is 1 (no multiprocessing).

        Returns:
            The estimated derivative. Returns a float for
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
        if stepsize <= 0:
            raise ValueError("stepsize must be positive.")

            # Validate early; prints + raises on unsupported combos
        self._validate_supported_combo(num_points, order)

        offsets, coeffs_table = self.get_finite_difference_tables(stepsize)
        key = (num_points, order)
        if key not in coeffs_table:
            # If table is out of sync with the validator, keep behavior consistent.
            msg = (f"[FiniteDifference] Internal table missing coefficients for "
                   f"stencil={num_points}, order={order}. This combo is not yet implemented "
                   "in this build.")
            self._log(msg)
            raise ValueError(msg)

        stencil = np.array([self.x0 + i * stepsize for i in offsets[num_points]])

        if n_workers > 1:
            n_workers = int(min(n_workers, len(stencil)))
            with Pool(n_workers) as pool:
                values = np.array(pool.map(self.function, stencil))
        else:
            values = np.array([self.function(x) for x in stencil])

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        derivs = values.T @ coeffs_table[key]
        return derivs.ravel() if derivs.size > 1 else derivs.item()

    def get_finite_difference_tables(
            self,
            stepsize: float
        ) -> tuple[dict[int, list[int]], dict[tuple[int, int], np.ndarray]]:
        """Returns offset patterns and coefficient tables.

        Args:
            stepsize: Stepsize for finite difference calculation.

        Returns:
            A tuple of two dictionaries. The first maps from
                stencil size to symmetric offsets. The second maps from
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

    def _log(self, msg: str) -> None:
        """Logs a message to stderr and optionally to a log file.

        Args:
            msg: The message to log.

        Returns:
            None
        """
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(msg.rstrip() + "\n")
            except (OSError, PermissionError) as e:
                print(f"{msg} [log write failed: {e}]", file=sys.stderr)
            else:
                print(msg, file=sys.stderr)

    def _validate_supported_combo(self, num_points: int, order: int) -> None:
        """Validates that the (stencil size, order) combo is supported.

        Args:
            num_points: Number of points in the finite difference stencil.
            order: The order of the derivative to compute.

        Raises:
            ValueError: If the combination of num_points and order is not supported.
        """
        if num_points not in (3, 5, 7, 9):
            msg = (f"[FiniteDifference] Unsupported stencil size: {num_points}. "
                   f"Must be one of [3, 5, 7, 9].")
            self._log(msg)
            raise ValueError(msg)
        if order not in (1, 2, 3, 4):
            msg = (f"[FiniteDifference] Unsupported derivative order: {order}. "
                   f"Must be one of [1, 2, 3, 4].")
            self._log(msg)
            raise ValueError(msg)

        allowed = _SUPPORTED_BY_STENCIL[num_points]
        if order not in allowed:
            # TODO: implement this. See https://github.com/derivkit/derivkit/issues/202
            msg = (
                "[FiniteDifference] Not implemented yet: "
                f"{num_points}-point stencil for order {order}.\n"
                "This is tracked in issue #202 "
                "('Complete finite-difference stencil support: all (3/5/7/9)-point "
                "central stencils for orders ≤ 4').\n"
                "Workarounds: choose a supported combo, e.g.\n"
                "  • 5-point for orders 1–4\n"
                "  • 7/9-point for orders 1–2\n"
                "Or switch methods (e.g., 'adaptive') if available."
            )
            self._log(msg)
            raise ValueError(msg)
