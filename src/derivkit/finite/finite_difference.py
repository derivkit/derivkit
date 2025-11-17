"""Provides the FiniteDifferenceDerivative class.

The user must specify the function to differentiate and the central value
at which the derivative should be evaluated. More details about available
options can be found in the documentation of the methods.

Examples:
--------
Basic usage without extrapolation:

>>> from derivkit.finite.finite_difference import FiniteDifferenceDerivative
>>> f = lambda x: x**3
>>> d = FiniteDifferenceDerivative(function=f, x0=2.0)
>>> d.differentiate(order=2)
12.0

First derivative with Ridders extrapolation and an error estimate:

>>> import numpy as np
>>> g = np.sin
>>> d = FiniteDifferenceDerivative(function=g, x0=0.7)
>>> val, err = d.differentiate(
...     order=1,
...     stepsize=1e-2,
...     num_points=5,
...     n_workers=2,
...     extrapolation="ridders",
...     levels=4,
...     return_error=True,
... )
>>> np.allclose(val, np.cos(0.7), rtol=1e-6)
True
"""


from collections.abc import Callable
from functools import partial

import numpy as np

from derivkit.finite.core import single_finite_step
from derivkit.finite.extrapolators import (
    adaptive_gre_fd,
    adaptive_richardson_fd,
    adaptive_ridders_fd,
    fixed_gre_fd,
    fixed_richardson_fd,
    fixed_ridders_fd,
)
from derivkit.finite.stencil import (
    TRUNCATION_ORDER,
    validate_supported_combo,
)


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
            float and return either a float or a 1D array-like object.
        x0: The point at which the derivative is evaluated.

    Supported Stencil and Derivative Combinations
    ---------------------------------------------
    - 3-point: first-order only
    - 5-point: first to fourth-order
    - 7-point: first and second-order
    - 9-point: first and second-order

    Examples:
    ---------
    Basic second derivative without extrapolation:

    >>> f = lambda x: x**3
    >>> d = FiniteDifferenceDerivative(function=f, x0=2.0)
    >>> d.differentiate(order=2)
    12.0

    First derivative with Ridders extrapolation and an error estimate:

    >>> import numpy as np
    >>> g = np.sin
    >>> d = FiniteDifferenceDerivative(function=g, x0=0.7)
    >>> val, err = d.differentiate(
    ...     order=1,
    ...     stepsize=1e-2,
    ...     num_points=5,
    ...     extrapolation="ridders",
    ...     levels=4,
    ...     return_error=True,
    ... )
    >>> np.allclose(val, np.cos(0.7), rtol=1e-6)
    True

    Vector-valued function with Gauss–Richardson extrapolation:

    >>> def vec_func(x):
    ...     return np.array([np.sin(x), np.cos(x)])
    >>> d = FiniteDifferenceDerivative(function=vec_func, x0=0.3)
    >>> val = d.differentiate(
    ...     order=1,
    ...     stepsize=1e-2,
    ...     num_points=5,
    ...     extrapolation="gauss-richardson",
    ...     levels=4,
    ... )
    >>> val.shape
    (2,)
    """

    def __init__(
        self,
        function: Callable,
        x0: float,
    ) -> None:
        """Initialises the class based on function and central value.

        Arguments:
            function: The function to differentiate. Must accept a single
                float and return either a float or a 1D array-like object.
            x0: The point at which the derivative is evaluated.
        """
        self.function = function
        self.x0 = x0

    def differentiate(
        self,
        order: int = 1,
        stepsize: float = 0.01,
        num_points: int = 5,
        n_workers: int = 1,
        extrapolation: str | None = None,
        levels: int | None = None,
        return_error: bool = False,
    ) -> np.ndarray[float] | float:
        """Computes the derivative using a central finite difference scheme.

        Supports 3-, 5-, 7-, or 9-point central difference stencils for
        derivative orders 1 through 4 (depending on the stencil size).
        Derivatives are computed for scalar or vector-valued functions.
        Allows for optional extrapolation (Richardson or Ridders) to improve accuracy.
        It also returns an error estimate if requested.

        Args:
            order: The order of the derivative to compute. Must be supported by
                the chosen stencil size. Default is 1.
            stepsize: Step size (h) used to evaluate the function around the central
                value. Default is 0.01.
            num_points: Number of points in the finite difference stencil. Must be one
                of [3, 5, 7, 9]. Default is 5.
            n_workers: Number of workers to use in multiprocessing. Default is 1
                (no multiprocessing).
            extrapolation: Extrapolation scheme to use for improving accuracy.
                Supported options are:

                * ``None``: no extrapolation (single finite difference).
                * ``"richardson"``:

                  - fixed-level if ``levels`` is not None
                  - adaptive if ``levels`` is None

                * ``"ridders"``:

                  - fixed-level if ``levels`` is not None
                  - adaptive if ``levels`` is None

                * ``"gauss-richardson"`` or ``"gre"``:

                  - fixed-level if ``levels`` is not None
                  - adaptive if ``levels`` is None

            levels: Number of extrapolation levels for fixed schemes. If None,
                the chosen extrapolation method runs in adaptive mode where
                supported.
            return_error: If True, also return an error estimate from the extrapolation
                (or two-step) routine.

        Returns:
            The estimated derivative. Returns a float for scalar-valued
            functions, or a NumPy array for vector-valued functions.

        Raises:
            ValueError:
                If the combination of ``num_points`` and ``order`` is not
                supported or if an unknown extrapolation scheme is given.

        Notes: The available (num_points, order) combinations are:
                - 3: order 1
                - 5: orders 1, 2, 3, 4
                - 7: orders 1, 2
                - 9: orders 1, 2
        """
        if stepsize <= 0:
            raise ValueError("stepsize must be positive.")

        validate_supported_combo(num_points, order)

        # We set up a partial function for single finite difference step
        single = partial(
            single_finite_step,
            self.function,
            self.x0,
        )

        # If we just want bare finite difference (no extrapolation)
        if extrapolation is None:
            value = single(order, stepsize, num_points, n_workers)

            if not return_error:
                return value

            # Our secret second evaluation at h/2 to get a crude error estimate
            r = 2.0
            value_refined = single(order, stepsize / r, num_points, n_workers)

            val_arr = np.asarray(value, dtype=float)
            ref_arr = np.asarray(value_refined, dtype=float)
            err_arr = np.abs(val_arr - ref_arr)

            if np.isscalar(value) or np.shape(value) == ():
                err_out: float | np.ndarray = float(err_arr)
            else:
                err_out = err_arr

            return value, err_out

        # If we wanted extrapolation, get the truncation order first
        key = (num_points, order)
        p = TRUNCATION_ORDER.get(key)
        if p is None:
            raise ValueError(
                f"Extrapolation not configured for stencil {key}."
            )

        # Choose extrapolator function based on scheme + levels
        if extrapolation == "richardson":
            extrap_fn = (fixed_richardson_fd if levels is not None else adaptive_richardson_fd)
        elif extrapolation == "ridders":
            extrap_fn = (fixed_ridders_fd if levels is not None else adaptive_ridders_fd)
        elif extrapolation in {"gauss-richardson", "gre"}:
            extrap_fn = fixed_gre_fd if levels is not None else adaptive_gre_fd
        else:
            raise ValueError(f"Unknown extrapolation scheme: {extrapolation!r}")

        # Common kwargs for all extrapolators
        extrap_kwargs: dict = dict(
            single_finite=single,
            order=order,
            stepsize=stepsize,
            num_points=num_points,
            n_workers=n_workers,
            p=p,
        )
        if levels is not None:
            extrap_kwargs["levels"] = levels

        if return_error:
            extrap_kwargs["return_error"] = True
            value, err = extrap_fn(**extrap_kwargs)
            return value, err

        # If no error requested, just return value of the estimated derivative
        return extrap_fn(**extrap_kwargs)
