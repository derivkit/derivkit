"""Provides the DerivativeKit class.

The class is essentially a wrapper for :class:`AdaptiveFitDerivative` and
:class:`FiniteDifferenceDerivative`. The user must specify the function to
differentiate and the central value at which the derivative should be
evaluated. More details about available options can be found in the
documentation of the methods.

Typical usage example:

>>>  derivative = DerivativeKit(function, 1)
>>>  adaptive = derivative.adaptive.differentiate()

derivative is the derivative of function_to_differerentiate at value 1.
"""

from collections.abc import Callable

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.finite.finite_difference import FiniteDifferenceDerivative


class DerivativeKit:
    """Provides access to adaptive and finite difference derivative calculators."""

    def __init__(
        self,
        function: Callable[[float], float],
        x0: float,
    ):
        """Initialises the class based on function and central value.

        Args:
            function: The scalar or vector-valued function to differentiate.
            x0: The point at which the derivative is evaluated.
        """
        self.adaptive = AdaptiveFitDerivative(function, x0)
        self.finite = FiniteDifferenceDerivative(function, x0)

    def get_used_points(self, order: int = 1, n_workers=1):
        """Returns x and y points used in the adaptive fit (for component 0).

        Args:
            order: Order of the derivative to compute diagnostics
                for (default is 1).
            n_workers (int, optional): Number of worker to use in
                multiprocessing. Default is 1 (no multiprocessing).

        Returns:
            A tuple of :class:`np.ndarray`.
        """
        _, diagnostics = self.adaptive.differentiate(
            order=order,
            diagnostics=True,
            n_workers=n_workers,
        )

        x_all = diagnostics["x_all"]
        y_all = diagnostics["y_all"][
            :, 0
        ]  # assuming scalar output or first component
        x_used = diagnostics["x_used"][0]
        y_used = diagnostics["y_used"][0]
        used_mask = diagnostics["used_mask"][0]

        return x_all, y_all, x_used, y_used, used_mask
