"""Provides the CalculusKit class.

The class is essentially a wrapper for the modules in derivkit.calculus.
The user must specify the base function and the central value at which
the derivatives should be evaluated. More details about available options
can be found in the documentation of the methods.

Typical usage examples:

Calculating the gradient of a scalar-valued function:
>>>  import numpy as np
>>>  from derivkit.calculus_kit import CalculusKit
>>>  calc  = CalculusKit(lambda x: x[0] * x[1], np.array([1, 2]))
>>>  grad = calc.gradient()

Calculating the Jacobian of a vector-valued function:
>>>  import numpy as np
>>>  from derivkit.calculus_kit import CalculusKit
>>>  calc  = CalculusKit(lambda x: x, np.array([1, 2]))
>>>  jacobian = calc.jacobian()

Calculating the Hessian of a scalar-valued function:
>>>  import numpy as np
>>>  from derivkit.calculus_kit import CalculusKit
>>>  calc  = CalculusKit(lambda x: x[0] * x[1], np.array([1, 2]))
>>>  hessian = calc.hessian()
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus.gradient import build_gradient
from derivkit.calculus.jacobian import build_jacobian
from derivkit.calculus.hessian import build_hessian


@dataclass
class CalculusKit:
    """Provides access to calculus functions.

    Attributes:
      function: The function to be differentiated.
      x0: The point at which the function must be differentiated.
    """

    function: Callable[[NDArray[np.floating]], np.floating | NDArray[np.floating]]
    x0: NDArray[np.floating]

    def gradient(self, *args, **kwargs) -> NDArray[np.floating]:
        """Wrapper function for build_gradient."""
        return build_gradient(self.function, self.x0, *args, **kwargs)


    def jacobian(self, *args, **kwargs) -> NDArray[np.floating]:
        """Wrapper function for build_jacobian."""
        return build_jacobian(self.function, self.x0, *args, **kwargs)


    def hessian(self, *args, **kwargs) -> NDArray[np.floating]:
        """Wrapper function for build_hessian."""
        return build_hessian(self.function, self.x0, *args, **kwargs)
