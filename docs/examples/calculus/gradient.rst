Gradient
========

:func:`derivkit.calculus.gradient.build_gradient` (and :meth:`derivkit.calculus_kit.CalculusKit.gradient`)
compute the gradient of a scalar-valued function :math:`f(\theta)` with respect to a 1D
parameter vector ``theta`` (shape ``(P,)``).

You can choose the derivative backend via ``method`` and pass any backend-specific
options via ``**dk_kwargs`` (forwarded to :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       # scalar output
       return np.sin(theta[0]) + theta[1] ** 2

   x0 = np.array([0.5, 2.0])
   calc = CalculusKit(func, x0=x0)

   grad = calc.gradient()
   print(grad)  # shape (2,)

   # reference gradient: [cos(x0[0]), 2*x0[1]]
   ref = np.array([np.cos(0.5), 4.0])
   print(ref)


Finite differences (pass FD options via dk_kwargs)
--------------------------------------------------

``method="finite"`` uses the finite-difference backend. Any finite-difference
options are passed through as keyword arguments.

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       return np.sin(theta[0]) + theta[1] ** 2

   calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   g = calc.gradient(
       method="finite",
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(g)


Adaptive polynomial fit (pass adaptive options via dk_kwargs)
-------------------------------------------------------------

``method="adaptive"`` uses the Chebyshev-spaced polynomial-fit backend.

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       return np.sin(theta[0]) + theta[1] ** 2

   calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   grad = calc.gradient(
       method="adaptive",
       n_points=12,
       spacing="auto",
       ridge=1e-10,
   )
   print(grad)


Parallelism across parameters
-----------------------------

``n_workers`` parallelizes *across gradient components* (across parameters).
It does not change the internal derivative engine behavior except for the
engine-level worker count, which is resolved automatically.

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def f(theta):
       return np.sin(theta[0]) + theta[1] ** 2 + np.cos(theta[2])

   calc = CalculusKit(f, x0=np.array([0.5, 2.0, 0.1]))

   g = calc.gradient(
       method="finite",
       n_workers=3,  # parallelize across parameters
       stepsize=1e-2,
       num_points=5,
   )
   print(g)
