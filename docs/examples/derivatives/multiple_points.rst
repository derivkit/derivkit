Derivatives at multiple points
==============================

You can evaluate derivatives at multiple expansion points by passing an array
to ``x0``. DerivKit computes the derivative independently at each point and
stacks the results with leading shape ``x0.shape``.

Finite differences (Ridders)
----------------------------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   f = np.sin
   x0 = np.array([0.1, 0.5, 1.0])

   dk = DerivativeKit(function=f, x0=x0)

   vals = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
   )

   print(vals)  # shape (3,)
   print(np.cos(x0))  # reference


Adaptive polynomial fit (Chebyshev)
-----------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   f = np.sin
   x0 = np.array([0.1, 0.5, 1.0])

   dk = DerivativeKit(function=f, x0=x0)

   vals = dk.differentiate(
       method="adaptive",
       order=1,
       n_points=12,
       spacing="auto",
   )

   print(vals)  # shape (3,)
   print(np.cos(x0))  # reference


Vector-valued output at multiple points
---------------------------------------

If the function returns a vector, the stacked output has shape
``x0.shape + output_shape``.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   def vec_func(x):
       return np.array([np.sin(x), np.cos(x)])

   x0 = np.array([0.2, 0.4, 0.6])
   dk = DerivativeKit(function=vec_func, x0=x0)

   vals = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
   )

   print(vals.shape)  # (3, 2)
