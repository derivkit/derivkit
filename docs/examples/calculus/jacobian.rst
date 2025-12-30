Jacobian
========

:func:`derivkit.calculus.jacobian.build_jacobian` (and :meth:`derivkit.calculus_kit.CalculusKit.jacobian`)
compute the Jacobian of a vector-valued function :math:`\mathbf{f}(\theta)`.

If ``f(theta)`` returns shape ``(M,)`` and ``theta`` has shape ``(P,)``, the
Jacobian has shape ``(M, P)``, where each column is the derivative with respect
to one parameter.

You can choose the derivative backend via ``method`` and pass backend-specific
options via ``**dk_kwargs`` (forwarded to :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       # vector output: shape (2,)
       return np.array([
           np.sin(theta[0]) + theta[1],
           theta[0] * theta[1],
       ])

   x0 = np.array([0.5, 2.0])
   calc = CalculusKit(func, x0=x0)

   jac = calc.jacobian()
   print(jac)
   print(jac.shape)  # (2, 2)

   # reference Jacobian:
   # df0/dt0 = cos(t0), df0/dt1 = 1
   # df1/dt0 = t1, df1/dt1 = t0
   ref = np.array([
       [np.cos(0.5), 1.0],
       [2.0, 0.5],
   ])
   print(ref)


Finite differences (Ridders) via dk_kwargs
-----------------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       return np.array([np.sin(theta[0]) + theta[1], theta[0] * theta[1]])

   calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   jac = calc.jacobian(
       method="finite",
       n_workers=2,  # parallelize across parameters (columns)
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(jac)


Adaptive backend via dk_kwargs
------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       return np.array([np.sin(theta[0]) + theta[1], theta[0] * theta[1]])

   calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   jac = calc.jacobian(
       method="adaptive",
       n_workers=2,  # parallelize across parameters (columns)
       n_points=12,
       spacing="auto",
       ridge=1e-10,
   )
   print(jac)


Notes
-----

- ``n_workers`` parallelizes across parameters (Jacobian columns).
- The function must return a **1D** vector (shape ``(M,)``). If it returns a scalar
  or higher-rank tensor, ``build_jacobian`` raises ``TypeError``.
