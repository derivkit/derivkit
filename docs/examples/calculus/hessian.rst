Hessian
=======

:func:`derivkit.calculus.hessian.build_hessian` (and :meth:`derivkit.calculus_kit.CalculusKit.hessian`)
compute the Hessian of a function :math:`f(\theta)`.

If ``theta`` has shape ``(p,)``, with ``p`` being the number of parameters, the Hessian has shape:

- scalar output ``f(theta)`` → Hessian shape ``(p, p)``
- tensor output ``f(theta)`` with shape ``out_shape`` → Hessian shape ``(*out_shape, p, p)``

You can choose the derivative backend via ``method`` and pass backend-specific
options via ``**dk_kwargs`` (forwarded to :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).

Basic usage (scalar-valued function)
------------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       # scalar output
       return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2

   x0 = np.array([0.5, 2.0])
   calc = CalculusKit(func, x0=x0)

   hess = calc.hessian()
   print(hess)
   print(hess.shape)  # (2, 2)

   ref = np.array([
       [-np.sin(0.5), 1.0],
       [1.0, 2.0],
   ])
   print(ref)


Hessian diagonal only
---------------------

For large parameter dimensions you may only need the diagonal:

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2

   calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   hess_diag = calc.hessian_diag()
   print(hess_diag)  # shape (2,)


Tensor-valued outputs
---------------------

If your function returns a tensor, the Hessian is computed per component and the
result is reshaped back to ``(*out_shape, p, p)``.

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def func(theta):
       # output shape (2,) so the Hessian shape is (2, p, p)
       return np.array([
           np.sin(theta[0]),
           theta[0] * theta[1] + theta[1] ** 2,
       ])

   x0 = np.array([0.5, 2.0])
   calc = CalculusKit(func, x0=x0)

   hess = calc.hessian()
   print(hess.shape)  # (2, 2, 2)

   hess0_ref = np.array([
       [-np.sin(0.5), 0.0],
       [0.0, 0.0],
   ])

   hess1_ref = np.array([
       [0.0, 1.0],
       [1.0, 2.0],
   ])

   print(hess0_ref)
   print(hess1_ref)


Choosing a backend + parallelism
--------------------------------

``n_workers`` parallelizes across Hessian tasks (entries / components).

.. code-block:: python

   import numpy as np
   from derivkit.calculus_kit import CalculusKit

   def f(theta):
       return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2

   calc = CalculusKit(f, x0=np.array([0.5, 2.0]))

   hess = calc.hessian(
       method="finite",
       n_workers=4,   # parallelize Hessian tasks
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(hess)


Notes
-----

- For scalar outputs, the full Hessian is computed by evaluating only the upper
  triangle and mirroring.
- For tensor outputs, each component is treated as a scalar function internally
  (output is flattened, differentiated, and reshaped back).
- When using :meth:`CalculusKit.hessian_diag`, mixed partials are skipped for speed.
