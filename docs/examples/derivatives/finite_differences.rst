Finite Differences
==================

This page shows minimal, task-oriented examples using the finite-difference
engine. Use this when your function is smooth and reasonably cheap to evaluate.

Plain finite difference (no extrapolation)
------------------------------------------

A single central-difference stencil evaluation.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   f = np.sin
   dk = DerivativeKit(function=f, x0=0.7)

   # 5-point stencil, first derivative
   val = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
   )
   print(val)  # ~ cos(0.7)
   print(np.cos(0.7))  # reference



Plain finite difference + crude error estimate
----------------------------------------------

If you set ``return_error=True`` with ``extrapolation=None``, the engine does
a second evaluation at ``h/2`` and returns ``|D(h) - D(h/2)|`` as a simple
error estimate.

.. code-block:: python

   import numpy as np
   from derivkit.finite.finite_difference import FiniteDifferenceDerivative

   d = FiniteDifferenceDerivative(function=np.sin, x0=0.7)

   val, err = d.differentiate(
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation=None,
       return_error=True,
   )
   print(val, err)


Richardson extrapolation (fixed levels)
---------------------------------------

Richardson uses a known truncation order to combine estimates at smaller step
sizes. Use ``levels`` for a fixed number of extrapolation steps.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="richardson",
       levels=4,
   )
   print(val)


Richardson extrapolation (adaptive)
-----------------------------------

If ``levels=None`` (default), Richardson runs in adaptive mode.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="richardson",
   )
   print(val)



Ridders extrapolation + error estimate
--------------------------------------

Ridders is similar in spirit, but includes an internal error estimate. This is
often a good default when you want automatic error control.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val, err = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
       return_error=True,
   )
   print(val, err)
   print(np.cos(0.7))  # reference


Vector-valued function + GRE extrapolation
------------------------------------------

Finite differences support vector outputs. GRE (Gauss–Richardson) is intended
to be more noise-robust.

.. code-block:: python

   import numpy as np
   from derivkit.finite.finite_difference import FiniteDifferenceDerivative

   def vec_func(x):
       return np.array([np.sin(x), np.cos(x)])

   d = FiniteDifferenceDerivative(function=vec_func, x0=0.3)

   val = d.differentiate(
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="gre",
       levels=4,
   )
   print(val)  # shape (2,)


Notes
-----

- Supported stencils: ``num_points in {3, 5, 7, 9}``.
- Supported derivative orders depend on the stencil size (see API docs for details).
- When in doubt, start with ``num_points=5`` and ``extrapolation="ridders"``.
- You can evaluate derivatives at multiple expansion points by passing an array to ``x0``.
  The derivative is computed independently at each point and the results are stacked with
  leading shape ``x0.shape``.

