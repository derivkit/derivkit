Local Polynomial Fit
====================

This method estimates derivatives by sampling the function near ``x0``, fitting
a local polynomial, and (optionally) trimming outlier samples before reading off
the derivative from the fitted coefficients.

Use this when finite differences are too sensitive to noise, but you still want
a lightweight method without the full adaptive Chebyshev machinery.

Basic usage (automatic degree)
------------------------------

By default, the degree is chosen as ``max(order + 2, 3)`` (capped by the config).

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="local_polynomial",  # aliases: "local-polynomial", "lp"
       order=1,
   )
   print(val)  # ~ cos(0.7)
   print(np.cos(0.7))  # reference


Specify polynomial degree
-------------------------

You can explicitly choose the polynomial degree (must satisfy ``degree >= order``).

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="lp",
       order=1,
       degree=5,
   )
   print(val)


Return an internal error estimate
---------------------------------

If ``return_error=True``, the method returns a relative error estimate based on
the disagreement between the trimmed and least-squares fits (or an internal LS
uncertainty if trimming fails).

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val, err = dk.differentiate(
       method="local_polynomial",
       order=1,
       degree=5,
       return_error=True,
   )
   print(val, err)


Diagnostics (what was sampled / what got trimmed)
-------------------------------------------------

If ``diagnostics=True``, the method returns a diagnostics dictionary that
includes sampled points, which samples were used, and fit metadata.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val, diag = dk.differentiate(
       method="local_polynomial",
       order=1,
       diagnostics=True,
   )

   print("fit_type:", diag.get("fit_type"))
   print("n_used:", int(np.sum(diag["used_mask"])))
   print("degree:", diag["degree"])
   print("order:", diag["order"])


Noisy function example (why you’d use this)
-------------------------------------------

A simple toy where finite differences can get jumpy as noise increases.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   rng = np.random.default_rng(0)

   def noisy_sin(x: float) -> float:
       return np.sin(x) + 0.01 * rng.normal()

   dk = DerivativeKit(function=noisy_sin, x0=0.7)

   # Local polynomial tends to be more stable than a raw stencil here.
   val = dk.differentiate(
       method="local_polynomial",
       order=1,
       degree=5,
       diagnostics=False,
   )
   print(val)


Notes
-----

- This backend computes derivatives component-wise for vector/tensor outputs.
- ``n_workers`` can be used to parallelize sampling/evaluation.
- For heavy noise or badly conditioned local fits, consider the ``adaptive`` backend.
- You can evaluate derivatives at multiple expansion points by passing an array to ``x0``.
  The derivative is computed independently at each point and the results are stacked with
  leading shape ``x0.shape``.

