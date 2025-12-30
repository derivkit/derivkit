Adaptive polynomial fit (Chebyshev)
===================================

This backend (``method="adaptive"``) estimates derivatives by fitting a local
polynomial to samples around ``x0`` on a symmetric Chebyshev grid. It is designed
for robustness: stable scaling, optional ridge regularization, and rich
diagnostics when a fit looks suspicious.

Basic usage (defaults)
----------------------

The default grid is Chebyshev-spaced around ``x0`` with automatically chosen
half-width (``spacing="auto"``).

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="adaptive",
       order=1,
   )
   print(val)  # ~ cos(0.7)
   print(np.cos(0.7))  # reference


Control sampling: number of points and spacing
----------------------------------------------

Use ``n_points`` to control the number of samples, and ``spacing`` to set the
grid half-width.

- ``spacing=float`` means absolute half-width ``h`` (samples in ``[x0-h, x0+h]``)
- ``spacing="auto"`` chooses a scale from ``x0`` (with a floor ``base_abs``)
- ``spacing="<pct>%"`` uses a percentage of a local scale

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="adaptive",
       order=1,
       n_points=12,
       spacing=1e-2,
   )
   print(val)


Domain-aware sampling (stay inside bounds)
------------------------------------------

If your function is only valid on a finite interval, pass ``domain=(lo, hi)``.
The default grid is clipped/transformed to stay within bounds.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   f = np.log
   x0 = 0.05
   dk = DerivativeKit(function=f, x0=x0)

   val = dk.differentiate(
       method="adaptive",
       order=1,
       domain=(0.0, None),  # stay in x > 0
       spacing="auto",
       base_abs=1e-3,
   )
   print(val)  # ~ 1/x0
   print(1.0 / x0)  # reference


User-supplied grids (offsets or absolute x)
-------------------------------------------

You can override the sampling points with ``grid=...``:

- ``grid=("offsets", offsets)`` samples at ``x = x0 + offsets`` (0 is inserted if missing)
- ``grid=("absolute", x_values)`` samples directly at given ``x`` values

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   # Explicit offsets around x0
   offsets = np.array([-2e-2, -1e-2, 0.0, 1e-2, 2e-2])
   val = dk.differentiate(
       method="adaptive",
       order=1,
       grid=("offsets", offsets),
   )
   print(val)


Ridge regularization (stabilize ill-conditioned fits)
-----------------------------------------------------

If the Vandermonde system becomes ill-conditioned (e.g. tight spacing, high
degree, or sensitive functions), a small ridge term can improve stability.

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val = dk.differentiate(
       method="adaptive",
       order=2,
       n_points=14,
       spacing=1e-2,
       ridge=1e-10,
   )
   print(val)  # ~ -sin(0.7)
   print(-np.sin(0.7))  # reference


Return an error proxy and diagnostics
------------------------------------

- ``return_error=True`` returns an RMS residual proxy from the polynomial fit.
- ``diagnostics=True`` returns a diagnostics dictionary (samples, scaling, fit metrics).

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   dk = DerivativeKit(function=np.sin, x0=0.7)

   val, err, diag = dk.differentiate(
       method="adaptive",
       order=1,
       return_error=True,
       diagnostics=True,
   )

   print(val, err)
   print("keys:", sorted(diag.keys())[:10])
   print("n_samples:", len(diag["x"]))
   print("degree:", diag["degree"])
   print("rrms:", diag["rrms"])


Noisy function example (where adaptive helps)
---------------------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   rng = np.random.default_rng(0)

   def noisy_sin(x: float) -> float:
       return np.sin(x) + 0.02 * rng.normal()

   dk = DerivativeKit(function=noisy_sin, x0=0.7)

   val, err = dk.differentiate(
       method="adaptive",
       order=1,
       n_points=16,
       spacing="auto",
       return_error=True,
   )
   print(val, err)


Notes
-----

- This backend supports scalar and vector outputs (computed component-wise).
- If you already know your function is smooth and clean, finite differences can be faster.
- If you need explicit trimming/outlier handling, see the ``local_polynomial`` backend.
