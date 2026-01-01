Tabulated derivatives
=====================

If you have samples of a 1D function on a grid (i.e. tabulated data without an explicit functional form),
you can compute derivatives directly by passing ``tab_x`` and ``tab_y`` to
:class:`derivkit.derivative_kit.DerivativeKit`. Internally, DerivKit wraps the data in a
:class:`derivkit.tabulated_model.one_d.Tabulated1DModel`, which is then treated as a callable by the
derivative engines.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   # Tabulate y = x^2 on a coarse grid
   x_tab = np.linspace(0.0, 3.0, 7)
   y_tab = x_tab**2

   x0 = 1.5
   dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)

   # First derivative (default method is "adaptive")
   d1 = dk.differentiate(order=1)
   print(d1)

   # Reference: d/dx x^2 = 2x
   print(2.0 * x0)


Finite differences on tabulated data
------------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   x_tab = np.linspace(0.0, 3.0, 21)
   y_tab = np.sin(x_tab)

   x0 = 0.7
   dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)

   d1 = dk.differentiate(
       method="finite",
       order=1,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(d1)
   print(np.cos(x0))


Adaptive fit on tabulated data
------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.derivative_kit import DerivativeKit

   x_tab = np.linspace(0.0, 3.0, 21)
   y_tab = np.sin(x_tab)

   x0 = 0.7
   dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)

   d1 = dk.differentiate(
       method="adaptive",
       order=1,
       n_points=12,
       spacing="auto",
       ridge=1e-10,
   )
   print(d1)
   print(np.cos(x0))


Notes
-----

- ``tab_x`` and ``tab_y`` must be provided together.
- The derivative quality depends on the tabulation density and interpolation
  behavior of the underlying tabulated model.
