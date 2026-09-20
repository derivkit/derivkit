.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Tabulated derivatives
==============================

This section shows how to compute derivatives using
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate` when your model is
available only as tabulated data on a one-dimensional grid, rather than as an
explicit Python function.

In this case, DerivKit first constructs an interpolated representation of the
tabulated values, and then applies the selected derivative backend to that
interpolant.

This is useful when working with simulation outputs, precomputed theory tables,
or externally generated data products where the underlying function cannot be
evaluated analytically or would be prohibitively expensive to evaluate repeatedly.

For more information on this and other implemented derivative methods,
see :doc:`../../about/kits/derivative_kit`.


Basic usage
-----------

.. doctest:: tabulated_basic

   >>> import numpy as np
   >>> from derivkit import DerivativeKit
   >>> # Tabulate y = x^2 on a coarse grid
   >>> x_tab = np.linspace(0.0, 3.0, 7)
   >>> y_tab = x_tab**2
   >>> x0 = 1.5  # point where to evaluate the derivative
   >>> # Initialize DerivativeKit with tabulated data
   >>> dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)
   >>> # First derivative (default method is "adaptive")
   >>> deriv = dk.differentiate(order=1)
   >>> err = abs(deriv - 2.0 * x0)  # reference: d/dx x^2 = 2x
   >>> bool(np.isfinite(deriv) and (err < 1e-5))
   True


Finite differences on tabulated data
------------------------------------

.. doctest:: tabulated_finite

   >>> import numpy as np
   >>> from derivkit import DerivativeKit
   >>> # Tabulate y = sin(x) on a coarse grid
   >>> x_tab = np.linspace(0.0, 3.0, 21)
   >>> y_tab = np.sin(x_tab)
   >>> x0 = 0.7  # derivative evaluation point
   >>> # Initialize DerivativeKit with tabulated data
   >>> dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)
   >>> # First derivative using finite differences
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> err = abs(deriv - np.cos(x0))  # reference
   >>> bool(np.isfinite(deriv) and (err < 0.1))
   True


Adaptive fit on tabulated data
------------------------------

.. doctest:: tabulated_adaptive

   >>> import numpy as np
   >>> from derivkit import DerivativeKit
   >>> # Tabulate y = sin(x) on a coarse grid
   >>> x_tab = np.linspace(0.0, 3.0, 21)
   >>> y_tab = np.sin(x_tab)
   >>> x0 = 0.7  # point where to evaluate the derivative
   >>> # Initialize DerivativeKit with tabulated data
   >>> dk = DerivativeKit(x0=x0, tab_x=x_tab, tab_y=y_tab)
   >>> # First derivative using adaptive polynomial fit
   >>> deriv = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=12,
   ...     spacing="auto",
   ...     ridge=1e-10,
   ... )
   >>> err = abs(deriv - np.cos(x0))  # reference
   >>> bool(np.isfinite(deriv) and (err < 0.1))
   True


Notes
-----

- ``tab_x`` and ``tab_y`` must be provided together.
- Derivative accuracy depends on the tabulation density and the interpolation
  behavior of the underlying tabulated model.
- As with other backends, you can pass an array to ``x0`` to evaluate derivatives
  at multiple points (stacked with leading shape ``x0.shape``).
