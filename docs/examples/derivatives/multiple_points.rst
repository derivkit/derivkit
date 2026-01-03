Derivatives at multiple points
==============================

You can evaluate derivatives at multiple expansion points by passing an array
to ``x0``. DerivKit computes the derivative independently at each point and
stacks the results with leading shape ``x0.shape``.

Each expansion point is treated independently. There is no coupling or shared
sampling across points, even for adaptive or polynomial-based backends.


Finite differences (Ridders)
----------------------------

.. doctest:: multi_finite_ridders

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> f = np.sin
   >>> x0 = np.array([0.1, 0.5, 1.0])

   >>> dk = DerivativeKit(function=f, x0=x0)

   >>> vals = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ... )
   >>> print(vals)
   [0.99500417 0.87758256 0.54030231]
   >>> print(np.cos(x0))  # reference
   [0.99500417 0.87758256 0.54030231]
   >>> print(bool(np.allclose(vals, np.cos(x0), atol=1e-10, rtol=0.0)))
   True


Adaptive polynomial fit (Chebyshev)
-----------------------------------

.. doctest:: multi_adaptive

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> f = np.sin
   >>> x0 = np.array([0.1, 0.5, 1.0])

   >>> dk = DerivativeKit(function=f, x0=x0)

   >>> vals = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=12,
   ...     spacing="auto",
   ... )
   >>> print(vals)
   [0.99500417 0.87758256 0.54030231]
   >>> print(np.cos(x0))  # reference
   [0.99500417 0.87758256 0.54030231]
   >>> print(bool(np.allclose(vals, np.cos(x0), atol=1e-10, rtol=0.0)))
   True


Vector-valued output at multiple points
---------------------------------------

If the function returns a vector, the stacked output has shape
``x0.shape + output_shape``.

.. doctest:: multi_vector_output

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def vec_func(x):
   ...     return np.array([np.sin(x), np.cos(x)])

   >>> x0 = np.array([0.2, 0.4, 0.6])
   >>> dk = DerivativeKit(function=vec_func, x0=x0)

   >>> vals = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ... )
   >>> print("shape:", vals.shape)
   shape: (3, 2)
   >>> ref = np.stack([np.cos(x0), -np.sin(x0)], axis=-1)
   >>> print(bool(np.allclose(vals, ref, atol=1e-10, rtol=0.0)))
   True

Note that the stacking order is always ``(x0_index, ...)``. For vector- or
tensor-valued outputs, the output dimensions appear *after* the ``x0`` axis.
