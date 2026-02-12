.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| Derivatives at multiple points
=======================================

You can evaluate derivatives at multiple expansion points by passing an array
to ``x0``. DerivKit computes the derivative independently at each point and
stacks the results with leading shape ``x0.shape``.

Each expansion point is treated independently. There is no coupling or shared
sampling across points, even for adaptive or polynomial-based backends. This
makes multi-point evaluation a convenience feature rather than a vectorized
solver.


Finite differences (Ridders)
----------------------------

.. doctest:: multi_finite_ridders

   >>> import numpy as np; from derivkit.derivative_kit import DerivativeKit
   >>> # Define the function and multiple expansion points
   >>> f = np.sin
   >>> x0 = np.array([0.1, 0.5, 1.0])
   >>> # Initialize DerivativeKit with array-valued x0
   >>> dk = DerivativeKit(function=f, x0=x0)
   >>> # Derivative at x0 using finite differences with Ridders extrapolation
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders"
   ...     )
   >>> # Result has the same shape as x0
   >>> print(deriv.shape)
   (3,)
   >>> # Compare against analytic derivative cos(x)
   >>> print(bool(np.allclose(deriv, np.cos(x0), atol=1e-6, rtol=0.0)))
   True


Adaptive polynomial fit (Chebyshev)
-----------------------------------

.. doctest:: multi_adaptive

   >>> import numpy as np; from derivkit.derivative_kit import DerivativeKit
   >>> # Same function and expansion points as above
   >>> f = np.sin; x0 = np.array([0.1, 0.5, 1.0])
   >>> dk = DerivativeKit(function=f, x0=x0)
   >>> # Adaptive Chebyshev-based local polynomial fitting at each point
   >>> deriv = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=12,
   ...     spacing="auto"
   ...     )
   >>> # Output is stacked in the order of x0
   >>> print(deriv.shape)
   (3,)
   >>> print(bool(np.allclose(deriv, np.cos(x0), atol=1e-6, rtol=0.0)))
   True


Vector-valued output at multiple points
---------------------------------------

If the function returns a vector, the stacked output has shape
``x0.shape + output_shape``.

.. doctest:: multi_vector_output

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Define a vector-valued function
   >>> def vec_func(x): return np.array([np.sin(x), np.cos(x)])
   >>> # Multiple expansion points
   >>> x0 = np.array([0.2, 0.4, 0.6])
   >>> dk = DerivativeKit(function=vec_func, x0=x0)
   >>> # Finite-difference derivative applied component-wise
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders"
   ...     )
   >>> # Shape is (n_points, output_dim)
   >>> print("shape:", deriv.shape)
   shape: (3, 2)
   >>> # Reference: d/dx [sin(x), cos(x)] = [cos(x), -sin(x)]
   >>> ref = np.stack([np.cos(x0), -np.sin(x0)], axis=-1)
   >>> print(bool(np.allclose(deriv, ref, atol=1e-6, rtol=0.0)))
   True


Notes
-----

- Each entry in ``x0`` is treated as an independent expansion point.
- There is no shared sampling or coupling between points, even for adaptive or
  polynomial-based backends.
- The stacked output always has leading shape ``x0.shape``.
- For vector- or tensor-valued functions, output dimensions appear *after* the
  ``x0`` axis.
- This feature is intended for convenience and clarity, not for exploiting
  correlations or shared structure between expansion points.
