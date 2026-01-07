Hyper-Hessian
=============

This section shows how to compute the hyper-Hessian (third-derivative tensor)
using DerivKit.

The hyper-Hessian generalizes the Hessian to third order. For a function
``f(theta)``, it contains all third-order partial derivatives evaluated at a
single point in parameter space.

Notation
--------

- ``p`` denotes the number of model parameters (``theta`` has shape ``(p,)``).

Depending on the output type of ``f(theta)``, the hyper-Hessian has the
following shape:

- scalar output ``f(theta)``: hyper-Hessian shape ``(p, p, p)``
- tensor output ``f(theta)`` with shape ``out_shape``: hyper-Hessian shape
  ``(*out_shape, p, p, p)``

The primary low-level interface for computing the hyper-Hessian is
:func:`derivkit.calculus.hyper_hessian.build_hyper_hessian`.

For advanced usage and backend-specific keyword arguments, you can choose the
derivative backend via ``method`` and pass backend-specific options via
``**dk_kwargs`` (forwarded to :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).

See also :doc:`gradient`, :doc:`jacobian`, and :doc:`hessian`.


Basic usage
-----------

.. doctest:: hyper_hessian_basic

   >>> import numpy as np
   >>> from derivkit.calculus.hyper_hessian import build_hyper_hessian
   >>> # Define a scalar-valued function with non-zero third derivatives.
   >>> def func(theta):
   ...     return theta[0] ** 3 + 2.0 * theta[0] * theta[1] + theta[1] ** 2
   >>> theta0 = np.array([0.5, 2.0])
   >>> hess3 = build_hyper_hessian(func, theta0)
   >>> print(hess3.shape)
   (2, 2, 2)
   >>> # Analytic reference:
   >>> # d^3 f / dtheta0^3 = 6, and all other third partials are zero.
   >>> ref = np.zeros((2, 2, 2), dtype=float)
   >>> ref[0, 0, 0] = 6.0
   >>> np.allclose(hess3, ref, atol=1e-6, rtol=0.0)
   True


Symmetry
--------

For smooth functions, the hyper-Hessian is symmetric under permutations of
its parameter indices:

.. doctest:: hyper_hessian_symmetry

   >>> import numpy as np
   >>> from derivkit.calculus.hyper_hessian import build_hyper_hessian
   >>>
   >>> def func(theta):
   ...     return theta[0] ** 3 + 2.0 * theta[0] * theta[1] + theta[1] ** 2
   >>>
   >>> theta0 = np.array([0.5, 2.0])
   >>> hess3 = build_hyper_hessian(func, theta0)
   >>> # Check symmetry for a representative index triplet.
   >>> bool(hess3[0, 1, 0] == hess3[1, 0, 0] == hess3[0, 0, 1])
   True


Tensor-valued outputs
---------------------

If the function returns a tensor, the hyper-Hessian is computed independently
for each output component and afterwards reshaped back to ``(*out_shape, p, p, p)``.

.. doctest:: hyper_hessian_tensor

   >>> import numpy as np
   >>> from derivkit.calculus.hyper_hessian import build_hyper_hessian
   >>> # Two-output function.
   >>> # f0(theta) = theta0^3
   >>> # f1(theta) = theta0*theta1 + theta1^2
   >>> def func(theta):
   ...     return np.array([
   ...         theta[0] ** 3,
   ...         theta[0] * theta[1] + theta[1] ** 2,
   ...     ])
   >>> theta0 = np.array([0.5, 2.0])
   >>> hess3 = build_hyper_hessian(func, theta0)
   >>> print(hess3.shape)
   (2, 2, 2, 2)
   >>>
   >>> # Component 0: only d^3/dtheta0^3 = 6 is non-zero.
   >>> ref0 = np.zeros((2, 2, 2), dtype=float)
   >>> ref0[0, 0, 0] = 6.0
   >>> np.allclose(hess3[0], ref0, atol=1e-6, rtol=0.0)
   True
   >>> # Component 1: all third derivatives are zero (quadratic/linear terms only).
   >>> np.allclose(hess3[1], 0.0, atol=1e-6, rtol=0.0)
   True


Finite differences (Ridders)
----------------------------

You can select a derivative backend using ``method`` and pass backend-specific
options via ``**dk_kwargs``.

.. doctest:: hyper_hessian_finite_ridders

   >>> import numpy as np
   >>> from derivkit.calculus.hyper_hessian import build_hyper_hessian
   >>> def func(theta):
   ...     return theta[0] ** 3 + 2.0 * theta[0] * theta[1] + theta[1] ** 2
   >>> theta0 = np.array([0.5, 2.0])
   >>> hess3 = build_hyper_hessian(
   ...     func,
   ...     theta0,
   ...     method="finite",
   ...     n_workers=4,
   ...     extrapolation="ridders",
   ... )
   >>> ref = np.zeros((2, 2, 2), dtype=float)
   >>> ref[0, 0, 0] = 6.0
   >>> np.allclose(hess3, ref, atol=5e-3, rtol=0.0)
   True


Notes
-----

- For scalar outputs, only unique entries with ``i <= j <= k`` are evaluated and
  the tensor is symmetrized by filling all index permutations.
- For tensor outputs, each component is treated as a scalar internally
  (flattened, differentiated, and reshaped back).
- ``n_workers`` parallelizes work across entries (scalar outputs) or across
  output components (tensor outputs).
