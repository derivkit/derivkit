Jacobian
========

This section shows how to compute the Jacobian of a vector-valued function using DerivKit.

The Jacobian describes how each component of a vector-valued output changes with respect
to each model parameter.

For a function :math:`\mathbf{f}(\theta)`, the Jacobian is the matrix of first
derivatives of the outputs with respect to the parameters.

Notation
--------

- ``p`` denotes the number of model parameters (``theta`` has shape ``(p,)``).
- ``n`` denotes the number of output components (``f(theta)`` has shape ``(n,)``).

If ``f(theta)`` returns shape ``(n,)`` and ``theta`` has shape ``(p,)``, the
Jacobian has shape ``(n, p)``, where each column is the derivative with respect
to one parameter.

See also :doc:`gradient` for scalar-valued outputs and :doc:`hessian` for second derivatives.

The primary interface for computing the Jacobian is
:meth:`derivkit.calculus_kit.CalculusKit.jacobian`.

For advanced usage and backend-specific keyword arguments, see
:func:`derivkit.calculus.jacobian.build_jacobian`.

You can choose the derivative backend via ``method`` and pass backend-specific
options via ``**dk_kwargs`` (forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).


Basic usage
-----------

.. doctest:: jacobian_basic

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.array([
   ...         np.sin(theta[0]) + theta[1],
   ...         theta[0] * theta[1],
   ...     ])

   >>> x0 = np.array([0.5, 2.0])
   >>> calc = CalculusKit(func, x0=x0)

   >>> jac = calc.jacobian()
   >>> print(jac)
   [[0.87758256 1.        ]
    [2.         0.5       ]]
   >>> print(jac.shape)  # (n, p) = (2, 2)
   (2, 2)

   >>> ref = np.array([
   ...     [np.cos(0.5), 1.0],
   ...     [2.0, 0.5],
   ... ])
   >>> print(ref)
   [[0.87758256 1.        ]
    [2.         0.5       ]]


Finite differences (Ridders) via dk_kwargs
------------------------------------------

.. doctest:: jacobian_finite_ridders

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.array([np.sin(theta[0]) + theta[1], theta[0] * theta[1]])

   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   >>> jac = calc.jacobian(
   ...     method="finite",
   ...     n_workers=2,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(jac)
   [[0.87758256 1.        ]
    [2.         0.5       ]]


Adaptive backend via dk_kwargs
------------------------------

.. doctest:: jacobian_adaptive

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.array([np.sin(theta[0]) + theta[1], theta[0] * theta[1]])

   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   >>> jac = calc.jacobian(
   ...     method="adaptive",
   ...     n_workers=2,
   ...     n_points=12,
   ...     spacing="auto",
   ...     ridge=1e-10,
   ... )
   >>> print(jac)
   [[0.87758256 1.        ]
    [2.         0.5       ]]


Notes
-----

- ``n_workers`` parallelizes across parameters (Jacobian columns).
- The function must return a **1D** vector (shape ``(n,)``). If it returns a scalar
  or higher-rank tensor, ``build_jacobian`` raises ``TypeError``.
