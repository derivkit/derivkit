Gradient
========

This section shows how to compute the gradient of a scalar-valued function using DerivKit.

The gradient describes how a scalar output changes with respect to each model
parameter.

For a set of parameters :math:`\theta` and a scalar-valued function :math:`f(\theta)`,
the gradient is the vector of first derivatives of :math:`f` with respect :math:`\theta`.

**Notation**

- ``p`` denotes the number of model parameters (``theta`` has shape ``(p,)``).

If ``f(theta)`` returns a scalar and ``theta`` has shape ``(p,)``, the gradient
has shape ``(p,)``, with one component per parameter.

See also :doc:`jacobian` for vector-valued outputs and :doc:`hessian` for second derivatives.

The primary interface for computing the gradient is
:meth:`derivkit.calculus_kit.CalculusKit.gradient`.

For advanced usage and backend-specific keyword arguments, see
:func:`derivkit.calculus.gradient.build_gradient`.

You can choose the derivative backend via ``method`` and pass backend-specific
options via ``**dk_kwargs`` (forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).


Basic usage
-----------

.. doctest:: gradient_basic

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2

   >>> x0 = np.array([0.5, 2.0])
   >>> calc = CalculusKit(func, x0=x0)

   >>> grad = calc.gradient()
   >>> print(grad)  # shape (p,)
   [0.87758256 4.        ]

   >>> ref = np.array([np.cos(0.5), 4.0])
   >>> print(ref)
   [0.87758256 4.        ]


Finite differences (Ridders) via ``dk_kwargs``
------------------------------------------

.. doctest:: gradient_finite_ridders

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2

   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   >>> grad = calc.gradient(
   ...     method="finite",
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(grad)
   [0.87758256 4.        ]


Adaptive backend via ``dk_kwargs``
------------------------------

.. doctest:: gradient_adaptive

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2

   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))

   >>> grad = calc.gradient(
   ...     method="adaptive",
   ...     n_points=12,
   ...     spacing="auto",
   ...     ridge=1e-10,
   ... )
   >>> print(grad)
   [0.87758256 4.        ]


Parallelism across parameters
-----------------------------

Different gradient components can be computed in parallel.
The number of parallel processes can be tuned with the ``n_workers`` parameter.

.. doctest:: gradient_parallel

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def f(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2 + np.cos(theta[2])

   >>> calc = CalculusKit(f, x0=np.array([0.5, 2.0, 0.1]))

   >>> grad = calc.gradient(
   ...     method="finite",
   ...     n_workers=3,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ... )
   >>> print(grad)
   [ 0.87758256  4.         -0.09983342 ]


Notes
-----

- ``n_workers`` parallelizes across parameters (gradient components).
- The function must return a **scalar**. If it returns a vector or higher-rank
  tensor, ``build_gradient`` raises ``TypeError``.
