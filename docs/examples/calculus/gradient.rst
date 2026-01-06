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
For more information on gradient, see :doc:`../../about/kits/calculus_kit`.

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
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2
   >>> # Point at which to compute the gradient
   >>> x0 = np.array([0.5, 2.0])
   >>> # Create CalculusKit instance and compute gradient
   >>> calc = CalculusKit(func, x0=x0)
   >>> grad = calc.gradient()
   >>> print(np.round(np.asarray(grad).reshape(-1), 6))  # shape (p,)
   [0.877583 4.      ]
   >>> ref = np.array([np.cos(0.5), 4.0])
   >>> print(np.round(ref, 6))
   [0.877583 4.      ]


Finite differences (Ridders) via ``dk_kwargs``
----------------------------------------------

.. doctest:: gradient_finite_ridders

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2
   >>> # Create CalculusKit instance and compute gradient
   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))
   >>> grad = calc.gradient(
   ...     method="finite",
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(np.round(np.asarray(grad).reshape(-1), 6))
   [0.877583 4.      ]


Adaptive backend via ``dk_kwargs``
----------------------------------

.. doctest:: gradient_adaptive

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2
   >>> # Create CalculusKit instance and compute gradient
   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))
   >>> grad = calc.gradient(
   ...     method="adaptive",
   ...     n_points=12,
   ...     spacing="auto",
   ...     ridge=1e-10,
   ... )
   >>> print(np.round(np.asarray(grad).reshape(-1), 6))
   [0.877583 4.      ]


Parallelism across parameters
-----------------------------

Different gradient components can be computed in parallel.
The number of parallel processes can be tuned with the ``n_workers`` parameter.

.. doctest:: gradient_parallel

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def f(theta):
   ...     return np.sin(theta[0]) + theta[1] ** 2 + np.cos(theta[2])
   >>> # Create CalculusKit instance and compute gradient
   >>> calc = CalculusKit(f, x0=np.array([0.5, 2.0, 0.1]))
   >>> grad = calc.gradient(
   ...     method="finite",
   ...     n_workers=3,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ... )
   >>> print(np.round(np.asarray(grad).reshape(-1), 6))
   [ 0.877583  4.       -0.099833]


Notes
-----

- ``n_workers`` can speed up expensive functions by parallelizing gradient components.
  For cheap functions, overhead may dominate.
- The function must return a **scalar**. If it returns a vector or higher-rank
  tensor, :meth:`derivkit.CalculusKit.gradient` raises ``TypeError``.
