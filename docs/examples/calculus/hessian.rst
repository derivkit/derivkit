Hessian
=======

This section shows how to compute the Hessian (matrix of second derivatives)
using DerivKit.

The Hessian describes the local curvature of a function with respect to the
model parameters.

For a function ``f(theta)``, the Hessian is the matrix of second derivatives
with respect to the parameters.

**Notation**

- ``p`` denotes the number of model parameters (``theta`` has shape ``(p,)``).

Depending on the output type of ``f(theta)``, the Hessian has the following shape:

- scalar output ``f(theta)`` → Hessian shape ``(p, p)``
- tensor output ``f(theta)`` with shape ``out_shape`` → Hessian shape
  ``(*out_shape, p, p)``

See also :doc:`gradient` for first derivatives and :doc:`jacobian` for
vector-valued outputs.

The primary interface for computing the Hessian is
:meth:`derivkit.calculus_kit.CalculusKit.hessian`.

For advanced usage and backend-specific keyword arguments, see
:func:`derivkit.calculus.hessian.build_hessian`.

You can choose the derivative backend via ``method`` and pass backend-specific
options via ``**dk_kwargs`` (forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`).


Basic usage (scalar-valued function)
------------------------------------

.. doctest:: hessian_basic

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2
   >>> # Point at which to compute the Hessian
   >>> x0 = np.array([0.5, 2.0])
   >>> # Create CalculusKit instance and compute Hessian
   >>> calc = CalculusKit(func, x0=x0)
   >>> hess = calc.hessian()
   >>> print(hess)
   [[-0.47942554  1.        ]
    [ 1.          2.        ]]
   >>> print(hess.shape)
   (2, 2)
   >>> ref = np.array([
   ...     [-np.sin(0.5), 1.0],
   ...     [1.0, 2.0],
   ... ])
   >>> print(ref)
   [[-0.47942554  1.        ]
    [ 1.          2.        ]]


Hessian diagonal only
---------------------

For large parameter spaces you may only need the diagonal of the Hessian.
DerivKit provides a fast helper for this case.

.. doctest:: hessian_diag

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2
   >>> # Instantiate CalculusKit and compute Hessian diagonal
   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))
   >>> hess_diag = calc.hessian_diag()
   >>> print(hess_diag)
   [-0.47942554  2.        ]
   >>> ref = np.array([-np.sin(0.5), 2.0])
   >>> np.allclose(hess_diag, ref)
   True


Tensor-valued outputs
---------------------

If the function returns a tensor, the Hessian is computed independently for
each output component.

The result is reshaped back to ``(*out_shape, p, p)``.

.. doctest:: hessian_tensor

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a tensor-valued function
   >>> def func(theta):
   ...     return np.array([
   ...         np.sin(theta[0]),
   ...         theta[0] * theta[1] + theta[1] ** 2,
   ...     ])
   >>> # Point at which to compute the Hessian
   >>> x0 = np.array([0.5, 2.0])
   >>> # Create CalculusKit instance and compute Hessian
   >>> calc = CalculusKit(func, x0=x0)
   >>> hess = calc.hessian()
   >>> print(hess.shape)
   (2, 2, 2)
   >>> hess0_ref = np.array([
   ...     [-np.sin(0.5), 0.0],
   ...     [0.0, 0.0],
   ... ])
   >>> hess1_ref = np.array([
   ...     [0.0, 1.0],
   ...     [1.0, 2.0],
   ... ])
   >>> np.allclose(hess[0], hess0_ref)
   True
   >>> np.allclose(hess[1], hess1_ref)
   True


Finite differences (Ridders) via ``dk_kwargs``
------------------------------------------

.. doctest:: hessian_finite_ridders

   >>> import numpy as np
   >>> from derivkit.calculus_kit import CalculusKit
   >>> # Define a scalar-valued function
   >>> def func(theta):
   ...     return np.sin(theta[0]) + theta[0] * theta[1] + theta[1] ** 2
   >>> # Create CalculusKit instance and compute Hessian
   >>> calc = CalculusKit(func, x0=np.array([0.5, 2.0]))
   >>> hess = calc.hessian(
   ...     method="finite",
   ...     n_workers=4,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(hess)
   [[-0.47942554  1.        ]
    [ 1.          2.        ]]


Notes
-----

- For scalar outputs, only the upper triangle of the Hessian is evaluated and
  mirrored for efficiency.
- For tensor outputs, each component is treated as a scalar internally
  (flattened, differentiated, and reshaped back).
- ``n_workers`` parallelizes Hessian tasks (entries and/or components).
- When using :meth:`CalculusKit.hessian_diag`, mixed partial derivatives are
  skipped for speed.
