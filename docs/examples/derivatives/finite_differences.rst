.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| Finite differences
===========================

This section shows how to compute derivatives using the finite-difference
backend in DerivKit.

With ``method="finite"``, derivatives are estimated using central-difference
stencils evaluated around the expansion point ``x0``. Optional extrapolation
schemes (such as Richardson or Ridders) can be used to reduce truncation error
and, in some cases, provide an internal error estimate.

This backend is typically **fast and effective for smooth, noise-free
functions** that are inexpensive to evaluate. Because finite differences rely
on explicit step sizes, their accuracy can degrade for noisy functions or poorly
scaled problems.

For more information on this and other implemented derivative methods,
see :doc:`../../about/kits/derivative_kit`.

The primary interface for this backend is
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
You can select this backend via ``method="finite"`` and control the stencil,
step size, and extrapolation behavior using keyword arguments forwarded to the
finite-difference engine.


Basic usage
-----------

A single central-difference stencil evaluation (no extrapolation).

.. doctest:: finite_basic

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Create a DerivativeKit at expansion point x0 for f(x)=sin(x)
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # First derivative using a 5-point central stencil with step h=1e-2
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation=None,
   ... )
   >>> # Compare against the analytic derivative cos(x0)
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print(abs(deriv - np.cos(0.7)) < 1e-6)
   True


Plain finite difference + crude error estimate
----------------------------------------------

If you set ``return_error=True`` with ``extrapolation=None``, the engine does
a second evaluation at ``h/2`` and returns ``|D(h) - D(h/2)|`` as a simple
error estimate.

.. doctest:: finite_plain_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv, err = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation=None,
   ...     return_error=True,
   ... )
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print("err_small:", err < 1e-4)
   err_small: True
   >>> print(abs(deriv - np.cos(0.7)) < 1e-6)
   True


Richardson extrapolation (fixed levels)
---------------------------------------

Richardson uses a known truncation order to combine estimates at smaller step
sizes. Use ``levels`` for a fixed number of extrapolation steps.

.. doctest:: finite_richardson_fixed

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Richardson extrapolation using 4 refinement levels
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="richardson",
   ...     levels=4,
   ... )
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print(abs(deriv - np.cos(0.7)) < 1e-10)
   True


Richardson extrapolation (adaptive levels)
------------------------------------------

If ``levels=None`` (default), Richardson runs in adaptive mode.

.. doctest:: finite_richardson_adaptive

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Adaptive Richardson extrapolation (stops when converged)
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="richardson",
   ... )
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print(abs(deriv - np.cos(0.7)) < 1e-10)
   True


Ridders extrapolation + error estimate
--------------------------------------

Ridders is similar in spirit, but includes an internal error estimate. This is
often a good default when you want automatic error control.

.. doctest:: finite_ridders_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Ridders extrapolation with a fixed number of levels
   >>> deriv, err = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ...     return_error=True,
   ... )
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print("err_small:", err < 1e-8)
   err_small: True
   >>> print(abs(deriv - np.cos(0.7)) < 1e-10)
   True


Vector-valued function (component-wise derivatives)
---------------------------------------------------

Finite differences support vector outputs; derivatives are computed component-wise.

.. doctest:: finite_vector_output

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Define a vector-valued function
   >>> def vec_func(x):
   ...     return np.array([np.sin(x), np.cos(x)])
   >>> x0 = 0.3  # expansion point
   >>> dk = DerivativeKit(function=vec_func, x0=x0)
   >>> # Differentiate each output component with respect to x
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-3,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> ref = np.array([np.cos(x0), -np.sin(x0)])
   >>> print(np.round(deriv, 8))
   [ 0.95533649 -0.29552021]
   >>> print(bool(np.allclose(deriv, ref, atol=1e-8, rtol=0.0)))
   True


Multiple expansion points
-------------------------

If ``x0`` is an array, the derivative is computed independently at each point
and stacked with leading shape ``x0.shape``.

.. doctest:: finite_multiple_x0

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Define multiple expansion points
   >>> x0 = np.array([0.1, 0.7, 1.2])
   >>> dk = DerivativeKit(function=np.sin, x0=x0)
   >>> # Compute the derivative at all points in x0 in one call
   >>> deriv = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-3,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> ref = np.cos(x0)
   >>> print("shape:", deriv.shape)
   shape: (3,)
   >>> print(bool(np.allclose(deriv, ref, atol=1e-8, rtol=0.0)))
   True


Notes
-----

- Finite differences support scalar and vector outputs (computed component-wise).
- Supported stencil sizes are ``num_points in {3, 5, 7, 9}``.
- Supported derivative orders depend on the stencil size.
- Richardson and Ridders extrapolation can improve accuracy but increase the
  number of function evaluations.
- Error estimates (when available) are heuristic and intended for diagnostics.
- For noisy functions or when step-size tuning becomes fragile, consider the
  ``local_polynomial`` or ``adaptive`` backends instead.
- If ``x0`` is an array, derivatives are computed independently at each point
  and stacked with leading shape ``x0.shape``.
