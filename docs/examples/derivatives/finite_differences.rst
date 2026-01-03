Finite differences
==================

This section shows how to compute derivatives using the finite-difference
backend in DerivKit.

This backend (``method="finite"``) estimates derivatives using central-difference
stencils and optional extrapolation schemes. It is typically fastest for smooth,
noise-free functions that are cheap to evaluate.

The primary interface for this backend is
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

You can select this backend via ``method="finite"`` and control its behavior
using keyword arguments forwarded to the finite-difference engine.


Basic usage (plain finite difference)
-------------------------------------

A single central-difference stencil evaluation (no extrapolation).

.. doctest:: finite_basic

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation=None,
   ... )
   >>> print(val)
   0.76484219
   >>> print(np.cos(0.7))  # reference
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Plain finite difference + crude error estimate
----------------------------------------------

If you set ``return_error=True`` with ``extrapolation=None``, the engine does
a second evaluation at ``h/2`` and returns ``|D(h) - D(h/2)|`` as a simple
error estimate.

.. doctest:: finite_plain_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val, err = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation=None,
   ...     return_error=True,
   ... )
   >>> print(val)
   0.76484219
   >>> print("err_small:", err < 1e-6)
   err_small: True
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Richardson extrapolation (fixed levels)
---------------------------------------

Richardson uses a known truncation order to combine estimates at smaller step
sizes. Use ``levels`` for a fixed number of extrapolation steps.

.. doctest:: finite_richardson_fixed

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="richardson",
   ...     levels=4,
   ... )
   >>> print(val)
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Richardson extrapolation (adaptive)
-----------------------------------

If ``levels=None`` (default), Richardson runs in adaptive mode.

.. doctest:: finite_richardson_adaptive

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="richardson",
   ... )
   >>> print(val)
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Ridders extrapolation + error estimate
--------------------------------------

Ridders is similar in spirit, but includes an internal error estimate. This is
often a good default when you want automatic error control.

.. doctest:: finite_ridders_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val, err = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ...     return_error=True,
   ... )
   >>> print(val)
   0.76484219
   >>> print("err_small:", err < 1e-10)
   err_small: True
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Vector-valued function (component-wise derivatives)
---------------------------------------------------

Finite differences support vector outputs; derivatives are computed component-wise.

.. doctest:: finite_vector_output

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> def vec_func(x):
   ...     return np.array([np.sin(x), np.cos(x)])

   >>> x0 = 0.3
   >>> dk = DerivativeKit(function=vec_func, x0=x0)

   >>> val = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> ref = np.array([np.cos(x0), -np.sin(x0)])
   >>> print(val)
   [ 0.95533649 -0.29552021]
   >>> print(ref)  # reference
   [ 0.95533649 -0.29552021]
   >>> print(bool(np.allclose(val, ref, atol=1e-12, rtol=0.0)))
   True


Multiple expansion points (array ``x0``)
----------------------------------------

If ``x0`` is an array, the derivative is computed independently at each point
and stacked with leading shape ``x0.shape``.

.. doctest:: finite_multiple_x0

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> x0 = np.array([0.1, 0.7, 1.2])
   >>> dk = DerivativeKit(function=np.sin, x0=x0)

   >>> val = dk.differentiate(
   ...     method="finite",
   ...     order=1,
   ...     stepsize=1e-3,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> ref = np.cos(x0)
   >>> print("shape:", val.shape)
   shape: (3,)
   >>> print(bool(np.allclose(val, ref, atol=1e-10, rtol=0.0)))
   True


Notes
-----

- Supported stencils: ``num_points in {3, 5, 7, 9}``.
- Supported derivative orders depend on the stencil size (see API docs for details).
- When in doubt, start with ``num_points=5`` and ``extrapolation="ridders"``.
- Adaptive polynomial fitting (``method="adaptive"``) may be more robust than
  finite differences when function evaluations are noisy.
