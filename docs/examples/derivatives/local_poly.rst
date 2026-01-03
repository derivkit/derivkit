Local polynomial fit
====================

This section shows how to compute derivatives using the local polynomial
backend in DerivKit.

This backend (``method="local_polynomial"``; aliases: ``"local-polynomial"``,
``"lp"``) estimates derivatives by sampling the function near ``x0``, fitting a
local polynomial, and (optionally) trimming outlier samples before reading off
the derivative from the fitted coefficients.

Use this when finite differences are too sensitive to noise, but you still want
a lightweight method without the full adaptive Chebyshev machinery.

The primary interface for this backend is
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.


Basic usage (automatic degree)
------------------------------

By default, the polynomial degree is chosen as ``max(order + 2, 3)`` (capped by
the backend configuration).

.. doctest:: lp_basic

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ... )
   >>> print(val)
   0.76484219
   >>> print(np.cos(0.7))  # reference
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Specify polynomial degree
-------------------------

You can explicitly choose the polynomial degree (must satisfy
``degree >= order``).

.. doctest:: lp_degree

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="lp",
   ...     order=1,
   ...     degree=5,
   ... )
   >>> print(val)
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Return an internal error estimate
---------------------------------

If ``return_error=True``, the method returns an internal error proxy. This is
typically based on disagreement between a trimmed fit and an untrimmed
least-squares fit (or a fallback uncertainty estimate if trimming is disabled
or fails).

.. doctest:: lp_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val, err = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ...     degree=5,
   ...     return_error=True,
   ... )
   >>> print(val)
   0.76484219
   >>> print("err_small:", err < 1e-12)
   err_small: True
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Diagnostics (what was sampled / what got trimmed)
-------------------------------------------------

If ``diagnostics=True``, the method returns a diagnostics dictionary that
includes the sampled points, which samples were used in the final fit, and
basic metadata about the fit.

.. doctest:: lp_diagnostics

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val, diag = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ...     diagnostics=True,
   ... )
   >>> print(val)
   0.76484219
   >>> print("has_keys:", all(k in diag for k in ["x", "degree", "order", "used_mask"]))
   has_keys: True
   >>> print("n_samples:", len(diag["x"]))
   n_samples: 8
   >>> print("n_used:", int(np.sum(diag["used_mask"])))
   n_used: 8


Noisy function example
----------------------

Local polynomial fitting is often more stable than plain finite differences when
function evaluations are noisy.

.. doctest:: lp_noisy

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> rng = np.random.default_rng(0)

   >>> def noisy_sin(x):
   ...     return np.sin(x) + 0.01 * rng.normal()

   >>> dk = DerivativeKit(function=noisy_sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ...     degree=5,
   ... )
   >>> print(abs(val - np.cos(0.7)) < 0.1)
   True


Notes
-----

- This backend supports scalar and vector outputs (computed component-wise).
- ``n_workers`` can be used to parallelize sampling/evaluation.
- For heavy noise or badly conditioned local fits, consider the ``adaptive`` backend.
- If ``x0`` is an array, derivatives are computed independently at each point
  and stacked with leading shape ``x0.shape``.
