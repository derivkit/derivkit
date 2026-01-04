Local polynomial fit
====================

This section shows how to compute derivatives using the local polynomial
backend in DerivKit.

With ``method="local_polynomial"`` (aliases: ``"local-polynomial"``, ``"lp"``),
derivatives are estimated by sampling the function near the expansion point
``x0`` and fitting a low-order polynomial locally. The desired derivative is
then read off from the fitted polynomial coefficients.

This method is intended as a **simple and robust alternative to finite
differences**, particularly when function evaluations are mildly noisy or when
step-size tuning becomes fragile. Unlike the adaptive Chebyshev backend, the
local polynomial method does not attempt to optimize the sampling grid or
conditioning automatically, making it lightweight and predictable.

The primary interface for this backend is
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

You can select this backend via ``method="local_polynomial"`` (or ``"lp"``) and
control the polynomial degree, trimming behavior, and diagnostics using keyword
arguments.


Basic usage (automatic degree)
------------------------------

By default, the polynomial degree is chosen as ``max(order + 2, 3)`` (capped by
the backend configuration).

.. doctest:: lp_basic

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Initialize DerivativeKit with the target function and expansion point
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Compute the first derivative
   >>> deriv = dk.differentiate(method="local_polynomial", order=1)
   >>> # Print a stable reference-friendly value (rounding avoids tiny platform diffs)
   >>> print(float(np.round(deriv, 8)))
   0.76484218
   >>> # Compare against the analytic derivative cos(x0)
   >>> print(abs(deriv - np.cos(0.7)) < 1e-6)
   True


Specify polynomial degree
-------------------------

You can explicitly choose the polynomial degree (must satisfy
``degree >= order``).

.. doctest:: lp_degree

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv = dk.differentiate(method="lp", order=1, degree=5)
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print(abs(deriv - np.cos(0.7)) < 1e-8)
   True


Return an internal error estimate
---------------------------------

If ``return_error=True``, the method returns an internal error proxy. This is
typically based on disagreement between a trimmed fit and an untrimmed
least-squares fit (or a fallback uncertainty estimate if trimming is disabled
or fails).

.. doctest:: lp_error

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv, err = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ...     degree=5,
   ...     return_error=True,
   ... )
   >>> print(f"{deriv:.8f}")
   0.76484219
   >>> print("err_small:", err < 1e-8)
   err_small: True
   >>> print(abs(deriv - np.cos(0.7)) < 1e-8)
   True


Diagnostics (what was sampled / what got trimmed)
-------------------------------------------------

If ``diagnostics=True``, the method returns a diagnostics dictionary that
includes the sampled points, which samples were used in the final fit, and
basic metadata about the fit.

.. doctest:: lp_diagnostics

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv, diagnostics = dk.differentiate(
   ...     method="local_polynomial",
   ...     order=1,
   ...     diagnostics=True,
   ... )
   >>> print(float(np.round(deriv, 8)))
   0.76484218
   >>> required = ["ok", "x0", "degree", "order", "n_all", "n_used", "x_used", "max_rel_err_used"]
   >>> print("has_keys:", all(k in diagnostics for k in required))
   has_keys: True
   >>> print("n_used_le_n_all:", diagnostics["n_used"] <= diagnostics["n_all"])
   n_used_le_n_all: True
   >>> print("x_used_len_matches_n_used:", len(diagnostics["x_used"]) == diagnostics["n_used"])
   x_used_len_matches_n_used: True
   >>> print("ok_is_bool:", isinstance(diagnostics["ok"], bool))
   ok_is_bool: True


Noisy function example
----------------------

Local polynomial fitting is often more stable than plain finite differences when
function evaluations are noisy.

.. doctest:: lp_noisy

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Build a deterministic "noisy" function: noise is sampled once and interpolated
   >>> rng = np.random.default_rng(42)
   >>> x_noise = np.linspace(0.0, 2.0, 2049)  # dense grid for stable interpolation
   >>> eps = 0.01 * rng.normal(size=x_noise.size)
   >>> def noisy_sin(x):
   ...     return np.sin(x) + np.interp(x, x_noise, eps)
   >>> dk = DerivativeKit(function=noisy_sin, x0=0.7)
   >>> deriv = dk.differentiate(method="local_polynomial", order=1, degree=5)
   >>> # Example-level check: derivative stays close to cos(x) despite mild noise
   >>> print(abs(deriv - np.cos(0.7)) < 0.2)
   True


Notes
-----

- This backend supports scalar and vector outputs (computed component-wise).
- The polynomial degree must satisfy ``degree >= order``.
- Internal error estimates are heuristic and intended for diagnostics, not
  rigorous uncertainty quantification.
- For strongly noisy functions or poorly conditioned local fits, consider the
  ``adaptive`` backend.
- If ``x0`` is an array, derivatives are computed independently at each point
  and stacked with leading shape ``x0.shape``.
