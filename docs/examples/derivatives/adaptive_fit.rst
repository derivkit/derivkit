Adaptive polynomial fit
=======================

This section shows how to compute derivatives using the adaptive polynomial
(Chebyshev) method in DerivKit.

With ``method="adaptive"``, derivatives are estimated by fitting a local
polynomial to samples around the expansion point ``x0`` on a symmetric
Chebyshev grid. This approach is designed for robustness, with stable scaling,
optional ridge regularization, and diagnostics to detect poorly conditioned
fits.

For more information on this and other implemented derivative methods,
see :doc:`../../about/kits/derivative_kit`.

The primary interface for this method is
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
You can select this backend via ``method="adaptive"`` and control its behavior
using keyword arguments forwarded to the adaptive polynomial fitter.


Basic usage
-----------

By default, an adaptive Chebyshev grid is constructed automatically around
``x0`` with an automatically chosen half-width (``spacing="auto"``).

.. doctest:: adaptive_basic

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Initialize DerivativeKit with the target function and expansion point
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Compute the first derivative (adaptive backend is the default)
   >>> deriv = dk.differentiate(order=1)
   >>> print(bool(np.allclose(deriv, np.cos(0.7), rtol=0, atol=1e-8)))
   True
   >>> print(bool(abs(deriv - np.cos(0.7)) < 1e-8))
   True


Control sampling: number of points and spacing
----------------------------------------------

You can control the number of sampling points and the grid half-width.

- ``n_points`` sets the number of samples used in the local fit
- ``spacing`` controls the half-width of the sampling region

Accepted forms for ``spacing`` are:

- ``float``: absolute half-width ``h`` (samples in ``[x0-h, x0+h]``)
- ``"auto"``: scale inferred from ``x0`` with a floor ``base_abs``
- ``"<pct>%"``: percentage of a local scale

.. doctest:: adaptive_sampling_control

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Initialize DerivativeKit with the target function and expansion point
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> # Compute the first derivative with custom sampling
   >>> deriv = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=12,
   ...     spacing=1e-2,
   ... )
   >>> np.allclose(deriv, np.cos(0.7), rtol=0, atol=1e-8)
   True


Domain-aware sampling (stay inside bounds)
------------------------------------------

If the function is only defined on a finite interval, you can specify a domain
to ensure all samples remain valid.
The adaptive grid is clipped or transformed as needed to stay inside bounds.

.. doctest:: adaptive_domain

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Define a function and an expansion point
   >>> f = np.log
   >>> x0 = 0.05
   >>> dk = DerivativeKit(function=f, x0=x0)
   >>> deriv = dk.differentiate(
   ...    method="adaptive",
   ...    order=1,
   ...    domain=(0.0, None),
   ...    spacing="auto",
   ...    base_abs=1e-3
   ...    )
   >>> print(bool(np.allclose(deriv, 1.0 / x0, rtol=0, atol=1e-6)))
   True


User-supplied grids (offsets or absolute coordinates)
-----------------------------------------------------

You can override the internally constructed grid by providing explicit sample
locations.
Supported forms are:

- ``grid=("offsets", offsets)``: samples at ``x = x0 + offsets``
- ``grid=("absolute", x_values)``: samples directly at given coordinates

If zero is missing from an offsets grid, it is inserted automatically.

.. doctest:: adaptive_custom_grid

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> offsets = np.array([-2e-2, -1e-2, 1e-2, 2e-2])  # no 0 (inserted automatically)
   >>> deriv = dk.differentiate(method="adaptive", order=1, grid=("offsets", offsets))
   >>> print(np.isfinite(deriv))
   True
   >>> print(abs(deriv - np.cos(0.7)) < 1.0)
   True


Ridge regularization (stabilize ill-conditioned fits)
-----------------------------------------------------

If the polynomial fit becomes ill-conditioned (for example due to tight spacing,
high degree, or noisy evaluations), a small ridge term can improve numerical
stability.

.. doctest:: adaptive_ridge

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv = dk.differentiate(
   ...    method="adaptive",
   ...    order=2,
   ...    n_points=27,
   ...    spacing=1e-2,
   ...    ridge=1e-10
   ...    )
   >>> print(bool(np.allclose(deriv, -np.sin(0.7), rtol=0, atol=1e-6)))
   True


Return an error proxy and diagnostics
-------------------------------------

You can request an error proxy and detailed diagnostics from the polynomial fit.

- ``return_error=True`` returns an RMS residual proxy
- ``diagnostics=True`` returns a dictionary with fit metadata

.. doctest:: adaptive_diagnostics

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> dk = DerivativeKit(function=np.sin, x0=0.7)
   >>> deriv, err, diag = dk.differentiate(
   ...    method="adaptive",
   ...    order=1,
   ...    return_error=True,
   ...    diagnostics=True
   ...    )
   >>> print(bool(abs(deriv - np.cos(0.7)) < 1e-6))
   True
   >>> print(bool(np.all(np.asarray(err) >= 0)))
   True
   >>> print(bool(isinstance(diag, dict) and all(k in diag for k in ["x", "degree", "rrms"])))
   True
   >>> print(int(len(diag["x"])) >= 3)
   True


Noisy function example
----------------------

Adaptive polynomial fitting is often more stable than finite differences when
function evaluations are noisy.

.. doctest:: adaptive_noisy

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> # Deterministic "noise": sampled once on a grid and interpolated (stable across calls)
   >>> rng = np.random.default_rng(42)
   >>> x_noise = np.linspace(0.0, 2.0, 2049)
   >>> eps = 0.02 * rng.normal(size=x_noise.size)
   >>> def noisy_sin(x):
   ...     return np.sin(x) + np.interp(x, x_noise, eps)
   >>> dk = DerivativeKit(function=noisy_sin, x0=0.7)
   >>> deriv, err = dk.differentiate(
   ...    method="adaptive",
   ...    order=1,
   ...    n_points=16,
   ...    spacing="auto",
   ...    return_error=True
   ...    )
   >>> # Example-level check: derivative stays in the right ballpark
   >>> print(bool(abs(deriv - np.cos(0.7)) < 0.3))
   True


Notes
-----

- This backend supports scalar and vector outputs (computed component-wise).
- For smooth, noise-free functions, finite differences may be faster.
- If you need explicit trimming or outlier handling, see the
  ``local_polynomial`` backend.
- If ``x0`` is an array, derivatives are computed independently at each point
  and stacked with leading shape ``x0.shape``.
