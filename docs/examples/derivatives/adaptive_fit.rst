Adaptive polynomial fit (Chebyshev)
===================================

This section shows how to compute derivatives using the adaptive polynomial
(Chebyshev) backend in DerivKit.

This backend (``method="adaptive"``) estimates derivatives by fitting a local
polynomial to samples around the expansion point ``x0`` on a symmetric
Chebyshev grid. It is designed for robustness, with stable scaling, optional
ridge regularization, and diagnostics to detect poorly conditioned fits.

The primary interface for this backend is
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
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ... )
   >>> print(val)
   0.76484219
   >>> print(np.cos(0.7))  # reference
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


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
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=12,
   ...     spacing=1e-2,
   ... )
   >>> print(val)
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0


Domain-aware sampling (stay inside bounds)
------------------------------------------

If the function is only defined on a finite interval, you can specify a domain
to ensure all samples remain valid.

The adaptive grid is clipped or transformed as needed to stay inside bounds.

.. doctest:: adaptive_domain

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> f = np.log
   >>> x0 = 0.05
   >>> dk = DerivativeKit(function=f, x0=x0)

   >>> val = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     domain=(0.0, None),
   ...     spacing="auto",
   ...     base_abs=1e-3,
   ... )
   >>> print(val)
   20.0
   >>> print(1.0 / x0)  # reference
   20.0
   >>> print(float(np.round(abs(val - 1.0 / x0), 12)))
   0.0


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
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> offsets = np.array([-2e-2, -1e-2, 0.0, 1e-2, 2e-2])
   >>> val = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     grid=("offsets", offsets),
   ... )
   >>> print(val)
   0.76484219
   >>> print(float(np.round(abs(val - np.cos(0.7)), 12)))
   0.0
   >>> print("n_grid:", len(offsets))
   n_grid: 5


Ridge regularization (stabilize ill-conditioned fits)
-----------------------------------------------------

If the polynomial fit becomes ill-conditioned (for example due to tight spacing,
high degree, or noisy evaluations), a small ridge term can improve numerical
stability.

.. doctest:: adaptive_ridge

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val = dk.differentiate(
   ...     method="adaptive",
   ...     order=2,
   ...     n_points=14,
   ...     spacing=1e-2,
   ...     ridge=1e-10,
   ... )
   >>> print(val)
   -0.64421769
   >>> print(-np.sin(0.7))  # reference
   -0.64421769
   >>> print(float(np.round(abs(val + np.sin(0.7)), 12)))
   0.0


Return an error proxy and diagnostics
-------------------------------------

You can request an error proxy and detailed diagnostics from the polynomial fit.

- ``return_error=True`` returns an RMS residual proxy
- ``diagnostics=True`` returns a dictionary with fit metadata

.. doctest:: adaptive_diagnostics

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> dk = DerivativeKit(function=np.sin, x0=0.7)

   >>> val, err, diag = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     return_error=True,
   ...     diagnostics=True,
   ... )

   >>> print(val)
   0.76484219
   >>> print("err:", float(np.round(err, 12)))
   err: 0.0
   >>> print("n_samples:", len(diag["x"]))
   n_samples: 8
   >>> print("has_keys:", all(k in diag for k in ["x", "degree", "rrms"]))
   has_keys: True


Noisy function example
----------------------

Adaptive polynomial fitting is often more stable than finite differences when
function evaluations are noisy.

.. doctest:: adaptive_noisy

   >>> import numpy as np
   >>> from derivkit.derivative_kit import DerivativeKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> rng = np.random.default_rng(0)

   >>> def noisy_sin(x):
   ...     return np.sin(x) + 0.02 * rng.normal()

   >>> dk = DerivativeKit(function=noisy_sin, x0=0.7)

   >>> val, err = dk.differentiate(
   ...     method="adaptive",
   ...     order=1,
   ...     n_points=16,
   ...     spacing="auto",
   ...     return_error=True,
   ... )
   >>> print(abs(val - np.cos(0.7)) < 0.1)
   True


Notes
-----

- This backend supports scalar and vector outputs (computed component-wise).
- For smooth, noise-free functions, finite differences may be faster.
- If you need explicit trimming or outlier handling, see the
  ``local_polynomial`` backend.
- If ``x0`` is an array, derivatives are computed independently at each point
  and stacked with leading shape ``x0.shape``.
