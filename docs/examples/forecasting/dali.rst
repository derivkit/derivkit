.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| DALI tensors
=====================

This section shows how to construct Derivative Approximation for
LIkelihoods (DALI) tensors using :class:`derivkit.forecast_kit.ForecastKit`.

The DALI expansion extends the Fisher matrix by including higher-order
derivative information, allowing non-Gaussian structure in the likelihood
to be approximated locally around a fiducial parameter point ``theta0``.

In DerivKit, DALI tensors are returned using an *introduced-at-order*
convention: each forecast order contributes a tuple of tensors, and all
orders up to the requested one are returned.

The model must map parameters ``theta`` to a 1D data vector, and the observable
covariance matrix must have shape ``(n, n)``, where ``n`` is the length of the
data vector.

For a conceptual overview of DALI forecasting and its interpretation, see
:doc:`../../about/kits/forecast_kit`.

If you want to visualize parameter contours from DALI tensors, see the example
:doc:`dali_contours`, which shows how to generate GetDist samples and plot
confidence regions from the returned multiplets.


DALI tensor shapes
------------------

For ``p`` model parameters, the DALI expansion returns tensors with the
following shapes:

- order 1 (Fisher; "singlet-DALI"):
  - ``F`` with shape ``(p, p)``

- order 2 (doublet-DALI):
  - ``D1`` with shape ``(p, p, p)``
  - ``D2`` with shape ``(p, p, p, p)``

- order 3 (triplet-DALI):
  - ``T1`` with shape ``(p, p, p, p)``
  - ``T2`` with shape ``(p, p, p, p, p)``
  - ``T3`` with shape ``(p, p, p, p, p, p)``

All tensors are evaluated at the fiducial parameter point ``theta0``.


Basic usage
-----------

The example below constructs DALI tensors around a fiducial parameter
point ``theta0`` using a simple toy model.

.. doctest:: dali_basic

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> # Define a simple toy model: R^2 -> R^3
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> # Fiducial parameters and covariance
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> # Build ForecastKit and compute DALI tensors up to second order
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(forecast_order=2)
   >>> F = dali[1][0]
   >>> D1, D2 = dali[2]
   >>> print(F.shape)
   (2, 2)
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Choosing a derivative backend
-----------------------------

As with Fisher forecasting, you can control how derivatives are computed by
passing ``method`` and backend-specific options.

All keyword arguments are forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

.. doctest:: dali_backend_control

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(
   ...     forecast_order=2,
   ...     method="finite",
   ...     n_workers=2,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> D1, D2 = dali[2]
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Parallel execution
------------------

DALI tensor components can be computed in parallel using ``n_workers``.

This parallelizes derivative evaluations across parameters and tensor entries.

.. doctest:: dali_parallel

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> def model(theta):
   ...     return np.array([theta[0], theta[1], theta[0] + 2.0 * theta[1]])
   >>> fk = ForecastKit(
   ...     function=model,
   ...     theta0=np.array([1.0, 2.0]),
   ...     cov=np.eye(3),
   ... )
   >>> dali = fk.dali(forecast_order=2, n_workers=4)
   >>> D1, D2 = dali[2]
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Notes
-----

- DALI tensors are evaluated locally at ``theta0``.
- Each forecast order contributes a multiplet of tensors, returned via an
  introduced-at-order convention.
- The likelihood is assumed Gaussian in the data with fixed covariance.
- Higher-order forecasts increase computational cost relative to Fisher.
- Changing the derivative backend affects numerical accuracy but not tensor
  structure.
