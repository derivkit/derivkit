Fisher matrix
=============

Use :class:`derivkit.forecast_kit.ForecastKit` to build a Fisher matrix for a
model :math:`\mu(\theta)` and observable covariance ``cov``.

The model must map parameters to a 1D data vector (shape ``(n,)``), and ``cov``
must have shape ``(n, n)`` with ``n`` being the number of data points.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   # toy model: mu(theta) in R^3, theta in R^2
   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   fish = fk.fisher()
   print(fish)
   print(fish.shape)  # (2, 2)


Choosing a derivative backend
----------------------------

Pass ``method=...`` and backend-specific options via ``**dk_kwargs`` (forwarded
to :meth:`derivkit.derivative_kit.DerivativeKit.differentiate` through the Fisher
builder).

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   fish = fk.fisher(
       method="finite",
       n_workers=2,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(fish)
