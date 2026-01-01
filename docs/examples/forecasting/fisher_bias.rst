Fisher bias
===========

Given a Fisher matrix ``F`` and a systematic-induced data shift
:math:`\Delta\nu`, the Fisher-bias workflow maps that shift into parameter space.

Compute :math:`\Delta\nu`
-------------------------

:meth:`derivkit.forecast_kit.ForecastKit.delta_nu` computes a consistent 1D
difference vector. It accepts 1D arrays or 2D arrays (which are flattened in
row-major ("C") order to match the package convention).

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)
   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   data_unbiased = model(theta0)
   data_biased = data_unbiased + np.array([0.01, -0.02, 0.005])

   dn = fk.delta_nu(data_unbiased=data_unbiased, data_biased=data_biased)
   print(dn)
   print(dn.shape)  # shape (n,)


Compute Fisher bias and parameter shifts
----------------------------------------

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)
   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   fish = fk.fisher(method="finite", stepsize=1e-2, num_points=5)

   data_unbiased = model(theta0)
   data_biased = data_unbiased + np.array([0.01, -0.02, 0.005])
   dn = fk.delta_nu(data_unbiased=data_unbiased, data_biased=data_biased)

   bias_vec, delta_theta = fk.fisher_bias(
       fisher_matrix=fish,
       delta_nu=dn,
       method="finite",
       stepsize=1e-2,
       num_points=5,
   )

   print(bias_vec)
   print(delta_theta)
