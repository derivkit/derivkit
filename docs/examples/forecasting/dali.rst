DALI tensors (G, H)
===================

Use :meth:`derivkit.forecast_kit.ForecastKit.dali` to build the doublet-DALI
tensors ``(g, h)`` around a fiducial point ``theta0``.

For ``p`` parameters:

- ``g`` has shape ``(p, p, p)``
- ``h`` has shape ``(p, p, p, p)``

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   g_tensor, h_tensor = fk.dali()
   print(g_tensor.shape)  # (2, 2, 2)
   print(h_tensor.shape)  # (2, 2, 2, 2)


Choosing a derivative backend
-----------------------------

As with Fisher, you can choose ``method`` and pass backend-specific options via
``**dk_kwargs``.

.. code-block:: python

   import numpy as np
   from derivkit.forecast_kit import ForecastKit

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   g_tensor, h_tensor = fk.dali(
       method="finite",
       n_workers=2,
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )
   print(g_tensor.shape)  # (2, 2, 2)
   print(h_tensor.shape)  # (2, 2, 2, 2)

