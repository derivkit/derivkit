Gaussian Likelihood
===================

Minimal, task-oriented examples for evaluating a Gaussian likelihood with
:class:`derivkit.likelihood_kit.LikelihoodKit`.

Gaussian log-likelihood (recommended)
-------------------------------------

Use ``return_log=True`` for numerical stability (this is what you usually want
for inference).

.. code-block:: python

   import numpy as np
   from derivkit.likelihood_kit import LikelihoodKit

   # Observed data vector
   data = np.array([0.2, -0.1, 0.05])

   # Model prediction (Gaussian mean)
   mu = np.array([0.0, 0.0, 0.0])

   # Diagonal covariance specified as variances (1D)
   cov = np.array([0.1**2, 0.1**2, 0.1**2])

   lkit = LikelihoodKit(data=data, model_parameters=mu)
   grid, logpdf = lkit.gaussian(cov=cov, return_log=True)

   print(logpdf)

Gaussian PDF (small problems)
-----------------------------

If you need probability density values instead of log-density, set
``return_log=False`` (default).

.. code-block:: python

   import numpy as np
   from derivkit.likelihood_kit import LikelihoodKit

   data = np.array([0.2, -0.1, 0.05])
   mu = np.array([0.0, 0.0, 0.0])

   # Full covariance matrix (2D)
   cov = np.array([
       [0.01, 0.00, 0.00],
       [0.00, 0.01, 0.00],
       [0.00, 0.00, 0.01],
   ])

   lkit = LikelihoodKit(data=data, model_parameters=mu)
   grid, pdf = lkit.gaussian(cov=cov)

   print(pdf)

Notes
-----

- ``cov`` may be a scalar, a 1D diagonal (variances), or a full 2D covariance.
- The return value is ``(coordinate_grids, probabilities)`` where
  ``coordinate_grids`` is a tuple of 1D arrays (one per dimension).
