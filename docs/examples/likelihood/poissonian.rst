Poisson Likelihood
==================

Minimal, task-oriented examples for evaluating a Poissonian likelihood with
:class:`derivkit.likelihood_kit.LikelihoodKit`.

Poisson log-likelihood (recommended)
------------------------------------

Use ``return_log=True`` for numerical stability (this is what you usually want
for inference).

.. code-block:: python

   import numpy as np
   from derivkit.likelihood_kit import LikelihoodKit

   # Observed counts
   counts = np.array([1, 2, 3, 4])

   # Expected counts (Poisson means)
   mu = np.array([0.8, 1.6, 2.4, 3.2])

   lkit = LikelihoodKit(data=counts, model_parameters=mu)
   counts_aligned, logpmf = lkit.poissonian(return_log=True)

   print(counts_aligned)
   print(logpmf)

Poisson PMF
-----------

If you need probability mass values instead of log-probabilities, set
``return_log=False`` (default).

.. code-block:: python

   import numpy as np
   from derivkit.likelihood_kit import LikelihoodKit

   counts = np.array([1, 2, 3, 4])
   mu = np.array([0.8, 1.6, 2.4, 3.2])

   lkit = LikelihoodKit(data=counts, model_parameters=mu)
   counts_aligned, pmf = lkit.poissonian()

   print(counts_aligned)
   print(pmf)

Notes
-----

- ``data`` is reshaped internally to align with ``model_parameters``.
- The return value is ``(counts, probabilities)`` where ``counts`` is the
  aligned count array and ``probabilities`` are PMF values (or log-PMF).
