Poisson Likelihood
==================

This section shows how to evaluate a Poisson likelihood using
:class:`derivkit.likelihood_kit.LikelihoodKit`.

In these examples, ``model_parameters`` is interpreted directly as the expected
counts (Poisson means) ``mu`` at the current parameter point. The likelihood is
evaluated for the provided observed integer ``data``.


Poisson log-likelihood (recommended)
------------------------------------

For inference, you should almost always work with the **log-likelihood**
for numerical stability.

.. doctest:: poisson_logpmf

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> # Observed counts (must be non-negative integers)
   >>> counts = np.array([1, 2, 3, 4])

   >>> # Expected counts (Poisson means)
   >>> mu = np.array([0.8, 1.6, 2.4, 3.2])

   >>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
   >>> counts_aligned, logpmf = lkit.poissonian(return_log=True)

   >>> print(np.array_equal(counts_aligned, counts))
   True
   >>> print(logpmf.shape)
   (4,)
   >>> print(np.all(np.isfinite(logpmf)))
   True


Poisson PMF
-----------

If you explicitly need probability mass values, set ``return_log=False``
(default).

.. doctest:: poisson_pmf

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> counts = np.array([1, 2, 3, 4])
   >>> mu = np.array([0.8, 1.6, 2.4, 3.2])

   >>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
   >>> counts_aligned, pmf = lkit.poissonian(return_log=False)

   >>> print(np.array_equal(counts_aligned, counts))
   True
   >>> print(pmf.shape)
   (4,)
   >>> print(np.all((pmf >= 0.0) & np.isfinite(pmf)))
   True


Log/linear consistency
----------------------

``return_log=True`` and ``return_log=False`` are consistent up to exponentiation.

.. doctest:: poisson_log_linear_consistency

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit

   >>> counts = np.array([0, 1, 2, 3])
   >>> mu = np.array([0.5, 0.8, 1.6, 2.4])

   >>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
   >>> _, logpmf = lkit.poissonian(return_log=True)
   >>> _, pmf = lkit.poissonian(return_log=False)

   >>> print(np.allclose(np.exp(logpmf), pmf))
   True


Notes
-----

- ``data`` is reshaped internally to align with ``model_parameters``.
- The return value is ``(counts, values)`` where ``values`` are PMF values
  (or log-PMF if ``return_log=True``).
- ``mu`` must be positive; counts must be non-negative integers.
