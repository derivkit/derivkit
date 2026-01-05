Poisson Likelihood
==================

This section shows how to evaluate a Poisson likelihood using
:class:`derivkit.likelihood_kit.LikelihoodKit`.

A Poisson likelihood describes the probability of observing discrete count data
under a Poisson noise model with expected counts ``mu``.

For observed integer counts ``data`` and a model prediction ``mu``, the Poisson
log-likelihood is

.. math::

   \ln p(\mathrm{data}\mid \mu)
   = \sum_i \left[ d_i \ln \mu_i - \mu_i - \ln(d_i!) \right].

**Notation**

- ``n`` denotes the number of count observations.
- ``data`` contains ``n`` observed non-negative integer counts.
- ``d_i`` denotes the observed count in the ``i``-th observation.
- ``mu`` is the expected count at each observation
  (``model_parameters`` has shape ``(n,)``).

The primary interface for evaluating the Poisson likelihood is
:meth:`derivkit.likelihood_kit.LikelihoodKit.poissonian`.

For advanced usage, see :func:`derivkit.likelihoods.poisson.build_poisson_likelihood`.


Poisson log-likelihood
----------------------

For inference, you should almost always work with the log-likelihood
for numerical stability.

.. doctest:: poisson_loglike

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> # Observed counts (must be non-negative integers)
   >>> counts = np.array([1, 2, 3, 4])
   >>> # Expected counts (Poisson means)
   >>> mu = np.array([0.8, 1.6, 2.4, 3.2])
   >>> # Create LikelihoodKit instance
   >>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
   >>> # Evaluate Poisson log-PMF
   >>> counts_aligned, loglike = lkit.poissonian()
   >>> print(np.size(loglike) == np.size(counts_aligned))
   True
   >>> print(np.all(np.isfinite(loglike)))
   True
   >>> print(np.array_equal(np.ravel(counts_aligned), counts))
   True


Poisson PMF
-----------

If you explicitly need probability mass values, set ``return_log=False``.

.. doctest:: poisson_pmf

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> counts = np.array([1, 2, 3, 4])
   >>> mu = np.array([0.8, 1.6, 2.4, 3.2])
   >>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
   >>> counts_aligned, pmf = lkit.poissonian(return_log=False)
   >>> print(np.array_equal(np.ravel(counts_aligned), counts))
   True
   >>> print(np.all((pmf >= 0.0) & np.isfinite(pmf)))
   True
   >>> print(np.size(pmf) == np.size(counts_aligned))
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
   >>> _, loglike = lkit.poissonian()
   >>> _, pmf = lkit.poissonian(return_log=False)
   >>> print(np.allclose(np.exp(loglike), pmf))
   True


Notes
-----

- ``model_parameters`` must provide one expected count per observation
  (``mu`` has shape ``(n,)``).
- ``data`` must contain non-negative integers.
- ``mu`` must be strictly positive to ensure a valid likelihood.
- The Poisson likelihood assumes observations are conditionally independent.
- By default, the Poisson likelihood returns the log-likelihood (``return_log=True``).
- For large count values, working with the PMF directly can lead to numerical
  underflow; prefer log-likelihoods.
- When combining multiple Poisson likelihood terms, sum log-likelihoods rather
  than multiplying PMFs.
- For continuous data or approximately Gaussian noise, consider using a
  Gaussian likelihood instead.
