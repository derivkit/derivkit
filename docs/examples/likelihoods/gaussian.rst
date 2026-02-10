.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Gaussian Likelihood
==============================

This section shows how to evaluate a Gaussian likelihood using
:class:`derivkit.likelihood_kit.LikelihoodKit`.

A Gaussian likelihood describes the probability of observed data under a normal
noise model with mean ``mu`` and covariance ``cov``.

For a set of samples ``data`` and a model prediction ``mu``, the Gaussian
log-likelihood is

.. math::

   \ln p(\mathrm{data}\mid \mu, \mathrm{cov})
   = -\frac{1}{2}\left[(\mathrm{data}-\mu)^{T}\,\mathrm{cov}^{-1}\,(\mathrm{data}-\mu)
   + \ln\det(2\pi\,\mathrm{cov})\right].

Notation
~~~~~~~~

- ``n`` denotes the number of data samples.
- ``data`` contains ``n`` samples (internally treated as a column of samples).
- ``mu`` is the Gaussian mean at each sample (``model_parameters`` has shape ``(n,)``).

The primary interface for evaluating the Gaussian likelihood is
:meth:`derivkit.likelihood_kit.LikelihoodKit.gaussian`.
For advanced usage, see :func:`derivkit.likelihoods.gaussian.build_gaussian_likelihood`.

For a conceptual overview of likelihoods, see :doc:`../../about/kits/likelihood_kit`.


Gaussian log-likelihood
-----------------------

For inference, you should almost always work with the log-likelihood
for numerical stability.

.. doctest:: gaussian_loglikelihood

   >>> import numpy as np
   >>> from derivkit import LikelihoodKit
   >>> # Observed data samples
   >>> data = np.array([[0.2], [-0.1], [0.05]])
   >>> # Model prediction (Gaussian mean at each sample)
   >>> mu = np.array([0.0, 0.0, 0.0])
   >>> # Diagonal covariance given as variances
   >>> cov = np.array([0.1**2, 0.1**2, 0.1**2])
   >>> # Create LikelihoodKit instance
   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> # Evaluate Gaussian log-likelihoods
   >>> grid, loglike = lkit.gaussian(cov=cov)
   >>> print(bool(np.isfinite(loglike)))
   True


Gaussian PDF (small problems only)
----------------------------------

If you explicitly need probability density values (not recommended for large
or high-dimensional problems), set ``return_log=False``.

.. doctest:: gaussian_pdf

   >>> import numpy as np
   >>> from derivkit import LikelihoodKit
   >>> data = np.array([[0.2], [-0.1], [0.05]])
   >>> mu = np.array([0.0, 0.0, 0.0])
   >>> cov = np.array([0.1**2, 0.1**2, 0.1**2])
   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> grid, pdf = lkit.gaussian(cov=cov, return_log=False)
   >>> print(bool(np.isfinite(pdf) and (pdf >= 0.0)))
   True


Covariance input forms
----------------------

The covariance can be provided in several equivalent forms.

.. doctest:: gaussian_covariance_forms

   >>> import numpy as np
   >>> from derivkit import LikelihoodKit
   >>> # Observed data samples and model prediction
   >>> data = np.array([[0.1], [-0.2]])
   >>> mu = np.array([0.0, 0.0])
   >>> # Initialize LikelihoodKit
   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> _, loglike1 = lkit.gaussian(cov=0.05**2)
   >>> # Diagonal variances (1D array)
   >>> _, loglike2 = lkit.gaussian(cov=np.array([0.05**2, 0.05**2]))
   >>> # Full covariance matrix (2D)
   >>> cov2d = np.array([
   ...     [0.0025, 0.0],
   ...     [0.0,    0.0025],
   ... ])
   >>> _, loglike3 = lkit.gaussian(cov=cov2d)
   >>> print(np.allclose(loglike1, loglike2) and np.allclose(loglike2, loglike3))
   True


Returned objects
----------------

The Gaussian likelihood returns a tuple ``(coordinate_grids, values)``.

- ``coordinate_grids`` is a tuple of 1D arrays, one per data dimension
- ``values`` is either the PDF or log-PDF evaluated on the grid

.. doctest:: gaussian_return_shapes

   >>> import numpy as np
   >>> from derivkit import LikelihoodKit
   >>> data = np.array([[0.1], [-0.1]])
   >>> mu = np.array([0.0, 0.0])
   >>> cov = np.array([0.05**2, 0.05**2])
   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> grid, loglike = lkit.gaussian(cov=cov)
   >>> print(isinstance(grid, tuple))
   True
   >>> print(bool(np.isfinite(loglike)))
   True


Notes
-----

- By default, the Gaussian likelihood returns the log-likelihood (``return_log=True``).
- ``model_parameters`` must provide one mean value per data sample (``mu`` has shape ``(n,)``).
- ``cov`` can be provided as a scalar variance, a 1D array of diagonal variances, or a full 2D covariance matrix.
- For high-dimensional data, working with the PDF directly can lead to numerical underflow; prefer log-likelihoods.
- The covariance matrix must be positive definite to ensure a valid likelihood.
- The Gaussian likelihood assumes samples are conditionally independent given the model parameters.
- For correlated data, provide the full covariance matrix to capture dependencies.
- The Gaussian likelihood is appropriate for continuous data; for discrete data, use a Poisson or multinomial likelihood.
- When combining multiple likelihood terms, sum log-likelihoods rather than multiplying PDFs.
