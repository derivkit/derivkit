Gaussian Likelihood
===================

This section shows how to evaluate a Gaussian likelihood using
:class:`derivkit.likelihood_kit.LikelihoodKit`.

In these examples, ``model_parameters`` is interpreted directly as the Gaussian
mean vector ``mu`` at the current parameter point. The likelihood is evaluated
for the provided observed ``data`` under a Gaussian noise model specified by
the covariance ``cov``.


Gaussian log-likelihood (recommended)
-------------------------------------

For inference, you should almost always work with the **log-likelihood**
for numerical stability.

.. doctest:: gaussian_loglikelihood

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> # Observed data vector
   >>> data = np.array([0.2, -0.1, 0.05])

   >>> # Model prediction (Gaussian mean)
   >>> mu = np.array([0.0, 0.0, 0.0])

   >>> # Diagonal covariance given as variances
   >>> cov = np.array([0.1**2, 0.1**2, 0.1**2])

   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> grid, logpdf = lkit.gaussian(cov=cov, return_log=True)

   >>> print(logpdf)
   -0.33787707


Gaussian PDF (small problems only)
----------------------------------

If you explicitly need probability density values (not recommended for large
or high-dimensional problems), set ``return_log=False``.

.. doctest:: gaussian_pdf

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit
   >>> np.set_printoptions(precision=8, suppress=True)

   >>> data = np.array([0.2, -0.1, 0.05])
   >>> mu = np.array([0.0, 0.0, 0.0])

   >>> cov = np.array([
   ...     [0.01, 0.00, 0.00],
   ...     [0.00, 0.01, 0.00],
   ...     [0.00, 0.00, 0.01],
   ... ])

   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> grid, pdf = lkit.gaussian(cov=cov)

   >>> print(pdf)
   0.71341232


Covariance input forms
----------------------

The covariance can be provided in several equivalent forms.

.. doctest:: gaussian_covariance_forms

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit

   >>> data = np.array([0.1, -0.2])
   >>> mu = np.array([0.0, 0.0])

   >>> # Scalar variance (applied to all dimensions)
   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> _, logpdf1 = lkit.gaussian(cov=0.05**2, return_log=True)

   >>> # Diagonal variances (1D array)
   >>> _, logpdf2 = lkit.gaussian(cov=np.array([0.05**2, 0.05**2]), return_log=True)

   >>> # Full covariance matrix (2D)
   >>> cov2d = np.array([
   ...     [0.0025, 0.0],
   ...     [0.0,    0.0025],
   ... ])
   >>> _, logpdf3 = lkit.gaussian(cov=cov2d, return_log=True)

   >>> print(np.allclose(logpdf1, logpdf2) and np.allclose(logpdf2, logpdf3))
   True


Returned objects
----------------

The Gaussian likelihood returns a tuple ``(coordinate_grids, values)``.

- ``coordinate_grids`` is a tuple of 1D arrays, one per data dimension
- ``values`` is either the PDF or log-PDF evaluated on the grid

.. doctest:: gaussian_return_shapes

   >>> import numpy as np
   >>> from derivkit.likelihood_kit import LikelihoodKit

   >>> data = np.array([0.1, -0.1])
   >>> mu = np.array([0.0, 0.0])
   >>> cov = np.array([0.05**2, 0.05**2])

   >>> lkit = LikelihoodKit(data=data, model_parameters=mu)
   >>> grid, logpdf = lkit.gaussian(cov=cov, return_log=True)

   >>> print(len(grid))
   2
   >>> print(all(g.ndim == 1 for g in grid))
   True
   >>> print(np.isscalar(logpdf))
   True


Notes
-----

- ``return_log=True`` is recommended for numerical stability.
- For large data vectors, always work in log-space.
- The likelihood is evaluated assuming the model parameters directly specify
  the Gaussian mean.
