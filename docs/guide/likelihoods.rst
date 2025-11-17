Likelihoods
===========

LikelihoodKit provides lightweight, easy-to-use wrappers for evaluating
standard likelihoods in forecasting and inference workflows.
They are designed to work naturally with ForecastKit but can be used
independently in any context.


Gaussian Likelihood
-------------------

The **Gaussian likelihood** is the most commonly used form in forecasting (see `A. Heavens, 2010. <https://arxiv.org/abs/0906.0664>`_):

.. math::

   -2 \ln \mathcal{L}
   = (d - m)^\top C^{-1} (d - m)
   + \ln\det C
   + \text{const},

where:

- ``d`` is the data vector,
- ``m`` is the model prediction,
- ``C`` is the covariance matrix.

LikelihoodKit automatically:

- flattens/reshapes inputs to consistent shapes,
- checks that ``data``, ``model``, and ``cov`` agree in dimension,
- supports identity covariance, diagonal covariance, or full covariance,
- returns both the processed data-vector (for reuse) and the likelihood value.

Example:

.. code-block:: python

   from derivkit.forecasting.likelihoods import build_gaussian_likelihood

   data = ...
   model = ...
   cov = ...

   aligned_data, loglike = build_gaussian_likelihood(data, model, cov)


Poissonian Likelihood
---------------------

The **Poisson likelihood** applies when the observables represent
counts, histograms, or binned number data (see `A. Heavens, 2010. <https://arxiv.org/abs/0906.0664>`_).

Given observed counts ``d`` and model expectation ``λ``:

.. math::

   \ln \mathcal{L}
   = \sum_i \left( d_i \ln \lambda_i - \lambda_i - \ln(d_i!) \right).

LikelihoodKit supports:

- scalar or vector ``λ`` values,
- reshaping ``data`` to match model parameters,
- automatic broadcasting when computing many bins at once.

Example:

.. code-block:: python

   from derivkit.forecasting.likelihoods import build_poissonian_likelihood

   observed = ...
   model_lambda = ...

   data_out, loglike = build_poissonian_likelihood(observed, model_lambda)


Sellentin–Heavens Likelihood (Planned)
--------------------------------------

The Gaussian likelihood assumes that the covariance matrix ``C``
is known perfectly. This is rarely true in practice: most real analyses
estimate the covariance from simulations, bootstrap resampling, or
jackknife procedures.

The **Sellentin–Heavens likelihood**
(`Sellentin & Heavens 2015 <https://arxiv.org/pdf/1506.04866>`_)
corrects the Gaussian likelihood to account for finite-sample covariance
uncertainty. It replaces the Gaussian form with a multivariate *Student-t*
structure:

.. math::

   \mathcal{L}_{\rm SH}
   \propto
   \left[1 + \frac{(d - m)^\top C^{-1}(d - m)}{N_\nu} \right]^{-N/2},

where ``N`` is the number of realizations from which ``C`` was estimated.

LikelihoodKit will implement this corrected likelihood so that users may
forecast directly from covariance-estimation pipelines without bias.


Input Handling and Validation
-----------------------------

Every likelihood wrapper in DerivKit ensures:

- shapes are consistent (e.g. 1D, 2D, flattened arrays),
- data and model parameters have matching sizes,
- covariance matrices are square and invertible (Gaussian case),
- dimension mismatches raise clear exceptions,
- return values follow the same ``(aligned_data, log_likelihood)`` convention.

These utilities make it easy to transition between forecasting tools,
numerical derivatives, and full inference workflows.


Integration with ForecastKit
----------------------------

All likelihoods are compatible with Fisher, DALI, and Fisher bias
calculations:

- Fisher derivatives use the aligned model output from the likelihood layer,
- reshaping logic ensures consistency,
- likelihoods can be swapped without changing forecasting code.

This modularity allows users to begin with simple Gaussian forecasts
and later transition to Poisson or Sellentin–Heavens likelihoods
without rewriting their models.
