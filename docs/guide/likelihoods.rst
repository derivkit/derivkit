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
