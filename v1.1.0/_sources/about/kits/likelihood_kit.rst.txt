.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| LikelihoodKit
======================

LikelihoodKit provides lightweight, easy-to-use wrappers for evaluating
standard likelihoods in forecasting and inference workflows.
They are designed to work naturally with ForecastKit but can be used
independently in any context.

Runnable examples illustrating these methods are collected in
:doc:`../../examples/index`.


Gaussian Likelihood
-------------------

The **Gaussian likelihood** is the most commonly used form in forecasting
(see `A. Heavens, 2010. <https://arxiv.org/abs/0906.0664>`_):

.. math::

   -2 \ln \mathcal{L}
   = (d - m)^\top C^{-1} (d - m)
   + \ln\det C
   + \text{const},

where:

- ``d`` is the data vector,
- ``m`` is the model prediction,
- ``C`` is the covariance matrix.

This likelihood assumes that the data vector is approximately Gaussian-
distributed around the model prediction, with known covariance ``C``.
It is appropriate for summary statistics such as power spectra, correlation
functions, or compressed observables.

LikelihoodKit treats the Gaussian likelihood as a *structural object* rather
than a raw formula: it enforces consistent data shapes, validates covariance
inputs, and returns aligned arrays suitable for downstream derivative-based
methods.


LikelihoodKit automatically:

- flattens/reshapes inputs to consistent shapes,
- checks that ``data``, ``model``, and ``cov`` agree in dimension,
- supports identity covariance, diagonal covariance, or full covariance,
- returns both the aligned data vector (for reuse) and the likelihood value.

**Example:**
Worked examples are provided in :doc:`../../examples/likelihoods/gaussian`.


Poissonian Likelihood
---------------------

The **Poissonian likelihood** applies when the observables represent
counts, histograms, or binned number data (see `A. Heavens, 2010. <https://arxiv.org/abs/0906.0664>`_).

Given observed counts ``d`` and model expectation ``λ``:

.. math::

   \ln \mathcal{L}
   = \sum_i \left( d_i \ln \lambda_i - \lambda_i - \ln(d_i!) \right).

This likelihood is appropriate when the data represent discrete counts and
Gaussian approximations are not valid, such as number counts or binned event
rates. The Poisson rate (mean) ``λ`` is supplied via model_parameters and is passed
directly to ``scipy.stats.poisson.pmf`` / ``logpmf``; the function returns the elementwise (log-)PMF
on the aligned data grid.


LikelihoodKit supports:

- scalar or vector-valued rate parameters ``λ``,
- automatic reshaping and alignment of observed counts,
- numerically stable evaluation for small counts.


**Example:**
Worked examples are provided in :doc:`../../examples/likelihoods/poissonian`.


Input Handling and API Contract
-------------------------------

LikelihoodKit defines a consistent interface between user models,
numerical derivatives, and forecasting tools.


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

All likelihoods in LikelihoodKit are designed to be drop-in compatible
with ForecastKit’s Fisher machinery and local likelihood expansion (DALI)
tools.

- Fisher derivatives use the aligned model output from the likelihood layer,
- reshaping logic ensures consistency,
- likelihoods can be swapped without changing forecasting code.

This modularity allows users to begin with simple Gaussian forecasts
and later transition to Poisson likelihood without rewriting their models.


Design Philosophy
-----------------

LikelihoodKit intentionally provides *minimal* likelihood implementations.
Its role is not to replace full probabilistic programming frameworks, but to
offer robust, well-defined likelihood building blocks that integrate cleanly
with derivative-based forecasting and approximation methods.

This design makes it easy to:

- prototype forecasts quickly,
- swap likelihood forms without touching model code,
- transition from forecasts to approximate posteriors.
