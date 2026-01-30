.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Workflows
==================


Common workflows / FAQ
----------------------

This page helps you decide **which DerivKit tool to use** based on your scientific use case.
Each section starts from a concrete question and points you to the appropriate
workflow and examples.

It is intended as a **decision guide**, not a full tutorial.

DerivKit does **not** construct data covariances or define scientific models.
Users are expected to provide:

- their own model mapping parameters to observables
- their own data covariance (or a function returning one)

DerivKit focuses on derivative evaluation, local likelihood approximations,
and fast forecasting utilities built on top of these inputs.

If you are looking for short answers to common questions, you can jump directly
to the :ref:`workflows-faq` section below.


I want Fisher constraints on my parameters
------------------------------------------

**You have**

- a parameter vector ``theta0``
- a model mapping parameters to observables
- a data covariance matrix (or a function returning one)

**We compute**

- the Fisher information matrix
- the Gaussian parameter covariance via inversion
- approximate posterior samples
- GetDist-compatible outputs for visualization

**Use**

- :class:`ForecastKit.fisher`

**Minimal example**

See :doc:`examples/forecasting/fisher` and :doc:`examples/forecasting/fisher_contours`

**Notes**

- This assumes the posterior is approximately Gaussian near ``theta0``.
- If this assumption fails, consider using DALI instead.
- Parameter-dependent covariances are handled automatically.


I expect non-Gaussian posteriors (banana-shaped, skewed, etc.)
--------------------------------------------------------------

**You have**

- a nonlinear model
- parameters where Fisher may underestimate uncertainties
- parameters with physical bounds or informative priors that truncate the Gaussian approximation
- a data covariance matrix

**We compute**

- a DALI expansion up to a chosen order:
  - order 2: Fisher
  - order 3: Fisher + G
  - order 4: Fisher + G + H
- approximate posterior samples
- GetDist-compatible outputs for visualization

**Use**

- :class:`ForecastKit.dali(expansion_order=N)`

**Minimal example**

See :doc:`examples/forecasting/dali` and :doc:`examples/forecasting/dali_contours`

**Notes**

- Choose the expansion order based on how non-Gaussian you expect the posterior
  to be.
- You do not need to manipulate DALI tensors directly.
- Sampling bounds and informative priors can make posteriors non-Gaussian even
  when the forward model is close to linear.
- Fisher/DALI describe the *likelihood* locally; prior truncation effects are only
  captured when you sample with explicit priors.


I already have Fisher / DALI tensors. What do I do next?
--------------------------------------------------------

**You have**

- Fisher ``F`` and optionally higher-order tensors ``G``, ``H``, …

**We compute**

- approximate posterior samples
- GetDist-compatible outputs for visualization

**Use**

- importance sampling (fast)
- ``emcee`` sampling (slower, more robust)

**Minimal example**

See :doc:`examples/forecasting/dali_contours`

**Notes**

- Importance sampling is extremely fast but may fail for strongly non-Gaussian
  posteriors.
- If importance sampling fails, switch to MCMC sampling via ``emcee``.


I have a parameter dependent covariance, can I still use Fisher / DALI?
-----------------------------------------------------------------------

**You have**

- a model with parameter-dependent covariance
- a parameter vector ``theta0``
- a data covariance function

**We compute**

- the Fisher expansion accounting for covariance derivatives
- approximate posterior samples
- GetDist-compatible outputs for visualization

**Use**

- :class:`ForecastKit` with a covariance function input
- Fisher method as usual
- DALI method as usual

**Minimal example**




I want a Gaussian approximation around a MAP
--------------------------------------------

**You have**

- a likelihood or log-posterior
- a maximum a posteriori (MAP) point

**We compute**

- a Laplace (Gaussian) approximation
- an estimate of the local covariance

**Use**

- Laplace approximation utilities

**Minimal example**

See :doc:`examples/forecasting/laplace_approx` and :doc:`examples/forecasting/laplace_contours`

**Notes**

- The Laplace mean is the expansion point (usually the MAP).
- This is a local approximation and may fail for strongly non-Gaussian posteriors.


I want to include priors
------------------------

**You have**

- prior information (bounds, Gaussian priors, correlated priors, etc.)

**We compute**

- log-prior contributions
- posterior sampling with explicit priors

**Use**

- :mod:`PriorKit`

**Minimal example**

See the DALI sampling examples with priors in :doc:`examples/forecasting/dali_contours`

**Notes**

- Priors are applied explicitly by design.
- Sampler bounds truncate the sampled region; informative priors modify the
  posterior shape.


My model is tabulated or expensive to evaluate
----------------------------------------------

**You have**

- samples of a function on a grid
- no analytic expression (or an expensive forward model)

**We compute**

- numerical derivatives from tabulated data

**Use**

- :class:`DerivativeKit` with ``x_tab`` and ``y_tab`` inputs

**Minimal example**

See :doc:`examples/derivatives/tabulated`

**Notes**

- This is especially useful when model evaluations are costly.
- Tabulated models are treated as callables by DerivKit.


I only want numerical derivatives (no forecasting yet)
------------------------------------------------------

**You have**

- a function or model
- a point where derivatives are needed

**We compute**

- first and higher-order derivatives
- gradients, Jacobians, and Hessians

**Use**

- :class:`DerivativeKit`
- :class:`CalculusKit`

**Minimal example**

See :doc:`examples/derivatives/index`


I want Fisher bias / parameter shifts
-------------------------------------

**You have**

- a mismatched model and data-generating process
- a Fisher matrix

**We compute**

- parameter bias induced by model mismatch

**Use**

- Fisher bias utilities in :class:`ForecastKit`

**Minimal example**

See :doc:`examples/forecasting/fisher_bias`


I have many parameters and derivative evaluation is expensive
-------------------------------------------------------------

**You have**

- a model with many parameters
- expensive function evaluations
- concerns about runtime or scaling

**We compute**

- derivatives using parallel execution
- Jacobians and higher-order tensors efficiently

**Use**

- :class:`DerivativeKit` with ``n_workers``
- :class:`ForecastKit` parallel derivative evaluation

**Notes**

- DerivKit parallelizes derivative evaluations across parameters and outputs.
- This is especially useful for large Fisher or DALI expansions.


I want to compare Fisher and DALI forecasts
-------------------------------------------

**You have**

- a Fisher forecast
- a DALI expansion for the same model

**We compute**

- approximate posterior samples from both
- directly comparable contours and summaries

**Use**

- :class:`ForecastKit.fisher`
- :class:`ForecastKit.dali`
- GetDist-based visualization utilities

**Notes**

- This is useful for diagnosing non-Gaussianity.
- Differences indicate where Fisher assumptions break down.


.. _workflows-faq:

FAQ / Frequently asked questions
--------------------------------

**Why does Fisher underestimate my errors?**

Because it assumes the posterior is locally Gaussian. Strong curvature or
parameter degeneracies require higher-order (DALI) terms.


**Why is the Laplace mean equal to my expansion point?**

The Laplace approximation expands around the MAP by construction. It does not
estimate a shifted mean.


**When should I use DALI instead of Fisher?**

When the posterior is visibly non-Gaussian or when Fisher forecasts are known
to be biased.


**How do I choose the DALI expansion order?**

The appropriate expansion order depends on the degree of non-Gaussianity in the
posterior. Order 2 corresponds to the Fisher approximation, while higher orders
capture increasing levels of skewness and curvature. In practice, comparing
results across orders can help diagnose when Fisher assumptions break down.


**Why are priors not included automatically?**

DerivKit separates likelihood information from prior assumptions by design.
This keeps approximations explicit and easier to reason about.


**Can I use DerivKit with MCMC samplers?**

Yes. DerivKit likelihoods and priors can be used with any sampler that accepts
log-posterior functions. We provide examples using ``emcee`` and importance
sampling, but DerivKit is sampler-agnostic and can be integrated with other
sampling frameworks by implementing a thin wrapper around the log-posterior API.


**Does DerivKit assume Gaussian likelihoods?**

No. Fisher and Laplace methods make local Gaussian approximations, while DALI
systematically captures non-Gaussian structure through higher-order terms.


**Where do my model and covariance come from?**

DerivKit is agnostic to how models and covariances are constructed. Users are
expected to supply these based on their scientific application, while DerivKit
provides derivative evaluation and inference utilities built on top of them.


**Can I use my own likelihood with DerivKit?**

Yes. DerivKit is agnostic to how likelihoods are defined. Users can supply their
own likelihood or log-posterior functions, which DerivKit treats as external
inputs for derivative evaluation, local approximations, and sampling.


**Can I use DerivKit within an existing inference pipeline?**

Yes. DerivKit is designed to integrate with externally defined models,
likelihoods, and covariances, and can be used alongside other inference or
sampling frameworks.


**Are derivatives computed analytically or numerically?**

DerivKit computes derivatives numerically using robust finite-difference and
polynomial-based methods. Optional automatic differentiation backends may be
used for validation, but numerical methods are the default and primary focus.


**Where can I find more examples?**

See the :doc:`examples/index` section of the documentation.
Additional extended demos are available at
https://github.com/derivkit/derivkit-demos


**Who do I contact for support?**

Please open an issue on the DerivKit GitHub repository.
Go to :doc:`contributing` for contribution guidelines and support options.
