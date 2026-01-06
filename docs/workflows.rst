Workflows
=========

Common workflows / FAQ
----------------------

This page helps you decide **which DerivKit tool to use** based on your problem.
Each section starts from a concrete question and points you to the appropriate
workflow and examples.

If you already know exactly what you want, you can skip this page and go
directly to the relevant section (Fisher, DALI, Laplace, etc.).


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
- Parameter-dependent covariances are handled automatically.


I expect non-Gaussian posteriors (banana-shaped, skewed, etc.)
--------------------------------------------------------------

**You have**

- a nonlinear model
- parameters where Fisher may underestimate uncertainties
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

**Why are priors not included automatically?**
DerivKit separates likelihood information from prior assumptions by design.
This keeps approximations explicit and easier to reason about.

**Can I use DerivKit with MCMC samplers?**
Yes. DerivKit likelihoods and priors can be used with any sampler that accepts
log-posterior functions.

**Where can I find more examples?**
See the :doc:`examples/index` section of the documentation.
Additional extended demos are available at
https://github.com/derivkit/derivkit-demos

**Who do I contact for support?**
Please open an issue on the DerivKit GitHub repository.
