.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: ``DerivKit`` logo black
   :width: 32px


|dklogo| Workflows
==================

This page helps you decide **which DerivKit tool to use** based on your
scientific use case. Each section starts from a concrete question and points
you to the appropriate workflow and examples.

It is intended as a **decision guide**, not a full tutorial.

Using ``DerivKit`` effectively requires an understanding of the assumptions,
strengths, and limitations of the chosen method. Fisher, DALI, Laplace, and
sampling-based approaches make different approximations and are suited to
different inference regimes. ``DerivKit`` does not guarantee the validity of any
approximation for a given problem; assessing whether a method is appropriate
for a specific model, parameterization, prior choice, and scientific goal
requires scientific judgment from the user.

``DerivKit`` does **not** construct data covariances or define scientific models.
Users are expected to provide:

- their own model mapping parameters to observables
- their own data covariance (or a function returning one)

``DerivKit`` focuses on derivative evaluation, local likelihood approximations,
and fast forecasting utilities built on top of these inputs.

If you are looking for short answers to common questions, you can jump directly
to the :ref:`workflows-faq` section below.
Some common mistakes are discussed in the :ref:`workflows-mistakes` section.


Quick decision guide
--------------------

This table provides a fast, high-level guide for choosing a numerical
differentiation strategy in DerivKit. It is intended as a quick reference;
for detailed workflows and examples, see the sections below.



.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - **Situation**
     - **Recommended method**
     - **Why**
   * - Smooth, cheap function
     - Finite differences
     - Fast and accurate for smooth functions
   * - Slightly noisy function
     - Ridders finite differences
     - Richardson extrapolation improves stability over simple finite differences
   * - Moderate or structured noise
     - Local polynomial fit
     - Local regression smooths noise better than finite differences
   * - High noise / messy signal
     - Adaptive polynomial fit (Chebyshev)
     - Robust trimming, Chebyshev grid, and fit diagnostics
   * - Expensive function
     - Adaptive polynomial fit (Chebyshev)
     - Achieves stable derivatives with fewer function evaluations near ``x0``
   * - Need robustness and diagnostics
     - Adaptive polynomial fit (Chebyshev)
     - Provides fit quality metrics, degree adjustment, and suggestions
   * - Unsure / first attempt
     - Local polynomial fit
     - Good default when function behavior is not well known


Choose a workflow
-----------------

I want Fisher constraints on my parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- a parameter vector ``theta0``
- a model mapping parameters to observables
- a data covariance matrix (or a function returning one)

**We compute**

- the Fisher information matrix
- the Gaussian parameter covariance via inversion
- approximate posterior samples
- :class:`GetDist`-compatible outputs for visualization

**Use**

- :class:`ForecastKit.fisher`

**Minimal example**

See :doc:`examples/forecasting/fisher` and :doc:`examples/forecasting/fisher_contours`

**Notes**

- This assumes the posterior is approximately Gaussian near ``theta0``.
- If this assumption fails, consider using DALI instead.
- Parameter-dependent covariances are handled automatically.


I expect non-Gaussian posteriors (banana-shaped, skewed, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- a nonlinear model
- parameters where Fisher may underestimate uncertainties
- parameters with physical bounds or informative priors that truncate the
  Gaussian approximation
- a data covariance matrix

**We compute**

- a DALI expansion up to a chosen order:
  - order 1: Fisher matrix
  - order 2: doublet DALI tensors (D1 and D2)
  - order 3: triplet DALI tensors (T1, T2, and T3)
- approximate posterior samples
- ``GetDist``-compatible outputs for visualization

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
- Fisher/DALI describe the *likelihood* locally; prior truncation effects are
  only captured when you sample with explicit priors.


I already have Fisher matrix / DALI tensors. What do I do next?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- Fisher ``F`` and optionally higher-order tensors ``D1``, ``D2``, …

**We compute**

- approximate posterior samples
- ``GetDist``-compatible outputs for visualization

**Use**

- importance sampling (fast)
- ``emcee`` sampling (slower, more robust)

**Minimal example**

See :doc:`examples/forecasting/dali_contours`

**Notes**

- Importance sampling is extremely fast but may fail for strongly non-Gaussian
  posteriors.
- If importance sampling fails, switch to MCMC sampling via ``emcee``.


I want a Gaussian approximation around a MAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
- Tabulated models are treated as callables by ``DerivKit``.


I only want numerical derivatives (no forecasting yet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

**Notes**

- Use :class:`CalculusKit` when you want direct access to gradients/Hessians.
- Use :class:`DerivativeKit` for higher-level derivative workflows and diagnostics.



I want Fisher bias / parameter shifts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- a mismatched model and data-generating process
- a Fisher matrix

**We compute**

- parameter bias induced by model mismatch

**Use**

- Fisher bias utilities in :class:`ForecastKit`

**Minimal example**

See :doc:`examples/forecasting/fisher_bias`


My covariance depends on parameters. What do I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- a model mapping parameters to observables, and
- a covariance that depends on parameters, :math:`C(\theta)`, **or**
- noisy inputs and outputs with a block covariance (the X–Y case)

**We compute**

- a **Gaussian Fisher matrix** that includes covariance-derivative terms when
  :math:`C(\theta)` depends on the parameters
- (optional) X–Y Gaussian Fisher constraints where input uncertainties are
  propagated into an effective output covariance

**Use**

- :meth:`ForecastKit.gaussian_fisher` for parameter-dependent covariances
- :func:`derivkit.forecasting.fisher_xy.build_xy_gaussian_fisher_matrix` for the
  X–Y Gaussian case (noisy inputs *and* outputs)

**Minimal examples**

See :doc:`examples/forecasting/fisher_gauss` and
:doc:`examples/forecasting/fisher_xy`.

**Notes**

- :meth:`ForecastKit.fisher` uses a fixed covariance :math:`C(\theta_0)` and
  computes only the mean-derivative term.
- :meth:`ForecastKit.gaussian_fisher` adds the covariance-derivative contribution
  when a callable covariance :math:`C(\theta)` is provided.
- DALI currently assumes a fixed covariance evaluated at :math:`\theta_0`;
  parameter-dependent covariance support for DALI is planned.


I have many parameters and derivative evaluation is expensive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

- ``DerivKit`` parallelizes derivative evaluations across parameters and outputs.
- This is especially useful for large Fisher or DALI expansions.


I want to compare Fisher and DALI forecasts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**You have**

- a Fisher forecast
- a DALI expansion for the same model

**We compute**

- approximate posterior samples from both
- directly comparable contours and summaries

**Use**

- :class:`ForecastKit.fisher`
- :class:`ForecastKit.dali`
- ``GetDist``-based visualization utilities

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

``DerivKit`` separates likelihood information from prior assumptions by design.
This keeps approximations explicit and easier to reason about.


**Can I use DerivKit with MCMC samplers?**

Yes. ``DerivKit`` likelihoods and priors can be used with any sampler that accepts
log-posterior functions. We provide examples using ``emcee`` and importance
sampling, but ``DerivKit`` is sampler-agnostic and can be integrated with other
sampling frameworks by implementing a thin wrapper around the log-posterior API.


**Does DerivKit assume Gaussian likelihoods?**

No. Fisher and Laplace methods make local Gaussian approximations, while DALI
systematically captures non-Gaussian structure through higher-order terms.


**My DALI doublet and triplet contours look identical. Is something wrong?**

Usually not. Triplet DALI only modifies the posterior where higher-order
corrections are non-negligible at the typical posterior radius. If most of the
posterior mass lies close to the expansion point ``theta0`` or if you are using
importance sampling, higher-order terms can be strongly suppressed and contours
may look identical. Remember that DALI is a local expansion: it captures local
skewness and curvature, but cannot reproduce global structure or multiple modes.
For triplet DALI to show significant differences from doublet DALI, the posterior
must extend far enough from ``theta0`` for cubic terms to become important. We
recommend always using emcee sampling for triplet DALI to fully capture its
effects. If in doubt, compare results across expansion orders and sampling
methods.


**Why does DALI behave poorly far from the expansion point?**

DALI is based on a Taylor expansion and is only expected to be accurate within a
finite neighborhood around ``theta0``. Far from this point, higher-order terms
can dominate and lead to unphysical behavior. For this reason, sampling bounds
and diagnostic checks are strongly recommended.


**Should I always use the highest DALI expansion order available?**

No. Higher-order expansions are more expensive and not always informative.
If increasing the expansion order does not change posterior summaries, lower
orders are usually sufficient and preferable for robustness.


**Where do my model and covariance come from?**

``DerivKit`` is agnostic to how models and covariances are constructed. Users are
expected to supply these based on their scientific application, while ``DerivKit``
provides derivative evaluation and inference utilities.


**Can I use my own likelihood with DerivKit?**

Yes. ``DerivKit`` is agnostic to how likelihoods are defined. Users can supply their
own likelihood or log-posterior functions, which ``DerivKit`` treats as external
inputs for derivative evaluation, local approximations, and sampling.


**Can I use ``DerivKit`` within an existing inference pipeline?**

Yes. ``DerivKit`` is designed to integrate with externally defined models,
likelihoods, and covariances, and can be used alongside other inference or
sampling frameworks.


**Are derivatives computed analytically or numerically?**

``DerivKit`` computes derivatives numerically using robust finite-difference and
polynomial-based methods. Optional automatic differentiation backends may be
used for validation, but numerical methods are the default and primary focus.


**Where can I find more examples?**

See the :doc:`examples/index` section of the documentation.
Additional extended demos are available at
https://github.com/derivkit/derivkit-demos


**Who do I contact for support?**

Please open an issue on the ``DerivKit`` GitHub repository.
Go to :doc:`contributing` for contribution guidelines and support options.


.. _workflows-mistakes:

Common mistakes
---------------


**Using local approximations for global inference**

Fisher, DALI, and Laplace are local approximations around a chosen expansion
point. As such, they may not reliably recover global posterior structure, multiple
modes, or long nonlocal tails beyond the radius of convergence of the local
expansion. If your inference problem is strongly nonlocal,
full MCMC or nested sampling is required.

**Expanding around a poorly chosen expansion point**

All local methods assume that ``theta0`` (or the MAP for Laplace) is close to the
region of highest posterior support. Expanding far from the true posterior peak
can lead to misleading forecasts or unstable higher-order corrections.


**Relying on importance sampling for strongly non-Gaussian posteriors**

Importance sampling is fast but fragile. For curved, skewed, or bounded
posteriors it can severely underestimate higher-order effects. In these cases,
use MCMC sampling (e.g. ``emcee``) instead.


**Assuming higher-order expansions always improve results**

Increasing the DALI expansion order does not guarantee better accuracy. If
higher-order terms are numerically small where the posterior mass lies, results
may be unchanged. Unnecessary higher-order terms can also reduce robustness.


**Ignoring sampling bounds and priors**

Sampling without explicit bounds or priors can lead to unphysical regions where
local approximations break down. Always include realistic bounds or priors when
sampling Fisher, DALI, or Laplace posteriors.


**Expecting priors to be applied automatically**

``DerivKit`` treats likelihood information and priors separately by design. Priors
must be applied explicitly during sampling; they are not combined with Fisher,
DALI, or Laplace objects automatically.


**Over-interpreting Laplace approximations**

The Laplace approximation provides a Gaussian approximation around the MAP. It
does not capture skewness, curvature, or truncation effects and should not be
used for strongly non-Gaussian posteriors.


**Using very tight data covariances without diagnostics**

When the posterior is extremely narrow, higher-order corrections can be
numerically suppressed and indistinguishable from Fisher. Always check the
typical posterior radius relative to the expansion point.


**Assuming numerical derivatives are exact**

All derivatives in ``DerivKit`` are computed numerically. Poorly scaled parameters,
discontinuous models, or insufficient step-size control can degrade derivative
accuracy. Diagnostic checks are recommended for sensitive applications.


**Assuming automatic differentiation is always correct**

Automatic differentiation (e.g. JAX) can silently fail for non-smooth models,
conditionals, or interpolations; derivative validation is still recommended,
especially for higher-order forecasts.


**Confusing likelihood curvature with posterior uncertainty**

Fisher, DALI, and Laplace characterize the local curvature of the likelihood.
Posterior uncertainties can differ significantly once priors, bounds, or
nonlinear transformations are applied. Always interpret results in the context
of the full posterior definition.


**Using overly informative priors without realizing it**

Strong priors can dominate posterior constraints and mask the information
content of the likelihood. When comparing Fisher, DALI, or Laplace results,
check whether constraints are prior-dominated rather than data-driven.


**Interpreting Fisher or DALI contours as exact confidence regions**

Contours produced from Fisher or DALI approximations are not guaranteed to have
exact frequentist or Bayesian coverage. They should be interpreted as
approximate summaries, not as precise confidence regions.


**Mixing Bayesian and frequentist interpretations**

Fisher matrices originate from frequentist theory, while posterior sampling and
priors are Bayesian concepts. Mixing interpretations without care can lead to
incorrect conclusions about parameter uncertainties or coverage.


**Ignoring parameter reparameterization effects**

Local approximations depend on the chosen parameterization. Strong nonlinear
transformations can change the apparent degree of non-Gaussianity and affect
Fisher or DALI performance. Reparameterization may improve stability and
interpretability.
