Forecasting Tools
=================

ForecastKit provides three core forecasting capabilities:
the *Fisher information matrix*, *Fisher bias*, and *DALI* (higher-order) expansions.
These tools all rely on DerivativeKit for stable numerical derivatives and work with any user-provided model.


Fisher Information Matrix
-------------------------

The **Fisher matrix** quantifies how precisely model parameters can be
determined from a set of observables.

Given:

- parameters
  `` theta = (theta_1, theta_2, …)``
- a model mapping parameters to observables
  ``m(theta)``
- a data covariance matrix ``C``

ForecastKit computes the Jacobian
``J[i, a] = d m[i] / d theta[a]`` using DerivativeKit.

The Fisher matrix is:

.. math::

   F = J^\top C^{-1} J.

From the Fisher matrix and its inverse ``F^{-1}``, ForecastKit returns:

- parameter variances
- marginalized errors
- correlation coefficients
- confidence ellipses (1σ, 2σ)
- any sub-block for parameter subsets

**Interpretation:**
The Fisher matrix predicts how tightly your parameters can be constrained,
without running a full MCMC or sampling-based inference.


Fisher Bias
-----------

Real measurements often contain small systematic deviations:

.. math::

   \delta m = m_{\rm true} - m_{\rm model}.

These systematics induce *biases* in the inferred parameters.
ForecastKit computes the first-order Fisher bias vector:

.. math::

   b_\alpha = \sum_{i,j}
   (F^{-1})_{\alpha\beta}\,
   J_{j\beta}^\top C^{-1}_{ji}\, \delta m_i.

The corresponding parameter shift is:

.. math::

   \Delta\theta = -F^{-1} b.

ForecastKit returns:

- the Fisher bias vector ``b``
- the parameter shift ``Δθ``
- optional visualization: Fisher ellipses + bias arrow

**Interpretation:**
Fisher bias tells you how far the best-fit parameters move due to small
systematic errors in the observables.

.. image:: ../assets/plots/fisher_bias_demo_1and2sigma.png


DALI (Higher-Order Forecasting)
-------------------------------

The **DALI** expansion (Derivative Approximation for LIkelihoods; `Sellentin et al. 2014 <https://arxiv.org/abs/1401.6892>`_)
extends the Fisher matrix by including *second* and *third* derivative
information, capturing non-Gaussian structure in the likelihood.

ForecastKit computes higher-order tensors:

- the Hessian of the model
- mixed second derivatives of the log-likelihood
- third derivatives needed for cubic-order DALI terms

This allows you to approximate:

- non-Gaussian likelihoods
- skewed or curved parameter degeneracies
- parameter covariances beyond quadratic order

**Interpretation:**
DALI generalizes the Fisher approach to cases where the likelihood is not
well approximated by a Gaussian, providing more accurate forecast contours.

.. image:: ../assets/plots/dali_vs_fisher_exact_1d.png

.. image:: ../assets/plots/dali_vs_fisher_2d_1and2sigma.png


Backend Notes
-------------

- If ``method`` is omitted, the **adaptive** derivative backend is used.
- Any DerivativeKit backend can be used:
  ``method="finite"``, ``"ridders"``, ``"gauss-richardson"``, ``"polyfit"``, etc.
- To list available derivative methods:

  .. code-block:: python

     from derivkit.derivative_kit import available_methods
     print(available_methods())

``ForecastKit`` is fully modular: changing the derivative backend changes
only how the Jacobian/Hessian are computed, not the forecasting logic itself.
