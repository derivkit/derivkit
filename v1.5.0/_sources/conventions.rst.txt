.. _conventions:

.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Conventions
====================

This page defines the mathematical notation and conventions used
throughout DerivKit, particularly in ForecastKit. These definitions
establish the signs, normalizations, and reference points used in
Fisher forecasts, DALI expansions, and systematic bias calculations.

Parameters and Observables
--------------------------

We denote the model parameters by :math:`\theta` and the model
predictions by :math:`\nu(\theta)`. The fiducial parameter point
is :math:`\theta_0`, and parameter displacements are defined as

.. math::

   \delta^a = \theta^a - \theta_0^a.

Parameter indices are denoted by Latin letters
:math:`a,b,c,\ldots`, while observable indices are denoted by
:math:`i,j,k,\ldots`. Repeated indices are summed unless stated
otherwise.

Model derivatives are written as

.. math::

   \nu_{,a}
   = \left.\frac{\partial\nu}{\partial\theta^a}\right|_{\theta_0},

.. math::

   \nu_{,ab}
   = \left.\frac{\partial^2\nu}
   {\partial\theta^a\partial\theta^b}\right|_{\theta_0}.

Higher-order derivatives follow the same notation.


Likelihood and Posterior
------------------------

For Gaussian data with parameter-independent covariance :math:`C`,
we define

.. math::

   \chi^2(\theta)
   =
   [d-\nu(\theta)]^{\mathrm T}
   C^{-1}
   [d-\nu(\theta)].

The log-likelihood is

.. math::

   \log L(\theta)
   =
   -\frac{1}{2}\chi^2(\theta)
   + \mathrm{const}.

Given a prior density :math:`\pi(\theta)`, the log-posterior is

.. math::

   \log p(\theta\mid d)
   =
   \log L(\theta)
   + \log\pi(\theta)
   + \mathrm{const}.

Thus, positive contributions to :math:`\chi^2` enter the
log-likelihood with a negative factor of :math:`1/2`.

The negative log-posterior is

.. math::

   \Phi(\theta)
   =
   -\log p(\theta\mid d).

Maximizing the log-posterior is equivalent to minimizing
:math:`\Phi`.


Forecast Tensors
----------------

ForecastKit constructs Fisher and DALI tensors from derivatives
of the model predictions, evaluated at :math:`\theta_0`.

The Fisher matrix is

.. math::

   F_{ab}
   =
   \nu_{,a}^{\mathrm T}C^{-1}\nu_{,b}.

The doublet DALI tensors are

.. math::

   D^{(1)}_{abc}
   =
   \nu_{,ab}^{\mathrm T}C^{-1}\nu_{,c},

.. math::

   D^{(2)}_{abcd}
   =
   \nu_{,ab}^{\mathrm T}C^{-1}\nu_{,cd}.

The triplet DALI tensors are

.. math::

   T^{(1)}_{abcd}
   =
   \nu_{,abc}^{\mathrm T}C^{-1}\nu_{,d},

.. math::

   T^{(2)}_{abcde}
   =
   \nu_{,abc}^{\mathrm T}C^{-1}\nu_{,de},

.. math::

   T^{(3)}_{abcdef}
   =
   \nu_{,abc}^{\mathrm T}C^{-1}\nu_{,def}.

Tensor definitions do not include Taylor factorial coefficients.
These coefficients are applied when evaluating the expansion.

Tensor contractions are understood to be symmetrized over the
parameter indices where appropriate.


DALI Expansion
--------------

For expected data at the fiducial model, define the change in
chi-squared relative to :math:`\theta_0` as

.. math::

   \Delta\chi^2(\theta)
   =
   \chi^2(\theta)-\chi^2(\theta_0).

The doublet DALI approximation is

.. math::

   \Delta\chi^2_{\mathrm DALI}
   =
   F_{ab}\delta^a\delta^b
   + D^{(1)}_{abc}\delta^a\delta^b\delta^c
   + \frac{1}{4}
     D^{(2)}_{abcd}\delta^a\delta^b\delta^c\delta^d.

Consequently,

.. math::

   \log L_{\mathrm DALI}(\theta)
   =
   \mathrm{const}
   -\frac{1}{2}F_{ab}\delta^a\delta^b
   -\frac{1}{2}D^{(1)}_{abc}\delta^a\delta^b\delta^c
   -\frac{1}{8}D^{(2)}_{abcd}\delta^a\delta^b\delta^c\delta^d.

The negative signs arise from the definition of the
Gaussian log-likelihood, not from the tensor definitions.

DALI truncates the Taylor expansion of the model predictions
before constructing the squared residual. It is therefore
not equivalent to truncating a Taylor series of the
log-likelihood at the same polynomial degree.

The Fisher, doublet, triplet, and quadruplet approximations
retain model derivatives through first, second, third, and
fourth order, respectively.


Systematic Mismatches and Bias Tensors
--------------------------------------

A systematic data mismatch is defined as

.. math::

   \Delta\nu
   =
   \langle d\rangle-\nu(\theta_0).

This convention always represents expected data minus the
model prediction at the fiducial parameters.

The corresponding bias tensors are

.. math::

   b_{a_1\ldots a_n}
   =
   \Delta\nu^{\mathrm T}
   C^{-1}
   \nu_{,a_1\ldots a_n}.

The first-order bias vector is

.. math::

   b_a
   =
   \Delta\nu^{\mathrm T}C^{-1}\nu_{,a}.

With this sign convention, the Fisher parameter bias is

.. math::

   \Delta\theta_{\mathrm F}^a
   =
   (F^{-1})^{ab}b_b.

The mismatch contribution to the log-likelihood is

.. math::

   B(\delta)
   =
   b_a\delta^a
   + \frac{1}{2}b_{ab}\delta^a\delta^b
   + \frac{1}{3!}b_{abc}\delta^a\delta^b\delta^c
   + \cdots.

Reversing the sign of :math:`\Delta\nu` reverses the
first-order Fisher bias. Higher-order DALI responses need
not reverse in the same way.


Parameter Displacements and Systematic Shifts
---------------------------------------------

A parameter displacement is measured relative to the
fiducial point:

.. math::

   \Delta\theta^{\mathrm{disp}}(A)
   =
   \theta(A)-\theta_0.

A systematic shift is measured relative to the corresponding
estimator in the absence of the mismatch:

.. math::

   \Delta\theta^{\mathrm{sys}}(A)
   =
   \theta(A)-\theta(0).

These quantities are not necessarily equal.

For a posterior mean, the systematic shift is

.. math::

   \Delta\bar{\theta}^{\mathrm{sys}}(A)
   =
   \bar{\theta}(A)-\bar{\theta}(0).

In a non-Gaussian posterior, the unbiased mean need not
coincide with the fiducial parameters:

.. math::

   \bar{\theta}(0)\neq\theta_0.

The same distinction applies to the posterior maximum when
the unbiased MAP differs from :math:`\theta_0`.


MAP and Posterior Mean
----------------------

The maximum a posteriori estimate is

.. math::

   \hat{\theta}
   =
   \operatorname*{arg\,max}_{\theta}
   p(\theta\mid d).

The posterior mean is

.. math::

   \bar{\theta}^a
   =
   \int \theta^a p(\theta\mid d)
   \,\mathrm{d}^{N_{\mathrm{par}}}\theta.

These estimators generally differ for non-Gaussian posteriors.
ForecastKit therefore distinguishes MAP shifts from posterior
mean shifts.

Unless explicitly stated otherwise, a reported systematic
shift refers to the change in the specified estimator
relative to its unbiased value.


Scope of the Conventions
------------------------

The Fisher and DALI tensor expressions above assume a
Gaussian likelihood with parameter-independent covariance.

ForecastKit also supports generalized Fisher formalisms for
parameter-dependent covariance and other likelihood models.
Their additional terms and assumptions are documented in
the corresponding method descriptions.

Priors are included separately from the likelihood
expansion unless explicitly incorporated into the
forecasting calculation.
