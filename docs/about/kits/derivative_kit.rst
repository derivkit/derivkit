DerivativeKit
=============

DerivKit implements several complementary derivative engines. Each has different
strengths depending on smoothness, noise level, and computational cost.

This page gives an overview of the main methods featured in ``DerivativeKit``,
how they work, and when to use which one.

All methods are accessed through the same DerivativeKit interface and can be swapped
without changing downstream code.

Runnable examples illustrating these methods are collected in
:doc:`../examples/index`.


Finite Differences
------------------

**Idea:**

Estimate derivatives by evaluating the function at points around ``x0`` and
combining them into a central-difference stencil [#fdiff]_.

In general, a central finite-difference approximation to the first derivative
can be written as a weighted sum over function evaluations at symmetric offsets
around ``x0``:

.. math::

   f'(x_0) \approx \frac{1}{h} \sum_{k=-m}^{m} c_k\, f(x_0 + k h),

where ``h`` is the step size (the spacing between adjacent stencil points),
and the stencil coefficients ``c_k`` satisfy symmetry conditions
(``c_{-k} = -c_k``) and are chosen to cancel low-order truncation errors.
The integer ``m`` determines the stencil width (e.g. ``m=1`` for a 3-point
stencil, ``m=2`` for a 5-point stencil). The coefficients ``c_k`` are computed
using standard algorithms (such as Fornberg’s method), which construct
finite-difference weights by enforcing the desired order of accuracy for a given stencil.

Higher-order stencils improve accuracy by cancelling additional error terms,
at the cost of more function evaluations.


**Features:**

- 3, 5, 7, 9-point central stencils
- Richardson extrapolation [#richardsonwiki]_ (reduces truncation error)
- Ridders extrapolation [#ridderswiki]_ (adaptive error control)
- Gauss–Richardson (GRE) [#gre]_ (noise-robust variant)


**Use when**:

- The function is smooth and cheap to evaluate
- Noise is low or moderate
- You want fast derivatives with minimal overhead


**Avoid when**:

- The function is noisy or discontinuous
- Step-size tuning is difficult or unstable
- Function evaluations are expensive


**Examples:**

A basic finite-difference example is shown in
:doc:`../examples/derivatives/finite_differences`.


Simple Polynomial Fit
---------------------

**Idea:**

Sample points in a small, user-controlled window around ``x0`` and fit a
fixed-order polynomial (e.g. quadratic or cubic) on a simple grid. The
derivative at ``x0`` is taken from the analytic derivative of the fitted polynomial
[#fornberg]_.

The expression for a first derivative from a centered quadratic fit

.. math::

   p(x) = a_0 + a_1 (x - x_0) + a_2 (x - x_0)^2

is:

.. math::

   p'(x_0) = a_1,

where ``a_1`` is the fitted linear coefficient of the polynomial.


**Features:**

- User-chosen window and polynomial degree
- Low overhead and easy to reason about
- Includes diagnostics on fit quality and conditioning


**Use when**:

- The function is smooth but mildly noisy
- You want a simple, local smoothing method
- A fixed window and polynomial degree are sufficient


**Avoid when**:

- Noise is strong or highly irregular
- The fit becomes ill-conditioned in the chosen window


**Examples:**

A basic polyfit example is shown in
:doc:`../examples/derivatives/local_poly`.


Adaptive Polynomial Fit
-----------------------

**Idea:**

Build a Chebyshev-spaced grid around ``x0`` (optionally domain-aware), rescale
offsets to a stable interval, and fit a local polynomial with optional ridge
regularisation. The method can enlarge the grid if there are too few samples,
adjust the effective polynomial degree, and reports detailed diagnostics
[#fornberg]_.

For a centered polynomial fit of degree ``d``,

.. math::

   p(x) = \sum_{k=0}^{d} a_k (x - x_0)^k,

the first derivative is

.. math::

   p'(x) = \sum_{k=1}^{d} k\, a_k (x - x_0)^{k-1},

and in particular

.. math::

   p'(x_0) = a_1,

where ``a_k`` are the fitted polynomial coefficients.


**Sampling strategy**

- Default: symmetric Chebyshev nodes around ``x0`` with automatic half-width
  (via ``spacing="auto"`` and ``base_abs``)
- Domain-aware: interval is clipped to stay inside a given ``(lo, hi)`` domain
- Custom grids: user can supply explicit offsets or absolute sample locations

Below is a visual example of the :py:mod:`derivkit.adaptive_fit` module estimating
the first derivative of a nonlinear function in the presence of noise. The method
selectively discards outlier points before fitting a polynomial, resulting in a
robust and smooth estimate.

.. image:: ../../assets/plots/adaptive_demo_linear_noisy_order1.png


**Stability / diagnostics**

- Scales offsets before fitting to reduce conditioning
- Optional ridge term to stabilise ill-conditioned Vandermonde systems
- Checks fit quality and flags “obviously bad” derivatives with suggestions
- Optional diagnostics dict with sampled points, fit metrics, and metadata


**Use when**:

- The function is noisy, irregular, or numerically delicate
- Finite differences or simple polyfits fail
- You want diagnostics to understand derivative quality


**Avoid when**:

- The function is extremely smooth and cheap to evaluate
- Minimal overhead is a priority
- You want a fully adaptive, robust method


**Examples:**

A basic adaptive polyfit example is shown in
:doc:`../examples/derivatives/adaptive_fit`.


Tabulated Functions
-------------------

**Idea:**

When the target function is provided as tabulated data ``(x, y)``, DerivKit first
wraps the table in a lightweight interpolator and then applies any of the
available derivative engines to the interpolated function.

Internally, tabulated data are represented by
:class:`derivkit.tabulated_model.one_d.Tabulated1DModel`, which exposes a callable
interface compatible with all derivative methods.


**Features:**

- Supports scalar-, vector-, and tensor-valued tabulated outputs
- Uses fast linear interpolation via ``numpy.interp``
- Seamlessly integrates with finite differences, adaptive fit, local polynomial,
  and Fornberg methods
- Identical API to function-based differentiation via :class:`DerivativeKit`


**Use when:**

- The function is only available as discrete samples
- Evaluating the function on demand is expensive or impossible
- You want to reuse DerivKit’s derivative engines on interpolated data


**Avoid when:**

- The tabulation is too coarse to resolve derivatives
- The function has sharp features not captured by interpolation
- Exact derivatives are required


**Examples:**

See :doc:`../examples/derivatives/tabulated_functions` for a basic example using
tabulated data.


JAX Autodiff
------------

**Idea:**

Use JAX’s automatic differentiation to compute exact derivatives of
Python-defined functions that are compatible with ``jax.numpy``.

This functionality is exposed for convenience and experimentation and is
*not* registered as a standard DerivKit derivative method by default.


**Features:**

- Exact derivatives via reverse- and forward-mode autodiff
- No step-size tuning or numerical differencing
- Useful for quick sanity checks against numerical methods


**Use when:**

- The function is analytic and fully JAX-compatible
- You want an exact reference derivative for validation
- You are experimenting or debugging numerical methods


**Avoid when:**

- The function is noisy, tabulated, or a black box
- You need production robustness or broad applicability
- JAX compatibility cannot be guaranteed

In most scientific workflows targeted by DerivKit, the adaptive polynomial fit
or finite-difference methods above are more appropriate.

Installation details are described in :doc:`installation`.
We do not recommend JAX for production use within DerivKit at this time.


References
----------

.. [#fdiff] Wikipedia, *Numerical differentiation – Finite differences*,
   <https://en.wikipedia.org/wiki/Numerical_differentiation#Finite_differences>
.. [#fornberg] B. Fornberg,
   *High-Accuracy Finite Difference Methods*, Cambridge University Press, 2025.
   https://www.cambridge.org/core/books/highaccuracy-finite-difference-methods/F894ADC234A8CCB286DE8C8B43B1E2AA
.. [#gre] C. J. Oates, T. Karvonen, A. L. Teckentrup, M. Strocchi, and S. A. Niederer,
   "Probabilistic Richardson Extrapolation",
   *arXiv:2401.07562*, 2024. https://arxiv.org/abs/2401.07562
.. [#richardsonwiki] Wikipedia,
   "Richardson extrapolation",
   https://en.wikipedia.org/wiki/Richardson_extrapolation
.. [#ridderswiki] Wikipedia,
   "Ridders' method",
   https://en.wikipedia.org/wiki/Ridders%27_method
