---
title: "`DerivKit`: stable numerical derivatives bridging Fisher forecasts and MCMC"
tags:
authors:
  - name: Nikolina Šarčević
    orcid: 0000-0001-7301-6415
    corresponding: true
    affiliation: 1
  - name: Matthijs van der Wild
    orcid: 0000-0002-3949-3063
    affiliation: 2
  - name: Cynthia Trendafilova
    orcid: 0000-0001-5500-4058
    affiliation: 3
affiliations:
  - index: 1
    name: |
        Department of Physics,
        Duke University,
        Science Dr,
        Durham,
        NC 27710,
        USA
  - index: 2
    name: |
        Department of Physics,
        Durham University,
        Lower Mountjoy,
        South Rd,
        Durham DH1 3LE,
        UK
  - index: 3
    name: |
        Center for AstroPhysical Surveys,
        National Center for Supercomputing Applications,
        University of Illinois Urbana-Champaign,
        Urbana,
        IL 61801,
        USA
bibliography: derivkit.bib
---

# Summary

`DerivKit` is a Python package for derivative-based statistical inference.
It implements stable numerical differentiation and derivative assembly utilities for Fisher-matrix forecasting and higher-order likelihood approximations in scientific applications, supporting scalar- and vector-valued models including black-box or tabulated functions where automatic differentiation is impractical or unavailable. 
These derivatives are used to construct Fisher forecasts, Fisher bias estimates, and non-Gaussian likelihood expansions based on the Derivative Approximation for Likelihoods (DALI).
By extending derivative-based inference beyond the Gaussian approximation, `DerivKit` forms a practical bridge between fast Fisher forecasts and more
computationally intensive sampling-based methods such as Markov chain Monte Carlo (MCMC).

# Statement of need

Reliable numerical derivatives of model predictions with respect to parameters
are a core requirement in many areas of scientific computing, including cosmology, particle physics, and climate science.
In practice, such derivatives are often estimated using fixed-step finite differences even when model evaluations are noisy, irregular, expensive, or available only in tabulated form, leading to fragile or irreproducible inference results.
A representative example can be found in cosmology, where forecasting methods based on the Fisher information matrix [@Fisher:1920;@Fisher:1922saa;@Fisher:1925;@Fisher:1935] are popular due to their computational efficiency.
Fisher-based methods rely on first-order derivatives of model predictions with respect to parameters.

These frameworks assume Gaussian posteriors and local linear parameter dependence that can break down in nonlinear regimes [@Tegmark_1997;@heavens2010statisticaltechniquescosmology].
Sampling-based methods such as Markov chain Monte Carlo (MCMC), grid sampling, and nested sampling provide robust posterior estimates but scale poorly with dimensionality and model complexity [@Christensen:2001gj;@Lewis:2002ah;@Audren:2012wb;@Lewis:2013hha;@Tegmark:2000db;@Skilling:2004;@Skilling:2006gxv;@Feroz:2007kg;@Feroz:2008xx;@Trotta_2008;@Verde_2010].
The Derivative Approximation for Likelihoods (DALI)
[@Sellentin_2014;@Sellentin:2015axa] extends Fisher formalism by incorporating higher-order derivatives to capture leading non-Gaussian features of the posterior.
Despite its conceptual appeal, DALI has seen limited practical adoption, in part due to the lack of general-purpose software and the numerical challenges associated with computing stable higher-order derivatives.

Automatic differentiation (`autodiff`) provides exact derivatives for differentiable programs, but is not directly applicable to tabulated models, legacy simulation suites, or workflows involving discontinuities or implicit solvers.
As a result, finite-difference methods remain widely used in scientific software, despite their sensitivity to numerical noise and algorithmic tuning.

`DerivKit` addresses these challenges by providing a diagnostics-driven framework for computing stable numerical derivatives from general numerical model evaluations, without requiring model rewrites or specialized `autodiff` frameworks.
While cosmological forecasting serves as a primary motivating example, `DerivKit` is broadly applicable to inference problems requiring robust derivative-based sensitivity analyses across scientific domains.

# Core Functionality

`DerivKit` is organized into modular components ("kits") that support derivative-based inference workflows, from numerical differentiation to forecasting and likelihood approximations.

## DerivativeKit: Numerical derivative engines

At the core of the library lies _DerivativeKit_, which provides several numerical differentiation strategies that can be selected or combined depending on the numerical properties of the function under consideration.

### Finite-difference derivatives

`DerivKit` implements high-order central finite-difference derivatives using 3-, 5-, 7-, and 9-point stencils, supporting derivative orders one through four.
Accuracy and robustness are improved through extrapolation and stabilization techniques such as Richardson extrapolation [@10.1098/rsta.1911.0009;@10.1098/rsta.1927.0008], Ridders’ method [@Ridders], and noise-robust Gauss--Richardson schemes [@oates2024probabilisticrichardsonextrapolation].
These methods are computationally efficient and well suited to smooth models with low to moderate numerical noise.


### Polynomial fitting

For noisy or numerically stiff models, `DerivKit` provides local polynomial-fit
derivatives [@Camera_2016;@Euclid2020;@Bonici_2023;@niko_jmas;@Fornberg_2025].
Two variants are available: a fixed-window polynomial fit, and an adaptive
polynomial-fit method.
The adaptive method constructs domain-aware Chebyshev sampling grids with
automatically chosen scales, applies internal offset scaling and optional ridge
regularization, and may adjust the polynomial degree to improve conditioning.
It reports diagnostics and warns when internal fit-quality checks indicate
unreliable derivative estimates.
\autoref{adaptive-demo} shows that, in the presence of noise ($\sigma = 0.2$), adaptive-fit differentiation yields accurate and precise derivative estimates, whereas \autoref{adaptive-finite} shows that estimator variance limits the reliability of standard finite-difference schemes.

![Adaptive-fit derivative estimation from noisy samples $y=f(x)+\varepsilon$ evaluated near $x_0$. A local polynomial fit is used to estimate the derivative at $x_0$ (blue line); the dashed red curve shows the noise-free function $f(x)$.\label{adaptive-demo}](figures/adaptive_demo.pdf)

![Distribution of first-derivative estimates at fixed $x_0$ from repeated noise realizations. Adaptive-fit and finite-difference stencil methods are compared; vertical lines indicate medians.\label{adaptive-finite}](figures/hist_adaptive_vs_finite.pdf)


## CalculusKit: Calculus utilities

`DerivKit` provides _CalculusKit_, which builds on the numerical derivative engines.
It constructs common quantities composed of derivatives such as gradients, Jacobians, Hessians, and higher-order derivative tensors for scalar- and vector-valued models.
These utilities provide a consistent interface for assembling derivative structures while delegating numerical differentiation to the underlying derivative engines.

## ForecastKit: Forecasting and likelihood expansions

The _ForecastKit_ uses derivative and calculus utilities to construct common forecasting quantities including Fisher matrices, Fisher bias estimates, and non-Gaussian likelihood expansions.
In particular, `DerivKit` provides utilities to assemble the derivative tensors required for the Derivative Approximation for Likelihoods (DALI), enabling practical extensions beyond the Gaussian Fisher-matrix forecasts and direct connections to downstream tasks such as likelihood evaluation, sampling, and visualization.
Some of these functionalities are illustrated in \autoref{fig:forecastkit_demo}.


## LikelihoodKit: Likelihood models

`DerivKit` includes a lightweight _LikelihoodKit_ that provides basic Gaussian and Poisson likelihood models.
While not intended as a full probabilistic programming framework, these implementations support testing, validation, and end-to-end examples within derivative-based inference workflows.

## Diagnostics and testing

`DerivKit` facilitates user diagnostics by reporting metadata describing sampling geometry, fit quality, and internal consistency checks.
It emits explicit warnings or fallback strategies when tolerance criteria are not met.

All components are accompanied by extensive unit tests to ensure consistency across derivative methods, calculus utilities, and inference workflows, which is particularly important for numerical differentiation and higher-order derivative handling.

![Examples of _ForecastKit_ functionality. Panels show standard Fisher forecasts (with and without priors $\mathcal P$). Standard Fisher contours with and without a Gaussian prior (top left). $X$--$Y$ Fisher contours accounting for uncertainty in both inputs $x$ and outputs $y$ (top right), including uncertainty in both inputs and outputs. Fisher bias, showing the parameter shift induced by a biased data vector (bottom left). DALI triplet contours compared to the posterior from MCMC (`emcee`). Fisher bias estimates, and DALI-based non-Gaussian likelihood approximations.\label{fig:forecastkit_demo}](figures/fig2_panel_2x2.pdf)

# Usage Examples

This section presents compact, runnable examples illustrating typical `DerivKit` workflows.
Additional worked examples and notebooks are available at <https://docs.derivkit.org/main/examples>.

## DerivativeKit: stable numerical differentiation

The example below compares adaptive polynomial-fit differentiation with a finite-difference baseline for a nonlinear scalar function evaluated at a central point.
The adaptive backend is designed to remain stable in regimes where fixed-step finite differences become sensitive to step-size choice.

```python
import numpy as np
from derivkit.derivative_kit import DerivativeKit

def func(x: float) -> float:
    return np.exp(-x*x) * np.sin(3.0*x) + 0.1 * x**3

x0 = 0.3
dk = DerivativeKit(function=func, x0=x0)

d1_adaptive = dk.differentiate(method="adaptive", order=1)

d1_finite = dk.differentiate(
    method="finite",
    order=1,
    extrapolation="ridders")

print("df/dx @ x0:", d1_adaptive, "(adaptive)", d1_finite, "(finite)")
```


## CalculusKit: gradients, Hessians, and Jacobians

`DerivKit` provides calculus utilities for assembling common derivative objects from numerical derivatives.
The example below computes the gradient and Hessian of a scalar-valued model and the Jacobian of a vector-valued model at a central parameter point, delegating numerical differentiation to _DerivativeKit_.

```python
import numpy as np
from derivkit.calculus_kit import CalculusKit

def func_scalar(theta: np.ndarray) -> float:
    x1, x2 = float(theta[0]), float(theta[1])
    return np.exp(x1) * np.sin(x2) \
        + 0.5 * x1**2 * x2**3 \
        - np.log(
            1.0 + x1**2 + x2**2
        )

def func_vector(theta: np.ndarray) -> np.ndarray:
    x1, x2 = float(theta[0]), float(theta[1])
    return np.array([
        np.exp(x1) * np.cos(x2) + x1 * x2**2,
        x1**2 * x2 + np.sin(x1 * x2),
        np.log(1.0 + x1**2 * x2**2) + np.cosh(x1) - np.sinh(x2),
    ], dtype=float)

theta0 = np.array([0.7, -1.2])

ck_scalar = CalculusKit(func_scalar, x0=theta0)
grad = ck_scalar.gradient()
hess = ck_scalar.hessian()
print("grad:", grad)
print("hess:", hess)

ck_vec = CalculusKit(func_vector, x0=theta0)
jac = ck_vec.jacobian()
print("jac shape:", jac.shape)
```

## Forecasting and likelihood expansions

Higher-level inference utilities are provided by _ForecastKit_.
The example below constructs a Fisher matrix for a simple two-parameter model, followed by a Fisher bias estimate and the assembly of second-order DALI tensors

```python
import numpy as np
from derivkit import ForecastKit

def model(theta) -> np.ndarray:
    t1, t2 = float(theta[0]), float(theta[1])
    return np.array([t1 + t2, t1**2 + 2.0 * t2**2], dtype=float)

theta0 = np.array([1.0, 2.0], dtype=float)
cov = np.eye(2, dtype=float)

fk = ForecastKit(function=model, theta0=theta0, cov=cov)

fisher = fk.fisher(method="adaptive")
print("Fisher matrix F:\n", fisher)

# This determines the difference vector
# between the biased and unbiased models
d_unbiased = model(theta0)
d_biased = d_unbiased + np.array([0.5, -0.2], dtype=float) 
delta_nu = fk.delta_nu(data_biased=d_biased, data_unbiased=d_unbiased)

bias_vec, dtheta = fk.fisher_bias(
    fisher_matrix=fisher,
    delta_nu=delta_nu
)
print("Fisher bias vector:\n", bias_vec)
print("Parameter shift:\n", dtheta)

# Construct the second-order DALI corrections
dali_dict = fk.dali()
d1 = dali_dict[2][0]
d2 = dali_dict[2][1]
print("DALI tensors: D1 shape =", d1.shape, ", D2 shape =", d2.shape)
```

`DerivKit` supports Fisher forecasting, Fisher bias calculations, and the construction of higher-order derivative tensors required for DALI likelihood expansions.
These quantities can be used directly for downstream analysis and visualization,
including Gaussian and non-Gaussian posterior contours.
Further scripts and notebook-based demonstrations are available at <https://github.com/derivkit/derivkit-demos>.

## Use cases

Typical applications of `DerivKit` span a wide range of scientific and engineering workflows that require robust numerical derivatives of noisy, interpolated, or tabulated model outputs, including surrogate models and emulators.
While cosmological analyses provide a motivating example, the use cases listed below are broadly applicable to any setting where analytic derivatives or reliable automatic differentiation are unavailable.

1. **Fisher forecasts and local sensitivity analyses:**
    Numerical derivatives of model outputs with respect to parameters enter directly into Fisher-matrix forecasts and related local sensitivity measures.
    `DerivKit` supports diagnostics-driven derivative estimation for noisy, irregular, or computationally expensive models.

2. **Higher-order likelihood expansions and non-Gaussian corrections:**
  When posterior distributions deviate from Gaussianity, higher-order derivatives of the log-likelihood are required to capture leading non-Gaussian features.
  `DerivKit` supports derivative estimation for likelihood expansions such as DALI and related non-Gaussian approximations.

3. **Derivative estimation from tabulated or precomputed models:**
  In many simulation pipelines, model predictions are available only on discrete parameter grids or as precomputed lookup tables.
  ``DerivKit`` enables stable derivative estimation directly from such tabulated data, avoiding the need for model refactoring or surrogate retraining.

4. **Sensitivity studies and gradient validation for black-box models:**
    In workflows involving interpolated models, emulators, or legacy simulation software, analytic gradients are often unavailable or unreliable.
    `DerivKit` can be used to explore local parameter sensitivities, validate finite-difference or automatic-differentiation gradients, and assess numerical stability.

5. **Derivatives of parameter-dependent covariance models:**
    In many inference problems, covariance matrices depend explicitly on model parameters.
    `DerivKit` supports numerical differentiation of such covariance models, enabling Fisher forecasts and likelihood expansions that consistently account for parameter-dependent noise.

A companion study applying `DerivKit` to standard cosmological probes, including weak lensing and galaxy clustering analyses, cosmic microwave background (CMB) observations, and potentially supernova data, is in preparation (_Šarčević et al._, in prep.).

# Acknowledgments

NŠ is supported in part by the OpenUniverse effort, which is funded by NASA under JPL Contract Task 70-711320, ‘Maximizing Science Exploitation of Simulated Cosmological Survey Data Across Surveys.’
MvdW is supported by the Science and Technology Facilities Council via LOFAR-U.K.~[ST/V002406/1] and UKSRC [ST/T000244/1].
CT is supported by the Center for AstroPhysical Surveys (CAPS) at the
National Center for Supercomputing Applications (NCSA), University of Illinois Urbana-Champaign.
This work made use of the Illinois Campus Cluster, a computing resource that is operated by the Illinois Campus Cluster Program (ICCP) in conjunction with the National Center for Supercomputing Applications (NCSA) and which is supported by funds from the University of Illinois at Urbana-Champaign.
The authors thank Marco Bonici, Bastien Carreres, Matthew Feickert, Arun Kannawadi, Konstantin Malanchev, Vivian Miranda, Charlie Mpetha, and Knut Morå for useful discussions.

# Software Acknowledgments

This work made extensive use of open-source scientific software.
We acknowledge the Python programming language [^1] as the primary development environment.
Core numerical functionality was provided by `NumPy` [@numpy] and `SciPy` [@scipy], with visualization handled using `Matplotlib` [@matplotlib].
Parallel evaluations within the library are implemented using the `multiprocess` package [@McKerns2012].
Interactive development and documentation were supported by `Jupyter Notebooks` [@jupyter].
Posterior sampling, analysis, and visualization are handled within the library via `emcee` [@Foreman_Mackey_2013_emcee] and `GetDist` [@Lewis_getdist], which are core dependencies used for sampling, post-processing, and inference workflows.


# Author Contributions

NŠ initiated and led the project, defined the overall scientific and software architecture, and was responsible for the majority of the code development, testing infrastructure, and manuscript preparation.

MvdW made substantial technical contributions throughout the project, including core code development, extensive code review and refinement, establishment and enforcement of development standards, expansion of the testing suite, and major contributions to the user-facing documentation and manuscript.

CT contributed targeted code implementations, performed benchmarking and validation checks, and provided detailed review and feedback on the software, documentation, and manuscript.

# References

[^1]: <https://www.python.org>
