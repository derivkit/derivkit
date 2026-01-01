.. Derivkit documentation master file, created by
   sphinx-quickstart on Wed Aug 20 20:21:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DerivKit documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/installation
   examples/index
   guide/methods
   guide/forecasting
   guide/likelihoods
   contributing
   team
   modules

What is DerivKit?
-----------------

**DerivKit** is a modular toolkit for reliable numerical derivatives and the calculations that depend on them.
It is designed for scientific computing and physics/cosmology, where functions are often noisy, expensive, or non-smooth,
and standard automatic differentiation is inappropriate.


DerivKit is organized into four layers:

.. dropdown:: **1. DerivativeKit Layer**
   :open:

   Tools for computing stable 1st–Nth derivatives:

   * Finite differences (3–9 point stencils, Richardson, Ridders, Gauss–Richardson)
   * Simple polynomial fits (local regression)
   * Adaptive local polynomial regression (Chebyshev grid, robust diagnostics)
   * Gaussian Process derivatives (probabilistic fit)
   * Fornberg analytic weights (in progress)
   * Complex-step derivatives (planned)

.. dropdown:: **2. CalculusKit Layer**

   Convenience wrappers built on the derivative engines:

   * Gradient
   * Jacobian
   * Hessian
   * Mixed partials

   All backends are interchangeable.

.. dropdown:: **3. ForecastKit Layer**

   Numerical expansions & forecasting tools:

   * Fisher matrices
   * Fisher bias vectors
   * DALI expansions

.. dropdown:: **4. LikelihoodKit**

   Lightweight, safe likelihood evaluation:

   * Gaussian likelihood (covariance shaping)
   * Poisson likelihood (scalar or binned)
   * Sellentin likelihood (planned)

   Handles flattening, reshaping, and validation.

Installation
------------

To see how to install derivkit, please see :doc:`guide/installation`.

Examples
--------

You can find a collection of usage examples in the :doc:`examples/index` section.

Derivative Methods
------------------

DerivKit provides several interchangeable derivative engines, including finite
differences, simple polynomial regression, and adaptive Chebyshev polynomial
fits.

For a detailed comparison, examples, and recommendations on which method to
use, see :doc:`guide/methods`.


Cheat Sheet: Choosing the Right Method
--------------------------------------

   +------------------------------+------------------------------+--------------------------------------------------------+
   | **Situation**                | **Recommended Method**       | **Why**                                                |
   +==============================+==============================+========================================================+
   | Smooth, cheap function       | Finite Difference            | Fast and accurate for clean functions                  |
   +------------------------------+------------------------------+--------------------------------------------------------+
   | Slightly noisy function      | Ridders Finite Difference    | Richardson + error control stabilises noise            |
   +------------------------------+------------------------------+--------------------------------------------------------+
   | Moderate or structured noise | Simple Polynomial Fit        | Local regression smooths noise better than FD          |
   +------------------------------+------------------------------+--------------------------------------------------------+
   | High noise / messy signal    | Adaptive PolyFit (Chebyshev) | Robust trimming, Chebyshev grid, diagnostics           |
   +------------------------------+------------------------------+--------------------------------------------------------+
   | Expensive function           | Adaptive PolyFit (Chebyshev) | Fewer evaluations and stable fit around ``x0``         |
   +------------------------------+------------------------------+--------------------------------------------------------+
   | Need robustness + diagnostics| Adaptive PolyFit (Chebyshev) | Fit quality metrics, degree adjustment, suggestions    |
   +------------------------------+------------------------------+--------------------------------------------------------+


Citation
--------

If you use ``derivkit`` in your research, please cite it as follows:

::

  @software{sarcevic2025derivkit,
    author       = {Nikolina Šarčević and Matthijs van der Wild and Cynthia Trendafilova and Bastien Carreres},
    title        = {derivkit: A Python Toolkit for Numerical Derivatives},
    year         = {2025},
    publisher    = {GitHub},
    journal      = {GitHub Repository},
    howpublished = {\url{https://github.com/derivkit/derivkit}},
  }

Contributing
------------

Interested in getting involved?
Have a look at :ref:`contributing`!

License
-------
MIT License © 2025 Niko Šarčević, Matthijs van der Wild et al.
