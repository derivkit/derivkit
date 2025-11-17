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
   guide/quickstart
   guide/methods
   guide/forecasting
   guide/likelihoods
   guide/references
   contributing
   team
   modules

What is DerivKit?
-----------------

**DerivKit** is a modular toolkit for reliable numerical derivatives and the calculations that depend on them.
It is designed for scientific computing and physics/cosmology, where functions are often noisy, expensive, or non-smooth,
and standard autodiff cannot be used.


DerivKit is organized into four layers:

.. raw:: html

   <details open>
   <summary><strong>1. DerivativeKit Layer</strong></summary>
   <p>Tools for computing stable 1st–Nth derivatives:</p>
   <ul>
     <li>Finite differences (3–9 point stencils, Richardson, Ridders, Gauss–Richardson)</li>
     <li>Simple polynomial fits (local regression)</li>
     <li>Adaptive local polynomial regression (Chebyshev grid, robust diagnostics)</li>
     <li>Gaussian Process derivatives (probabilistic fit, in progress)</li>
     <li>Fornberg analytic weights (in progress)</li>
     <li>Complex-step derivatives (planned)</li>
   </ul>
   </details>

   <details>
   <summary><strong>2. CalculusKit Layer</strong></summary>
   <p>Convenience wrappers built on top of the derivative engines:</p>
   <ul>
     <li>Gradient</li>
     <li>Jacobian</li>
     <li>Hessian</li>
     <li>Mixed partials</li>
   </ul>
   <p>All backends are interchangeable — you can compute a Hessian using adaptive in one call
   and finite difference in the next.</p>
   </details>

   <details>
   <summary><strong>3. ForecastKit Layer</strong></summary>
   <p>Numerical expansions and likelihood-based inference tools:</p>
   <ul>
     <li>Fisher matrices</li>
     <li>DALI expansions</li>
     <li>Fisher bias vectors</li>
     <li>Likelihood wrappers (Gaussian, Poissonian)</li>
   </ul>
   <p>The forecasting tools rely on the derivative engines but are fully general — you can use them
   for any model, not just cosmology.</p>
   </details>

   <details>
   <summary><strong>4. LikelihoodKit</strong></summary>
   <p>Lightweight wrappers for likelihood evaluation:</p>
   <ul>
     <li>Gaussian likelihood (with covariance shaping support)</li>
     <li>Poissonian likelihood (scalar or binned)</li>
     <li>Sellentin likelihood (planned)</li>
   </ul>
   <p>Handles flattening/reshaping data vectors, covariance consistency checks, input validation.</p>
   </details>

Installation
------------

DerivKit is currently distributed from source. To install the latest development
version:

.. code-block:: bash

   git clone https://github.com/derivkit-org/derivkit.git
   cd derivkit
   pip install -e .

For development (tests, linting, docs build tools), you can install the optional
extras:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
-----------

::

  from derivkit import DerivativeKit

  def simple_function(x):
      return x**2 + x

  dk = DerivativeKit(
    function=simple_function,
    x0=1.0
  )
  print("Adaptive:", dk.differentiate(order=1, method="adaptive"))
  print("Finite Difference:", dk.differentiate(order=1, method="finite"))


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
