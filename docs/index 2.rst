.. Derivkit documentation master file, created by
   sphinx-quickstart on Wed Aug 20 20:21:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Derivkit documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   contributing

**DerivKit** is a robust Python toolkit for stable numerical derivatives, built for scientific computing, cosmology, and any domain requiring accurate gradients or higher-order expansions.

It provides:

* Adaptive polynomial fitting that excludes noisy points based on residuals,
* High-order finite differences for accurate stencil-based derivatives,
* A simple API for comparing both approaches side by side.

Detailed documentation of the toolkit's modules can be found in the sidebar.

Installation
------------

::

   pip install derivkit

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
  print("Adaptive:", dk.adaptive.differentiate(order=1))
  print("Finite Difference:", dk.finite.differentiate(order=1))


Adaptive Fit Example
--------------------

Below is a visual example of the :py:mod:`derivkit.adaptive_fit` module estimating the first derivative of a nonlinear function in the presence of noise.
The method selectively discards outlier points before fitting a polynomial, resulting in a robust and smooth estimate.

.. image:: assets/plots/adaptive_fit_with_noise_order1_demo_func.png


Citation
--------

If you use ``derivkit`` in your research, please cite it as follows:

::

  @software{sarcevic2025derivkit,
    author       = {Nikolina Šarčević and Matthijs van der Wild},
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
MIT License © 2025 Niko Šarčević, Matthijs van der Wild
