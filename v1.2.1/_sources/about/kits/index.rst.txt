.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| Kits
==============

This section provides a conceptual overview of DerivKit’s core kits.
Each kit groups related functionality around a specific role in derivative
evaluation, calculus operations, forecasting, or likelihood-based inference.

The focus here is on *what each kit is responsible for* and *how the kits are
intended to be combined*, rather than on step-by-step usage examples.

Examples demonstrating how to use each kit can be found in the
:doc:`../../examples/index` section, while more detailed and in-depth demos
are available in the
`DerivKit demos repository <https://github.com/derivkit/derivkit-demos>`_.


----

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: DerivativeKit
      :link: derivative_kit
      :link-type: doc

      Numerical differentiation engines and derivative evaluation strategies.

   .. grid-item-card:: CalculusKit
      :link: calculus_kit
      :link-type: doc

      Construction of gradients, Jacobians, and Hessians from derivatives.

   .. grid-item-card:: ForecastKit
      :link: forecast_kit
      :link-type: doc

      Fisher matrices, DALI expansions, and Laplace approximations.

   .. grid-item-card:: LikelihoodKit
      :link: likelihood_kit
      :link-type: doc

      Likelihood evaluation and approximate inference utilities.

.. toctree::
   :maxdepth: 2
   :hidden:

   derivative_kit
   calculus_kit
   forecast_kit
   likelihood_kit
