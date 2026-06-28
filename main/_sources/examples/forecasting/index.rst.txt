.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| Forecasting
====================

Task-oriented examples for forecasting workflows, including Fisher matrices,
DALI expansions, and Laplace approximations.

For end-to-end workflows and larger examples, see the
`derivkit-demos repository <https://github.com/derivkit/derivkit-demos>`_.

----

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Fisher
      :link: fisher
      :link-type: doc

      Standard Fisher matrix forecasts from model derivatives.

   .. grid-item-card:: Fisher bias
      :link: fisher_bias
      :link-type: doc

      Parameter biases from systematic model mismatch.

   .. grid-item-card:: DALI
      :link: dali
      :link-type: doc

      Higher-order likelihood expansions beyond Fisher.

   .. grid-item-card:: Fisher contours
      :link: fisher_contours
      :link-type: doc

      Visualize Fisher-Gaussian posteriors with GetDist.

   .. grid-item-card:: DALI contours
      :link: dali_contours
      :link-type: doc

      Visualize DALI-expanded posteriors with GetDist.

   .. grid-item-card:: Laplace approximation
      :link: laplace_approx
      :link-type: doc

      Gaussian approximation around a posterior peak.

   .. grid-item-card:: Laplace contours
      :link: laplace_contours
      :link-type: doc

      Visualize Laplace-approximated posteriors with GetDist.

   .. grid-item-card:: Gaussian Fisher
      :link: fisher_gauss
      :link-type: doc

      Fisher matrix including parameter-dependent covariance.

   .. grid-item-card:: X–Y Gaussian Fisher
      :link: fisher_xy
      :link-type: doc

      Fisher forecasts with uncertainty in both inputs and outputs.

.. toctree::
   :maxdepth: 2
   :hidden:

   fisher
   fisher_bias
   dali
   fisher_contours
   dali_contours
   laplace_approx
   laplace_contours
   fisher_gauss
   fisher_xy
