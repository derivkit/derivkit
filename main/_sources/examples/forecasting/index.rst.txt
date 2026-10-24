.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| Forecasting
====================

Task-oriented examples for forecasting workflows, including Fisher matrices,
DALI expansions, systematic parameter biases, and Laplace approximations.

For end-to-end workflows and larger examples, see the
`derivkit-demos repository <https://github.com/derivkit/derivkit-demos>`_.

----

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Fisher
      :link: fisher
      :link-type: doc

      Fisher matrix forecasts, parameter uncertainties, and posterior contours.

   .. grid-item-card:: Fisher bias
      :link: fisher_bias
      :link-type: doc

      Parameter shifts from systematic model mismatch.

   .. grid-item-card:: DALI
      :link: dali
      :link-type: doc

      Higher-order likelihood expansions and non-Gaussian posterior contours.

   .. grid-item-card:: DALI bias
      :link: dali_bias
      :link-type: doc

      Non-Gaussian MAP and posterior mean shifts from systematic mismatch.

   .. grid-item-card:: Laplace approximation
      :link: laplace_approx
      :link-type: doc

      Gaussian approximation around a posterior peak and its contours.

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
   dali_bias
   laplace_approx
   laplace_contours
   fisher_gauss
   fisher_xy