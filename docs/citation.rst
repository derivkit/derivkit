.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Citation
=================

If you use DerivKit in your work, please cite it using the reference below.


How to cite DerivKit
--------------------

.. code-block:: bibtex

   @software{Sarcevic_DerivKit,
     author  = {Šarčević, Nikolina and van der Wild, Matthijs and Trendafilova, Cynthia},
     title   = {{DerivKit}},
     license = {MIT},
     url     = {https://github.com/derivkit/derivkit}
   }


Software using DerivKit
-----------------------

DerivKit is used as a dependency or internal component in the following
external software and analysis pipelines:

- **Augur**

  A DESC forecasting and inference validation tool that generates synthetic
  datasets and performs parameter inference using either full MCMC or
  second-order derivative–based Fisher matrix forecasts.
  DerivKit is used internally for derivative-based Fisher forecasts.
  Repository: https://github.com/LSSTDESC/augur

- **COCOA**

  A cosmological analysis framework that integrates CosmoLike within the Cobaya
  inference engine, enabling forecasts and multi-probe analyses for surveys such
  as DES, LSST, and the Roman Space Telescope.
  COCOA provides a containerized workflow for reproducible inference, ensuring
  consistent compiler and library environments across platforms.
  DerivKit is used internally for derivative-based forecasting and Fisher matrix
  calculations.
  Repository: https://github.com/CosmoLike/cocoa


Publications using DerivKit
---------------------------

The following publications make use of DerivKit, either directly or as part
of a larger analysis pipeline.

*No publications listed yet.*

If you use DerivKit in a publication and would like it listed here, please open
an issue or pull request on the DerivKit repository.
