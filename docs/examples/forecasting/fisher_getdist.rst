Fisher forecasts and GetDist integration
=======================================

DerivKit provides lightweight helpers to turn Fisher-matrix forecasts into
GetDist-ready objects for immediate visualization and exploratory analysis.
This is intended for **fast inspection and plotting**, not as a replacement
for full MCMC sampling.

Two complementary workflows are supported:

1. **Analytic Gaussian forecasts** via a Fisher-derived covariance, returned as
   :class:`getdist.gaussian_mixtures.GaussianND`.
2. **Monte Carlo samples drawn from the Fisher Gaussian**, returned as
   :class:`getdist.MCSamples`, optionally including parameter support bounds
   and priors.

Both outputs can be passed directly to GetDist plotting utilities
(e.g. triangle / corner plots).

Overview
--------

Given a model :math:`\nu(\theta)`, a fiducial point :math:`\theta_0`, and a
covariance matrix :math:`C`, DerivKit computes the Fisher matrix

.. math::

   F_{ij} = \frac{\partial \nu}{\partial \theta_i}^T C^{-1}
            \frac{\partial \nu}{\partial \theta_j}

From this Gaussian approximation, DerivKit can either:

- construct an **analytic multivariate Gaussian** with covariance
  :math:`F^{-1}`, or
- **draw samples** from that Gaussian, optionally applying parameter bounds
  or priors.

These helpers are especially useful for:

- quick sanity checks of forecasted constraints,
- rapid comparison between different experimental setups,
- producing publication-quality corner plots without running an MCMC.

Analytic Gaussian (no sampling)
-------------------------------

The simplest option is to convert the Fisher matrix into an analytic Gaussian
object compatible with GetDist.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecast_kit import ForecastKit
   from derivkit.forecasting.integrations.getdist_fisher import (
       fisher_to_getdist_gaussiannd,
   )

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   fisher = fk.fisher(
       method="finite",
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )

   gnd = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher,
       names=["a", "b"],
       labels=[r"a", r"b"],
       label="Fisher (Gaussian)",
   )

   g = plots.get_subplot_plotter()
   g.triangle_plot([gnd], filled=True)

Sampling from the Fisher Gaussian
---------------------------------

For more flexibility (e.g. plotting marginal histograms, applying hard bounds,
or combining with priors), DerivKit can draw Monte Carlo samples directly from
the Fisher Gaussian and return them as :class:`getdist.MCSamples`.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecast_kit import ForecastKit
   from derivkit.forecasting.integrations.getdist_fisher import (
       fisher_to_getdist_samples,
   )

   def model(theta):
       a, b = theta
       return np.array([a, b, a + 2.0 * b], dtype=float)

   theta0 = np.array([1.0, 2.0])
   cov = np.eye(3)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   fisher = fk.fisher(
       method="finite",
       stepsize=1e-2,
       num_points=5,
       extrapolation="ridders",
       levels=4,
   )

   samples = fisher_to_getdist_samples(
       theta0=theta0,
       fisher=fisher,
       names=["a", "b"],
       labels=[r"a", r"b"],
       n_samples=40_000,
       seed=0,
       kernel_scale=1.0,
       hard_bounds=[(0.0, None), (0.0, None)],
       store_loglikes=True,
       label="Fisher (samples)",
   )

   g = plots.get_subplot_plotter()
   g.triangle_plot(samples, filled=True)

Notes and conventions
---------------------

- The Fisher matrix is inverted using a pseudo-inverse to form the Gaussian
  covariance. You can control regularization via ``rcond``.
- ``getdist.MCSamples.loglikes`` stores **minus the log-posterior** (up to an
  additive constant), following GetDist conventions.
- Hard bounds and priors are **optional** and intended for light truncation,
  not for defining complex posteriors.
- For non-Gaussian posteriors or strong parameter degeneracies, use the
  DALI expansions or a full sampler instead.

See also
--------

- :class:`derivkit.forecast_kit.ForecastKit`
- :func:`derivkit.forecasting.fisher.build_fisher_matrix`
- :func:`derivkit.forecasting.integrations.getdist_fisher.fisher_to_getdist_samples`
- :func:`derivkit.forecasting.integrations.getdist_fisher.fisher_to_getdist_gaussiannd`
