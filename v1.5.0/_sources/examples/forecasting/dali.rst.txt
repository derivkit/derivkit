.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px

|dklogo| DALI
==============

This section shows how to construct Derivative Approximation for
LIkelihoods (DALI) tensors using :class:`derivkit.forecast_kit.ForecastKit`,
and how to sample and visualize the resulting non-Gaussian posterior.

The DALI expansion extends the Fisher matrix by including higher-order
derivative information, allowing non-Gaussian structure in the likelihood
to be approximated locally around a fiducial parameter point ``theta0``.

In DerivKit, DALI tensors are returned using an *introduced-at-order*
convention: each forecast order contributes a tuple of tensors, and all
orders up to the requested one are returned.

The model must map parameters ``theta`` to a 1D data vector, and the observable
covariance matrix must have shape ``(n, n)``, where ``n`` is the length of the
data vector.

For a conceptual overview of DALI forecasting and its interpretation, see
:doc:`../../about/kits/forecast_kit`.


Basic usage
-----------


DALI tensor shapes
~~~~~~~~~~~~~~~~~~

For ``p`` model parameters, the DALI expansion returns tensors with the
following shapes:

- order 1 (Fisher; "singlet-DALI"):

  - ``F`` with shape ``(p, p)``

- order 2 (doublet-DALI):

  - ``D1`` with shape ``(p, p, p)``
  - ``D2`` with shape ``(p, p, p, p)``

- order 3 (triplet-DALI):

  - ``T1`` with shape ``(p, p, p, p)``
  - ``T2`` with shape ``(p, p, p, p, p)``
  - ``T3`` with shape ``(p, p, p, p, p, p)``

All tensors are evaluated at the fiducial parameter point ``theta0``.

The forecast order specifies the highest derivative order of the forward
model included in the DALI expansion, rather than the polynomial degree
of the resulting log-likelihood. For example, doublet-DALI includes
terms up to fourth order in parameter displacements.


The example below constructs DALI tensors around a fiducial parameter
point ``theta0`` using a simple toy model.

.. doctest:: dali_basic

   >>> import numpy as np
   >>> from derivkit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> # Define a nonlinear toy model: R^2 -> R^3
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a**2 + 2.0 * b], dtype=float)
   >>> # Fiducial parameters and covariance
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> # Build ForecastKit and compute DALI tensors up to second order
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(forecast_order=2)
   >>> F = dali[1][0]
   >>> D1, D2 = dali[2]
   >>> print(F.shape)
   (2, 2)
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Choosing a derivative backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with Fisher forecasting, you can control how derivatives are computed by
passing ``method`` and backend-specific options.

All keyword arguments are forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

.. doctest:: dali_backend_control

   >>> import numpy as np
   >>> from derivkit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a**2 + 2.0 * b], dtype=float)
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(
   ...     forecast_order=2,
   ...     method="finite",
   ...     n_workers=2,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> D1, D2 = dali[2]
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Parallel execution
~~~~~~~~~~~~~~~~~~

DALI tensor components can be computed in parallel using ``n_workers``.

This parallelizes derivative evaluations across parameters and tensor entries.

.. doctest:: dali_parallel

   >>> import numpy as np
   >>> from derivkit import ForecastKit
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a**2 + 2.0 * b], dtype=float)
   >>> fk = ForecastKit(
   ...     function=model,
   ...     theta0=np.array([1.0, 2.0]),
   ...     cov=np.eye(3),
   ... )
   >>> dali = fk.dali(forecast_order=2, n_workers=4)
   >>> D1, D2 = dali[2]
   >>> print(D1.shape)
   (2, 2, 2)
   >>> print(D2.shape)
   (2, 2, 2, 2)


Sampling and visualization
--------------------------

Once the DALI tensors have been constructed, DerivKit can generate GetDist
samples from the DALI-expanded posterior using either ``emcee`` or importance
sampling with a Fisher–Gaussian proposal. Both return a
:class:`getdist.MCSamples` object. The resulting contours describe the
DALI approximation, not the exact nonlinear posterior.


Sampling the DALI posterior with emcee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses a two-parameter nonlinear benchmark with a curved
parameter degeneracy. The DALI expansion captures departures from the
locally Gaussian Fisher approximation, and ``emcee`` samples the resulting
DALI-expanded posterior.

.. doctest:: dali_getdist_emcee

   >>> import numpy as np
   >>> from getdist import plots as getdist_plots
   >>> from derivkit import ForecastKit
   >>> def model(theta):
   ...     u, v = theta
   ...     du, dv = u - 0.15, v - 0.05
   ...     return np.array([
   ...         1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
   ...         0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
   ...     ])
   >>> theta0 = np.array([0.15, 0.05])
   >>> sigma = 0.10 * np.abs(model(theta0))
   >>> cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(forecast_order=2)
   >>> samples = fk.getdist_dali_emcee(
   ...     dali=dali,
   ...     names=["u", "v"],
   ...     labels=[r"u", r"v"],
   ...     label="DALI (emcee)",
   ... )
   >>> dk_red = "#f21901"
   >>> line_width = 1.5
   >>> plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   >>> plotter.settings.linewidth_contour = line_width
   >>> plotter.settings.linewidth = line_width
   >>> plotter.settings.figure_legend_frame = False
   >>> plotter.settings.legend_rect_border = False
   >>> plotter.triangle_plot(
   ...     [samples],
   ...     params=["u", "v"],
   ...     filled=False,
   ...     contour_colors=[dk_red],
   ...     contour_lws=[line_width],
   ...     contour_ls=["-"],
   ... )
   >>> samples.numrows > 0
   True


.. plot::
   :include-source: False
   :width: 520

   import numpy as np
   from getdist import plots as getdist_plots
   from derivkit import ForecastKit

   def model(theta):
       u, v = theta
       du, dv = u - 0.15, v - 0.05
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
       ])

   theta0 = np.array([0.15, 0.05])
   sigma = 0.10 * np.abs(model(theta0))
   cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   dali = fk.dali(forecast_order=2)

   samples = fk.getdist_dali_emcee(
       dali=dali,
       names=["u", "v"],
       labels=[r"u", r"v"],
       label="DALI (emcee)",
   )

   dk_red = "#f21901"
   line_width = 1.5

   plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False
   plotter.settings.legend_rect_border = False

   plotter.triangle_plot(
       [samples],
       params=["u", "v"],
       filled=False,
       contour_colors=[dk_red],
       contour_lws=[line_width],
       contour_ls=["-"],
   )


Sampling the DALI posterior with importance sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the same nonlinear benchmark as the previous example, we now
sample the DALI posterior through importance weighting of a
Fisher–Gaussian proposal rather than running an MCMC chain.

This provides a faster alternative to ``emcee`` when the Fisher–Gaussian
proposal adequately covers the DALI posterior.

.. doctest:: dali_getdist_importance

   >>> import numpy as np
   >>> from getdist import plots as getdist_plots
   >>> from derivkit import ForecastKit
   >>> def model(theta):
   ...     u, v = theta
   ...     du, dv = u - 0.15, v - 0.05
   ...     return np.array([
   ...         1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
   ...         0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
   ...     ])
   >>> theta0 = np.array([0.15, 0.05])
   >>> sigma = 0.10 * np.abs(model(theta0))
   >>> cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(forecast_order=2)
   >>> samples = fk.getdist_dali_importance(
   ...     dali=dali,
   ...     names=["u", "v"],
   ...     labels=[r"u", r"v"],
   ...     label="DALI (importance)",
   ...     n_samples=50_000,
   ...     seed=0,
   ...     kernel_scale=1.3,
   ... )
   >>> dk_yellow = "#f2b701"
   >>> line_width = 1.5
   >>> plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   >>> plotter.settings.linewidth_contour = line_width
   >>> plotter.settings.linewidth = line_width
   >>> plotter.settings.figure_legend_frame = False
   >>> plotter.settings.legend_rect_border = False
   >>> plotter.triangle_plot(
   ...     [samples],
   ...     params=["u", "v"],
   ...     filled=False,
   ...     contour_colors=[dk_yellow],
   ...     contour_lws=[line_width],
   ...     contour_ls=["-"],
   ... )
   >>> samples.numrows > 0
   True


.. plot::
   :include-source: False
   :width: 520

   import numpy as np
   from getdist import plots as getdist_plots
   from derivkit import ForecastKit

   def model(theta):
       u, v = theta
       du, dv = u - 0.15, v - 0.05
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
       ])

   theta0 = np.array([0.15, 0.05])
   sigma = 0.10 * np.abs(model(theta0))
   cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   dali = fk.dali(forecast_order=2)

   samples = fk.getdist_dali_importance(
       dali=dali,
       names=["u", "v"],
       labels=[r"u", r"v"],
       label="DALI (importance)",
       n_samples=50_000,
       seed=0,
       kernel_scale=1.3,
   )

   dk_yellow = "#f2b701"
   line_width = 1.5

   plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False
   plotter.settings.legend_rect_border = False

   plotter.triangle_plot(
       [samples],
       params=["u", "v"],
       filled=False,
       contour_colors=[dk_yellow],
       contour_lws=[line_width],
       contour_ls=["-"],
   )

.. note::

   This example illustrates the importance-sampling API rather than
   validating its accuracy for this particular posterior. For scientific
   applications, check the effective sample size and compare against
   an independent sampler such as ``emcee``, especially for strongly
   non-Gaussian posteriors.


.. _dali-including-priors:

Including priors in DALI contours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Priors can be included in DALI sampling through ``prior_terms`` and
``prior_bounds``. Here, we use the same nonlinear benchmark as above
to compare the DALI posterior with and without a correlated Gaussian prior.

An informative prior can modify the orientation and extent of the posterior
contours. If sufficiently restrictive, it can also suppress the non-Gaussian
structure captured by DALI.

.. doctest:: dali_with_priors_overlay_emcee

   >>> import numpy as np
   >>> from getdist import plots as getdist_plots
   >>> from derivkit import ForecastKit
   >>> def model(theta):
   ...     u, v = theta
   ...     du, dv = u - 0.15, v - 0.05
   ...     return np.array([
   ...         1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
   ...         0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
   ...     ])
   >>> theta0 = np.array([0.15, 0.05])
   >>> sigma = 0.10 * np.abs(model(theta0))
   >>> cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> dali = fk.dali(forecast_order=2)
   >>> samples_base = fk.getdist_dali_emcee(
   ...     dali=dali,
   ...     names=["u", "v"],
   ...     labels=[r"u", r"v"],
   ...     label="DALI",
   ... )
   >>> prior_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
   >>> su, sv, rho = 0.08, 0.03, -0.6
   >>> cov_prior = np.array([
   ...     [su**2, rho * su * sv],
   ...     [rho * su * sv, sv**2],
   ... ])
   >>> prior_terms = [("gaussian", {"mean": theta0, "cov": cov_prior})]
   >>> samples_prior = fk.getdist_dali_emcee(
   ...     dali=dali,
   ...     names=["u", "v"],
   ...     labels=[r"u", r"v"],
   ...     label="DALI + correlated prior",
   ...     prior_bounds=prior_bounds,
   ...     prior_terms=prior_terms,
   ... )
   >>> dk_red = "#f21901"
   >>> dk_yellow = "#f2b701"
   >>> line_width = 1.5
   >>> plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   >>> plotter.settings.linewidth_contour = line_width
   >>> plotter.settings.linewidth = line_width
   >>> plotter.settings.figure_legend_frame = False
   >>> plotter.settings.legend_rect_border = False
   >>> plotter.triangle_plot(
   ...     [samples_base, samples_prior],
   ...     params=["u", "v"],
   ...     filled=[False, False],
   ...     contour_colors=[dk_red, dk_yellow],
   ...     contour_lws=[line_width, line_width],
   ...     contour_ls=["-", "-"],
   ... )
   >>> (samples_base.numrows > 0) and (samples_prior.numrows > 0)
   True


.. plot::
   :include-source: False
   :width: 520

   import numpy as np
   from getdist import plots as getdist_plots
   from derivkit import ForecastKit

   def model(theta):
       u, v = theta
       du, dv = u - 0.15, v - 0.05
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (dv + 1.8 * du**2 - 0.3 * du**3 + 0.20 * du * dv),
       ])

   theta0 = np.array([0.15, 0.05])
   sigma = 0.10 * np.abs(model(theta0))
   cov = np.outer(sigma, sigma) * np.array([[1.0, 0.3], [0.3, 1.0]])

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   dali = fk.dali(forecast_order=2)

   samples_base = fk.getdist_dali_emcee(
       dali=dali,
       names=["u", "v"],
       labels=[r"u", r"v"],
       label="DALI",
   )

   prior_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
   su, sv, rho = 0.08, 0.03, -0.6
   cov_prior = np.array([
       [su**2, rho * su * sv],
       [rho * su * sv, sv**2],
   ])
   prior_terms = [("gaussian", {"mean": theta0, "cov": cov_prior})]

   samples_prior = fk.getdist_dali_emcee(
       dali=dali,
       names=["u", "v"],
       labels=[r"u", r"v"],
       label="DALI + correlated prior",
       prior_bounds=prior_bounds,
       prior_terms=prior_terms,
   )

   dk_red = "#f21901"
   dk_yellow = "#f2b701"
   line_width = 1.5

   plotter = getdist_plots.get_subplot_plotter(width_inch=3.9)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False
   plotter.settings.legend_rect_border = False

   plotter.triangle_plot(
       [samples_base, samples_prior],
       params=["u", "v"],
       filled=[False, False],
       contour_colors=[dk_red, dk_yellow],
       contour_lws=[line_width, line_width],
       contour_ls=["-", "-"],
   )


Notes and conventions
---------------------

- The non-Gaussianity here comes from the nonlinear forward model.
- ``getdist.MCSamples.loglikes`` stores minus the log-posterior (up to an additive constant),
  following GetDist conventions.
- Importance sampling uses a Fisher–Gaussian proposal; ``kernel_scale`` controls its width.
  If weights become extremely uneven, try increasing the scale slightly.
- Importance sampling is intended for fast visualization and exploratory work,
  and is reliable when the Fisher–Gaussian proposal closely matches the
  DALI posterior.
- For science analyses requiring a robust exploration of the posterior,
  including non-Gaussian structure and tails, we recommend using **emcee**
  to sample the DALI-expanded posterior.


Typical workflow
----------------

1. Compute DALI tensors with :class:`ForecastKit` (e.g. ``dali = fk.dali(forecast_order=2)``).
2. Use importance sampling for fast visualization and iteration.
3. Switch to ``emcee`` when robustness is required or strong non-Gaussianity
   leads to unstable importance weights.
4. Visualize and compare results using GetDist triangle plots.


See also
--------

- :class:`derivkit.forecast_kit.ForecastKit`
- :meth:`derivkit.forecast_kit.ForecastKit.dali`
- :func:`derivkit.forecast_kit.ForecastKit.logposterior_dali`
- :meth:`derivkit.forecast_kit.ForecastKit.getdist_dali_emcee`
- :meth:`derivkit.forecast_kit.ForecastKit.getdist_dali_importance`
