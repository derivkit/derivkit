X–Y Gaussian Fisher matrix
==========================

This section shows how to compute a Gaussian Fisher matrix when both the inputs
and outputs are noisy and may be correlated.

The primary interface for this workflow is
:func:`derivkit.forecasting.fisher_xy.build_xy_gaussian_fisher_matrix`.


A mock example
--------------

This example compares:

- **Standard Fisher**: treat the measured inputs ``x0`` as exact and use only the
  output covariance ``cov_yy``.
- **X–Y Fisher**: propagate input uncertainty using ``cov_xx`` and ``cov_xy`` and compute
  a Fisher matrix with the effective covariance ``R(theta)``.

.. doctest:: fisher_xy_standard_vs_xy

   >>> import numpy as np
   >>> from getdist import plots as getdist_plots
   >>> from derivkit.forecast_kit import ForecastKit
   >>> from derivkit.forecasting.fisher_xy import build_xy_gaussian_fisher_matrix
   >>> from derivkit.forecasting.getdist_fisher_samples import fisher_to_getdist_gaussiannd
   >>> # Toy model: x in R^2, theta in R^2, y in R^3
   >>> def mu_xy(x, theta):
   ...     x = np.asarray(x, dtype=float)
   ...     theta = np.asarray(theta, dtype=float)
   ...     a, b = theta
   ...     x0, x1 = x
   ...     return np.array(
   ...         [
   ...             a * x0 + 0.6 * b,
   ...             a * (x1 ** 2) + 0.3 * b,
   ...             a * (x0 + 0.5 * x1) + b * x0,
   ...         ],
   ...         dtype=float,
   ...     )
   >>> # Fiducial parameters and inputs
   >>> theta0 = np.array([1.2, -0.3])
   >>> x0 = np.array([0.7, 1.1])
   >>> # Covariances
   >>> cyy = np.array(
   ...     [
   ...         [0.25, 0.04, 0.01],
   ...         [0.04, 0.20, 0.03],
   ...         [0.01, 0.03, 0.18],
   ...     ],
   ...     dtype=float,
   ... )
   >>> cxx = np.array([[0.06, 0.02], [0.02, 0.10]], dtype=float)
   >>> cxy = np.array(
   ...     [
   ...         [0.03, -0.01, 0.02],
   ...         [0.00,  0.02, -0.015],
   ...     ],
   ...     dtype=float,
   ... )
   >>> def mu_theta(theta):
   ...     return mu_xy(x0, theta)
   >>> fk_std = ForecastKit(function=mu_theta, theta0=theta0, cov=cyy)
   >>> fisher_std = fk_std.fisher()  # standard Fisher: fixed cov_yy
   >>> # X–Y Fisher: propagate input uncertainty into an effective covariance R(theta)
   >>> fisher_xy = build_xy_gaussian_fisher_matrix(
   ...     theta0=theta0,
   ...     x0=x0,
   ...     mu_xy=mu_xy,
   ...     cxx=cxx,
   ...     cxy=cxy,
   ...     cyy=cyy,
   ... )
   >>> bool(np.allclose(fisher_std, fisher_xy, atol=1e-10, rtol=0.0))
   False
   >>> # Convert to GetDist GaussianND samples for visualization
   >>> gnd_std = fisher_to_getdist_gaussiannd(
   ...     theta0=theta0,
   ...     fisher=fisher_std,
   ...     names=["a", "b"],
   ...     labels=[r"a", r"b"],
   ...     label="Standard Fisher",
   ... )
   >>> gnd_xy = fisher_to_getdist_gaussiannd(
   ...     theta0=theta0,
   ...     fisher=fisher_xy,
   ...     names=["a", "b"],
   ...     labels=[r"a", r"b"],
   ...     label="X–Y Fisher",
   ... )
   >>> (gnd_std is not None) and (gnd_xy is not None)
   True
   >>> # Plot the contours
   >>> dk_yellow = "#e1af00"
   >>> dk_red = "#f21901"
   >>> line_width = 1.5
   >>> plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
   >>> plotter.settings.linewidth_contour = line_width
   >>> plotter.settings.linewidth = line_width
   >>> plotter.settings.figure_legend_frame = False
   >>> plotter.triangle_plot(
   ...     [gnd_std, gnd_xy],
   ...     params=["a", "b"],
   ...     legend_labels=["Standard Fisher", "X–Y Fisher"],
   ...     legend_ncol=1,
   ...     filled=[False, False],
   ...     contour_colors=[dk_yellow, dk_red],
   ...     contour_lws=[line_width, line_width],
   ...     contour_ls=["-", "-"],
   ...     )


.. plot::
   :include-source: False
   :width: 420

   import numpy as np
   from getdist import plots as getdist_plots

   from derivkit.forecast_kit import ForecastKit
   from derivkit.forecasting.fisher_xy import build_xy_gaussian_fisher_matrix
   from derivkit.forecasting.getdist_fisher_samples import fisher_to_getdist_gaussiannd

   def mu_xy(x, theta):
       x = np.asarray(x, dtype=float)
       theta = np.asarray(theta, dtype=float)
       a, b = theta
       x0, x1 = x
       return np.array(
           [
               a * x0 + 0.6 * b,
               a * (x1 ** 2) + 0.3 * b,
               a * (x0 + 0.5 * x1) + b * x0,
           ],
           dtype=float,
       )

   theta0 = np.array([1.2, -0.3])
   x0 = np.array([0.7, 1.1])

   cyy = np.array(
       [
           [0.25, 0.04, 0.01],
           [0.04, 0.20, 0.03],
           [0.01, 0.03, 0.18],
       ],
       dtype=float,
   )

   cxx = np.array([[0.06, 0.02], [0.02, 0.10]], dtype=float)
   cxy = np.array(
       [
           [0.03, -0.01, 0.02],
           [0.00,  0.02, -0.015],
       ],
       dtype=float,
   )

   def mu_theta(theta):
       return mu_xy(x0, theta)

   fk_std = ForecastKit(function=mu_theta, theta0=theta0, cov=cyy)
   fisher_std = fk_std.fisher(
   )

   fisher_xy = build_xy_gaussian_fisher_matrix(
       theta0=theta0,
       x0=x0,
       mu_xy=mu_xy,
       cxx=cxx,
       cxy=cxy,
       cyy=cyy,
       rcond=1e-12,
       symmetrize_dcov=True,
   )

   gnd_std = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_std,
       names=["a", "b"],
       labels=[r"a", r"b"],
       label="Standard Fisher",
   )
   gnd_xy = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_xy,
       names=["a", "b"],
       labels=[r"a", r"b"],
       label="X–Y Fisher",
   )

   dk_yellow = "#e1af00"
   dk_red = "#f21901"
   line_width = 1.5

   plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False

   plotter.triangle_plot(
       [gnd_std, gnd_xy],
       params=["a", "b"],
       legend_labels=["Standard Fisher", "X-Y Fisher"],
       legend_ncol=1,
       filled=[False, False],
       contour_colors=[dk_yellow, dk_red],
       contour_lws=[line_width, line_width],
       contour_ls=["-", "-"],
   )


Notes
-----

- The **standard** Fisher here treats the measured inputs ``x0`` as exact and uses
  ``cov_yy`` directly.
- The **X–Y** Fisher propagates input uncertainty using ``cov_xx`` and ``cov_xy`` via an
  effective covariance ``R(theta)`` built from the local sensitivity matrix
  :math:`T = d mu_xy / d x` evaluated at ``(x0, theta)``.
- The result depends on the derivative backend used to compute the sensitivity matrix.
  Use ``method`` and ``**dk_kwargs`` to control the derivative settings consistently.
