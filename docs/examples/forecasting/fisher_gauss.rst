Gaussian Fisher matrix
======================

This section shows how to compute a Gaussian Fisher matrix for a model with
parameter-dependent mean and (optionally) parameter-dependent covariance.

The primary interface for this workflow is
:class:`derivkit.forecast_kit.ForecastKit`.
For conceptual background, see :doc:`../../guide/forecasting`.


A mock example
--------------

This example compares:

- **Standard Fisher**: uses a fixed covariance :math:`C(\theta_0)` and computes
  :math:`F = J^\mathsf{T} C^{-1} J`.

- **Full Gaussian Fisher**: allows :math:`C(\theta)` and adds the corresponding
  covariance-derivative terms to the Fisher matrix.

- The covariance dependence is chosen to be noticeable but not dominant, while
  remaining SPD by construction.


.. doctest:: gaussian_fisher_standard_vs_full

   >>> import numpy as np
   >>> from getdist import plots as getdist_plots
   >>> from derivkit.forecast_kit import ForecastKit
   >>> from derivkit.forecasting.getdist_fisher_samples import fisher_to_getdist_gaussiannd
   >>> # Toy model: data vector in R^3, parameters in R^2
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a + 0.5 * b, 0.4 * a * b, 0.8 * a + 0.3 * b], dtype=float)
   >>> # Parameter-dependent covariance
   >>> # C(theta) = C_base + alpha * S(theta) S(theta)^T.
   >>> def cov_fn(theta):
   ...     a, b = theta
   ...     c_base = np.array(
   ...         [
   ...             [0.30, 0.03, 0.01],
   ...             [0.03, 0.26, 0.02],
   ...             [0.01, 0.02, 0.22],
   ...         ],
   ...         dtype=float,
   ...     )
   ...     s = np.array(
   ...         [
   ...             [1.0 + 0.9 * a,  0.7 * b,        0.15 * a],
   ...             [0.25 * b,       0.9 + 0.8 * b,  0.35 * a],
   ...             [0.10 * a,       0.45 * b,       0.8 + 1.0 * a],
   ...         ],
   ...         dtype=float,
   ...     )
   ...     alpha = 0.25
   ...     return c_base + alpha * (s @ s.T)
   >>> # Fiducial parameters
   >>> theta0 = np.array([1.2, -0.3])
   >>> # Build ForecastKit and compute both Fisher matrices
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov_fn)
   >>> fisher_std = fk.fisher()
   >>> fisher_full = fk.gaussian_fisher()
   >>> # Convert to GetDist GaussianND samples for visualization
   >>> gnd_std = fisher_to_getdist_gaussiannd(
   ...     theta0=theta0,
   ...     fisher=fisher_std,
   ...     names=["a", "b"],
   ...     labels=[r"a", r"b"],
   ...     label="Standard Fisher",
   ... )
   >>> gnd_full = fisher_to_getdist_gaussiannd(
   ...     theta0=theta0,
   ...     fisher=fisher_full,
   ...     names=["a", "b"],
   ...     labels=[r"a", r"b"],
   ...     label="Full Gaussian Fisher",
   ... )
   >>> (gnd_std is not None) and (gnd_full is not None)
   True
   >>> # Plot the results
   >>> dk_yellow = "#e1af00"
   >>> dk_red = "#f21901"
   >>> line_width = 1.5
   >>> plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
   >>> plotter.settings.linewidth_contour = line_width
   >>> plotter.settings.linewidth = line_width
   >>> plotter.settings.figure_legend_frame = False
   >>> plotter.triangle_plot(
   ...     [gnd_std, gnd_full],
   ...     params=["a", "b"],
   ...     legend_labels=["Standard Fisher", "Full Fisher"],
   ...     legend_ncol=1,
   ...     filled=[False, False],
   ...     contour_colors=[dk_yellow, dk_red],
   ...     contour_lws=[line_width, line_width],
   ...     contour_ls=["-", "-"],
   ... )



.. plot::
   :include-source: False
   :width: 420

   import numpy as np
   from getdist import plots as getdist_plots

   from derivkit.forecast_kit import ForecastKit
   from derivkit.forecasting.getdist_fisher_samples import fisher_to_getdist_gaussiannd

   def model(theta):
       a, b = theta
       return np.array([a + 0.5 * b, 0.4 * a * b, 0.8 * a + 0.3 * b], dtype=float)

   def cov_fn(theta):
       a, b = theta
       c_base = np.array(
           [
               [0.30, 0.03, 0.01],
               [0.03, 0.26, 0.02],
               [0.01, 0.02, 0.22],
           ],
           dtype=float,
       )
       s = np.array(
           [
               [1.0 + 0.9 * a,  0.7 * b,        0.15 * a],
               [0.25 * b,       0.9 + 0.8 * b,  0.35 * a],
               [0.10 * a,       0.45 * b,       0.8 + 1.0 * a],
           ],
           dtype=float,
       )
       alpha = 0.25
       return c_base + alpha * (s @ s.T)

   theta0 = np.array([1.2, -0.3])
   fk = ForecastKit(function=model, theta0=theta0, cov=cov_fn)

   fisher_std = fk.fisher()
   fisher_full = fk.gaussian_fisher()

   gnd_std = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_std,
       names=["a", "b"],
       labels=[r"a", r"b"],
       label="Standard Fisher",
   )
   gnd_full = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_full,
       names=["a", "b"],
       labels=[r"a", r"b"],
       label="Full Gaussian Fisher",
   )

   dk_yellow = "#e1af00"
   dk_red = "#f21901"
   line_width = 1.5

   plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False

   plotter.triangle_plot(
       [gnd_std, gnd_full],
       params=["a", "b"],
       legend_labels=["Standard Fisher", "Full Fisher"],
       legend_ncol=1,
       filled=[False, False],
       contour_colors=[dk_yellow, dk_red],
       contour_lws=[line_width, line_width],
       contour_ls=["-", "-"],
   )


Notes
-----

- :meth:`ForecastKit.fisher` uses a fixed covariance :math:`C(\theta_0)` and computes
  only the mean-derivative term.

- :meth:`ForecastKit.gaussian_fisher` includes the additional covariance-derivative
  contribution when a parameter-dependent covariance :math:`C(\theta)` is provided.

