Laplace approximation
=====================

The Laplace approximation replaces a posterior near its peak with a Gaussian.
It expands the negative log-posterior around an expansion point
:math:`\theta_{\mathrm{MAP}}` (typically the maximum a posteriori estimate)
and uses the local Hessian as a precision matrix:

.. math::

   p(\theta \mid d)
   \approx \mathcal{N}\,\left(\theta_{\mathrm{MAP}}, H^{-1}\right),

   H
   =
   \left.
   \nabla^{2}\,\mathrm{neg\_logposterior}(\theta)
   \right|_{\theta=\theta_{\mathrm{MAP}}}.

This example uses the public utilities in :mod:`derivkit.forecasting.laplace`.

Basic usage (2 parameters)
--------------------------

We use a simple two-dimensional Gaussian–Gaussian model:

.. math::

   y = A\,\theta + \varepsilon, \qquad
   \varepsilon \sim \mathcal{N}(0, \Sigma_{\mathrm{data}}),

   \theta \sim \mathcal{N}(\mu_0, \Sigma_0),

   \theta = (\theta_0, \theta_1).

.. code-block:: python

   import numpy as np

   from derivkit.forecasting.laplace import laplace_approximation

   observed_y = np.array([1.2, -0.4], dtype=np.float64)

   design_matrix = np.array(
       [
           [1.0, 0.5],
           [0.2, 1.3],
       ],
       dtype=np.float64,
   )

   data_cov = np.array(
       [
           [0.30**2, 0.0],
           [0.0, 0.25**2],
       ],
       dtype=np.float64,
   )
   data_prec = np.linalg.inv(data_cov)

   prior_mean = np.array([0.0, 0.0], dtype=np.float64)
   prior_cov = np.array(
       [
           [1.0**2, 0.0],
           [0.0, 1.5**2],
       ],
       dtype=np.float64,
   )
   prior_prec = np.linalg.inv(prior_cov)

   def neg_logposterior(theta: np.ndarray) -> float:
       theta = np.asarray(theta, dtype=np.float64)

       residual = observed_y - design_matrix @ theta
       neg_loglike = 0.5 * (residual @ data_prec @ residual)

       delta_theta = theta - prior_mean
       neg_logprior = 0.5 * (delta_theta @ prior_prec @ delta_theta)

       return float(neg_loglike + neg_logprior)

   # Expansion point (ideally the MAP; using a placeholder here)
   theta_map = np.array([0.0, 0.0], dtype=np.float64)

   result = laplace_approximation(
       neg_logposterior=neg_logposterior,
       theta_map=theta_map,
       method="finite",
       dk_kwargs={"stepsize": 1e-3},
   )

   print("theta_map:", result["theta_map"])
   print("cov:\n", result["cov"])
   print("jitter:", result["jitter"])

Analytic sanity check (Gaussian–Gaussian)
-----------------------------------------

For a Gaussian likelihood with a Gaussian prior, the posterior is exactly Gaussian.
For the linear–Gaussian model above, the exact posterior is given by

.. math::

   \Sigma_{\mathrm{post}}^{-1}
   = A^{\mathsf T}\Sigma_{\mathrm{data}}^{-1}A + \Sigma_0^{-1},

   \Sigma_{\mathrm{post}}
   = \left(\Sigma_{\mathrm{post}}^{-1}\right)^{-1},

   \mu_{\mathrm{post}}
   = \Sigma_{\mathrm{post}}
     \left(A^{\mathsf T}\Sigma_{\mathrm{data}}^{-1}y
     + \Sigma_0^{-1}\mu_0\right).

.. code-block:: python

   posterior_precision = design_matrix.T @ data_prec @ design_matrix + prior_prec
   posterior_cov = np.linalg.inv(posterior_precision)

   posterior_mean = posterior_cov @ (
       design_matrix.T @ data_prec @ observed_y + prior_prec @ prior_mean
   )

   print("Laplace mean:", result["theta_map"])
   print("Exact mean:", posterior_mean)
   print("Laplace covariance:\n", result["cov"])
   print("Exact covariance:\n", posterior_cov)

GetDist contour plot (Laplace Gaussian)
---------------------------------------

Since the Laplace result is already a Gaussian approximation (mean + covariance),
we can plot its 2D contours directly with GetDist using an analytic Gaussian.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecasting.fisher_gaussian import fisher_to_getdist_gaussiannd

   laplace_mean = np.asarray(result["theta_map"], dtype=np.float64)
   laplace_cov = np.asarray(result["cov"], dtype=np.float64)
   laplace_precision = np.linalg.pinv(laplace_cov)

   laplace_gaussian = fisher_to_getdist_gaussiannd(
       theta0=laplace_mean,
       fisher=laplace_precision,
       names=["theta0", "theta1"],
       labels=[r"\theta_0", r"\theta_1"],
       label="Laplace (Gaussian)",
   )

   plotter = plots.get_subplot_plotter()
   plotter.triangle_plot([laplace_gaussian], params=["theta0", "theta1"], filled=True)

Optional: sampled contours (Monte Carlo)
----------------------------------------

If you prefer sample-based plots, you can draw samples from the Laplace Gaussian
and wrap them as :class:`getdist.MCSamples`.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecasting.fisher_gaussian import fisher_to_getdist_samples

   laplace_mean = np.asarray(result["theta_map"], dtype=np.float64)
   laplace_cov = np.asarray(result["cov"], dtype=np.float64)
   laplace_precision = np.linalg.pinv(laplace_cov)

   laplace_mc_samples = fisher_to_getdist_samples(
       theta0=laplace_mean,
       fisher=laplace_precision,
       names=["theta0", "theta1"],
       labels=[r"\theta_0", r"\theta_1"],
       n_samples=30_000,
       seed=0,
       kernel_scale=1.0,
       label="Laplace (samples)",
   )

   plotter = plots.get_subplot_plotter()
   plotter.triangle_plot([laplace_mc_samples], params=["theta0", "theta1"], filled=False)

Notes
-----

- ``theta_map`` should be the MAP when possible. If you only have an initial guess,
  you can still use it as an expansion point, but approximation quality depends
  on how close it is to the posterior peak.
- ``ensure_spd=True`` (default) adds diagonal jitter if needed so that the covariance
  is valid.
- ``method`` and ``dk_kwargs`` are forwarded to the Hessian construction through
  :class:`derivkit.calculus_kit.CalculusKit`.
