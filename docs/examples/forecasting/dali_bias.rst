
.. |dklogo| image:: ../../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| DALI bias
==================

Systematic differences between observed data and a model can shift inferred
parameters. `Fisher bias <docs.derivkit.org/main/examples/forecasting/fisher_bias.html`_ predicts these shifts using a linear approximation
around the fiducial parameters. For nonlinear models, however, the parameter
response can depart substantially from this prediction.

DALI bias extends Fisher bias by retaining higher-order derivatives of the
model. It provides predictions for shifts in both the
`maximum a posteriori <https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>`_ (MAP) estimator and
the posterior mean.

This example illustrates Fisher and DALI bias calculations using a
two-parameter nonlinear model with a non-Gaussian posterior.


Nonlinear model and systematic mismatch
---------------------------------------

Model definition
~~~~~~~~~~~~~~~~

We consider two parameters, :math:`\boldsymbol{\theta}=(u,v)`, and two
observables,

.. math::

   \boldsymbol{\nu}(\boldsymbol{\theta})
   =
   \begin{pmatrix}
   1 + 1.4\Delta u + 0.15\Delta v
   + 0.20\Delta u^2 + 0.06\Delta u\Delta v
   \\
   0.8 + 10\left(
   \Delta v + 1.8\Delta u^2
   - 0.3\Delta u^3 + 0.20\Delta u\Delta v
   \right)
   \end{pmatrix},

where :math:`\Delta u=u-u_0`, :math:`\Delta v=v-v_0`, and

.. math::

   \boldsymbol{\theta}_0 =
   \begin{pmatrix}
   0.15 \\ 0.05
   \end{pmatrix}.

The first observable provides a broad constraint on the parameters, while
the second constrains a curved parameter combination. We assume a fixed
covariance with `fractional errors <https://en.wikipedia.org/wiki/Propagation_of_uncertainty>`_ of 10% and a 
`correlation coefficient <https://en.wikipedia.org/wiki/Correlation_coefficient>`_ of 0.3.

.. doctest:: dali_bias

   >>> import numpy as np
   >>> from derivkit import ForecastKit
   >>> theta0 = np.array([0.15, 0.05])
   >>> def model(theta):
   ...     u, v = theta
   ...     du = u - theta0[0]
   ...     dv = v - theta0[1]
   ...     return np.array([
   ...         1.0 + 1.4 * du + 0.15 * dv
   ...         + 0.20 * du**2 + 0.06 * du * dv,
   ...         0.8 + 10.0 * (
   ...             dv + 1.8 * du**2 - 0.3 * du**3
   ...             + 0.20 * du * dv
   ...         ),
   ...     ])
   >>> sigma = 0.10 * np.abs(model(theta0))
   >>> correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
   >>> cov = np.outer(sigma, sigma) * correlation
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)

The nonlinear dependence of the observables produces a curved posterior
geometry that is not generally captured by a Gaussian approximation.


Systematic displacement
~~~~~~~~~~~~~~~~~~~~~~~

We introduce an additive systematic displacement in data space,

.. math::

   \Delta\boldsymbol{\nu}(A)
   =
   A
   \begin{pmatrix}
   0.020 \\ -0.003
   \end{pmatrix},

where :math:`A` controls the systematic amplitude.

The model used for inference remains unchanged. The systematic modifies
only the data vector, allowing us to study the resulting parameter shift.

For the following calculations, we choose :math:`A=10`.

.. doctest:: dali_bias

   >>> amplitude = 10.0
   >>> systematic = np.array([0.020, -0.003])
   >>> data_unbiased = model(theta0)
   >>> data_biased = data_unbiased + amplitude * systematic
   >>> dn = fk.delta_nu(
   ...     data_unbiased=data_unbiased,
   ...     data_biased=data_biased,
   ... )
   >>> print(np.round(dn, 3))
   [ 0.2  -0.03]


Systematic bias tensors
-----------------------

Fisher bias
~~~~~~~~~~~

Fisher bias maps a systematic displacement in data space onto a
parameter shift,

.. math::

   \Delta\theta_{\mathrm{F}}^a
   =
   (F^{-1})^{ab}b_b,

where

.. math::

   F_{ab}
   =
   \nu_{,a}^{\mathrm T}C^{-1}\nu_{,b},
   \qquad
   b_a
   =
   \nu_{,a}^{\mathrm T}C^{-1}\Delta\nu.

The Fisher matrix describes the local parameter sensitivity, while
the bias vector describes the coupling between the systematic and
the first derivatives of the model.

.. doctest:: dali_bias

   >>> fisher = fk.fisher()
   >>> bias_vector, fisher_shift = fk.fisher_bias(
   ...     fisher_matrix=fisher,
   ...     delta_nu=dn,
   ... )

The resulting parameter shift is linear in the systematic amplitude.


DALI bias
~~~~~~~~~

DALI generalizes the Fisher calculation by retaining higher derivatives
of the model. The systematic bias tensors are

.. math::

   b_{a_1\ldots a_n}
   =
   \Delta\nu^{\mathrm T}
   C^{-1}
   \nu_{,a_1\ldots a_n}.

The first-order tensor corresponds to the Fisher bias vector.
Higher-order tensors describe how the systematic couples to the
nonlinear model response.

We calculate the bias tensors up to (and including) third order.

.. doctest:: dali_bias

   >>> bias = fk.dali_bias_tensor(
   ...     delta_nu=dn,
   ...     bias_order=3,
   ... )
   >>> sorted(bias)
   [1, 2, 3]

The corresponding unbiased DALI forecast tensors can be obtained
independently:

.. doctest:: dali_bias

   >>> dali = fk.dali(forecast_order=3)
   >>> sorted(dali)
   [1, 2, 3]

Together, these tensors describe the nonlinear posterior geometry
and its response to the systematic displacement.


Parameter shifts
----------------

MAP shift
~~~~~~~~~

The MAP shift describes how the maximum of the posterior moves when
the systematic is introduced.

DALI expands this response in successive orders of the systematic
amplitude,

.. math::

   \Delta\boldsymbol{\theta}_{\mathrm{MAP}}
   =
   \Delta\boldsymbol{\theta}^{(1)}
   +
   \Delta\boldsymbol{\theta}^{(2)}
   +
   \Delta\boldsymbol{\theta}^{(3)}
   + \cdots.

The first-order contribution reproduces Fisher bias, while the
higher-order contributions account for nonlinear corrections.

.. doctest:: dali_bias

   >>> map_shift, map_metadata = fk.dali_map_shift(
   ...     delta_nu=dn,
   ...     expansion_order=3,
   ...     return_metadata=True,
   ... )
   >>> sorted(map_metadata["terms"])
   [1, 2, 3]

The metadata separates the individual response orders:

.. code-block:: python

   print("Fisher shift:", fisher_shift)
   print("DALI MAP shift:", map_shift)

   for order, term in map_metadata["terms"].items():
       print(f"Order {order}:", term)

The predicted biased MAP is also available as
``map_metadata["estimator"]``.


Posterior mean shift
~~~~~~~~~~~~~~~~~~~~

For a non-Gaussian posterior, the MAP and posterior mean generally
differ. The systematic shift of the posterior mean is defined as

.. math::

   \Delta\overline{\boldsymbol{\theta}}
   =
   \overline{\boldsymbol{\theta}}_{\mathrm{biased}}
   -
   \overline{\boldsymbol{\theta}}_{\mathrm{unbiased}}.

DerivKit provides an analytic approximation using a local expansion
around the MAP.

.. doctest:: dali_bias

   >>> mean_shift, mean_metadata = fk.dali_mean_shift(
   ...     delta_nu=dn,
   ...     return_metadata=True,
   ... )

The metadata contains the predicted baseline and biased posterior means.

.. code-block:: python

   print("Unbiased mean:", mean_metadata["baseline"])
   print("Biased mean:", mean_metadata["estimator"])
   print("Mean shift:", mean_shift)

Unlike the MAP shift, the posterior mean shift is not generally equal
to the displacement of the biased mean from :math:`\boldsymbol{\theta}_0`.


Posterior mean response
~~~~~~~~~~~~~~~~~~~~~~~

An alternative approach expands the posterior mean directly around
the unbiased reference posterior.

The method uses samples or integration nodes representing that
posterior. Here we sample the unbiased DALI posterior using ``emcee``
through the :class:`derivkit.forecast_kit.ForecastKit` interface.

.. doctest:: dali_bias

   >>> posterior_unbiased = fk.getdist_dali_emcee(
   ...     dali=dali,
   ...     names=["u", "v"],
   ...     labels=["u", "v"],
   ...     forecast_order=3,
   ...     n_walkers=32,
   ...     n_steps=3000,
   ...     burn=500,
   ...     thin=5,
   ...     seed=11,
   ...     label="Unbiased DALI",
   ... )

The MCMC samples represent the reference posterior with equal
sample weights, so no explicit weights are needed for the response
calculation.

.. doctest:: dali_bias

   >>> response_shift, response_metadata = fk.dali_mean_response(
   ...     delta_nu=dn,
   ...     theta=posterior_unbiased.samples,
   ...     forecast_order=3,
   ...     response_order=4,
   ...     return_metadata=True,
   ... )

The response expansion can be evaluated through fourth order in the
systematic amplitude. The contributions are available in
``response_metadata["terms"]``.

The two posterior mean methods differ in their construction:
``dali_mean_shift`` uses a local expansion around the MAP, whereas
:meth:`derivkit.forecast_kit.ForecastKit.dali_mean_response` evaluates the response using a reference
posterior distribution.


Posterior visualization
-----------------------

We compare the unbiased and biased Fisher and DALI posteriors.
The Fisher approximation describes a Gaussian posterior whose mean
shifts in response to the systematic displacement, while its
covariance remains unchanged.

The DALI approximation retains the nonlinear posterior geometry.
Both DALI posteriors are sampled using ``emcee``, with the systematic
bias tensors included only in the biased posterior.

.. plot::
   :include-source: True
   :width: 520

   import numpy as np
   from getdist import plots
   from derivkit import ForecastKit

   theta0 = np.array([0.15, 0.05])

   def model(theta):
       u, v = theta
       du = u - theta0[0]
       dv = v - theta0[1]
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv
           + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (
               dv + 1.8 * du**2 - 0.3 * du**3
               + 0.20 * du * dv
           ),
       ])

   sigma = 0.10 * np.abs(model(theta0))
   correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
   cov = np.outer(sigma, sigma) * correlation

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   dn = 10.0 * np.array([0.020, -0.003])

   # Fisher forecast and systematic bias.
   fisher = fk.fisher()
   fisher_bias_vector, fisher_shift = fk.fisher_bias(
       fisher_matrix=fisher,
       delta_nu=dn,
   )

   # DALI forecast and systematic bias tensors.
   dali = fk.dali(forecast_order=3)
   bias = fk.dali_bias_tensor(
       delta_nu=dn,
       bias_order=3,
   )

   # Unbiased Fisher posterior.
   posterior_fisher_unbiased = fk.getdist_fisher_gaussian(
       fisher=fisher,
       names=["u", "v"],
       labels=["u", "v"],
       label="Unbiased Fisher",
   )

   # Biased Fisher posterior.
   posterior_fisher_biased = fk.getdist_fisher_gaussian(
       fisher=fisher,
       bias=fisher_bias_vector,
       names=["u", "v"],
       labels=["u", "v"],
       label="Biased Fisher",
   )

   # Unbiased DALI posterior.
   posterior_dali_unbiased = fk.getdist_dali_emcee(
       dali=dali,
       names=["u", "v"],
       labels=["u", "v"],
       forecast_order=3,
       label="Unbiased DALI",
   )

   # Biased DALI posterior.
   posterior_dali_biased = fk.getdist_dali_emcee(
       dali=dali,
       bias=bias,
       names=["u", "v"],
       labels=["u", "v"],
       forecast_order=3,
       label="Biased DALI",
   )

   # DerivKit color palette.
   dk_blue = "#3b9ab2"
   dk_yellow = "#e1af00"
   dk_red = "#f21901"
   line_width = 1.5

   plotter = plots.get_subplot_plotter(width_inch=3.6)
   plotter.settings.linewidth_contour = line_width
   plotter.settings.linewidth = line_width
   plotter.settings.figure_legend_frame = False

   plotter.triangle_plot(
       [
           posterior_fisher_unbiased,
           posterior_fisher_biased,
           posterior_dali_unbiased,
           posterior_dali_biased,
       ],
       params=["u", "v"],
       legend_labels=[
           "Unbiased Fisher",
           "Biased Fisher",
           "Unbiased DALI",
           "Biased DALI",
       ],
       legend_ncol=1,
       filled=[False, False, False, False],
       contour_colors=[dk_blue, dk_blue, dk_yellow, dk_red],
       contour_ls=["--", "-", "-", "-"],
   )

The dashed blue and solid yellow contours represent the unbiased Fisher and
DALI posteriors, respectively. The solid blue and red contours show
their corresponding biased posteriors.

Fisher bias translates the Gaussian posterior without changing its
covariance. DALI bias instead modifies the nonlinear posterior through
higher-order systematic response terms.


Systematic amplitude dependence
-------------------------------

To explore the nonlinear response, we vary the systematic amplitude
from :math:`A=0` to :math:`A=20`.

For each amplitude, we compare the Fisher prediction, the DALI MAP
prediction, and the MAP obtained by minimizing the full nonlinear
likelihood.

.. plot::
   :include-source: True
   :width: 620

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.optimize import minimize
   from derivkit import ForecastKit

   theta0 = np.array([0.15, 0.05])

   def model(theta):
       u, v = theta
       du = u - theta0[0]
       dv = v - theta0[1]
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv
           + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (
               dv + 1.8 * du**2 - 0.3 * du**3
               + 0.20 * du * dv
           ),
       ])

   sigma = 0.10 * np.abs(model(theta0))
   correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
   cov = np.outer(sigma, sigma) * correlation
   inv_cov = np.linalg.inv(cov)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   fisher = fk.fisher()

   systematic = np.array([0.020, -0.003])
   amplitudes = np.linspace(0.0, 20.0, 21)

   fisher_shifts = []
   dali_shifts = []
   exact_shifts = []

   for amplitude in amplitudes:
       dn = amplitude * systematic

       _, fisher_shift = fk.fisher_bias(
           fisher_matrix=fisher,
           delta_nu=dn,
       )

       dali_shift = fk.dali_map_shift(
           delta_nu=dn,
           expansion_order=3,
       )

       data = model(theta0) + dn

       def negative_loglike(theta):
           residual = data - model(theta)
           return residual @ inv_cov @ residual

       result = minimize(negative_loglike, theta0, method="BFGS")

       fisher_shifts.append(fisher_shift)
       dali_shifts.append(dali_shift)
       exact_shifts.append(result.x - theta0)

   fisher_shifts = np.asarray(fisher_shifts)
   dali_shifts = np.asarray(dali_shifts)
   exact_shifts = np.asarray(exact_shifts)

   dk_yellow = "#e1af00"
   dk_red = "#f21901"
   dk_dark = "#555555"
   line_width = 1.5

   fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

   for i, ax in enumerate(axes):
       ax.plot(
           amplitudes,
           fisher_shifts[:, i],
           "--",
           color=dk_dark,
           linewidth=line_width,
           label="Fisher",
       )
       ax.plot(
           amplitudes,
           dali_shifts[:, i],
           "-",
           color=dk_yellow,
           linewidth=line_width,
           label="DALI MAP",
       )
       ax.plot(
           amplitudes,
           exact_shifts[:, i],
           "o",
           color=dk_red,
           markersize=3,
           label="Numerical MAP",
       )

       ax.set_xlabel("Systematic amplitude $A$")
       ax.set_ylabel(
           r"$\Delta u$" if i == 0 else r"$\Delta v$"
       )

   axes[0].legend(frameon=False)
   fig.tight_layout()
   plt.show()

The Fisher prediction is linear in the systematic amplitude, while
the DALI prediction includes higher-order corrections. The numerical
MAP provides a reference for the full nonlinear model.


Numerical comparison
--------------------

The accuracy of the DALI predictions can be assessed by comparing them
with direct inference using the original nonlinear model.

For the MAP, we minimize the negative log likelihood. For the posterior
mean, we integrate the corresponding posterior over parameter space.

The following example compares the Fisher shift, DALI MAP shift, and
both DALI posterior mean approximations with their numerical references.

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize
   from derivkit import ForecastKit

   theta0 = np.array([0.15, 0.05])

   def model(theta):
       u, v = theta
       du = u - theta0[0]
       dv = v - theta0[1]
       return np.array([
           1.0 + 1.4 * du + 0.15 * dv
           + 0.20 * du**2 + 0.06 * du * dv,
           0.8 + 10.0 * (
               dv + 1.8 * du**2 - 0.3 * du**3
               + 0.20 * du * dv
           ),
       ])

   sigma = 0.10 * np.abs(model(theta0))
   correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
   cov = np.outer(sigma, sigma) * correlation
   inv_cov = np.linalg.inv(cov)

   fk = ForecastKit(function=model, theta0=theta0, cov=cov)

   dn = 10.0 * np.array([0.020, -0.003])
   data_unbiased = model(theta0)
   data_biased = data_unbiased + dn

   fisher = fk.fisher()

   _, fisher_shift = fk.fisher_bias(
       fisher_matrix=fisher,
       delta_nu=dn,
   )

   map_shift = fk.dali_map_shift(
       delta_nu=dn,
       expansion_order=3,
   )

   local_mean_shift = fk.dali_mean_shift(delta_nu=dn)

   dali = fk.dali(forecast_order=3)

   reference = fk.getdist_dali_emcee(
       dali=dali,
       names=["u", "v"],
       labels=["u", "v"],
       forecast_order=3,
       n_walkers=32,
       n_steps=3000,
       burn=500,
       thin=5,
       seed=11,
       label="Unbiased DALI",
   )

   response_mean_shift = fk.dali_mean_response(
       delta_nu=dn,
       theta=reference.samples,
       forecast_order=3,
       response_order=4,
   )

   axis_u = np.linspace(-0.5, 0.8, 151)
   axis_v = np.linspace(-0.5, 0.8, 151)
   uu, vv = np.meshgrid(axis_u, axis_v, indexing="ij")
   nodes = np.column_stack([uu.ravel(), vv.ravel()])
   predictions = np.array([model(theta) for theta in nodes])

   def posterior_weights(data):
       residual = data - predictions
       chi2 = np.einsum(
           "ni,ij,nj->n",
           residual,
           inv_cov,
           residual,
       )
       return np.exp(-0.5 * (chi2 - chi2.min()))

   weights0 = posterior_weights(data_unbiased)
   weights1 = posterior_weights(data_biased)

   mean0 = np.average(nodes, axis=0, weights=weights0)
   mean1 = np.average(nodes, axis=0, weights=weights1)
   exact_mean_shift = mean1 - mean0

   def objective(theta):
       residual = data_biased - model(theta)
       return residual @ inv_cov @ residual

   exact_map_shift = (
       minimize(objective, theta0, method="BFGS").x - theta0
   )

   print("Fisher shift:", fisher_shift)
   print("DALI MAP shift:", map_shift)
   print("Numerical MAP shift:", exact_map_shift)
   print()
   print("Local DALI mean shift:", local_mean_shift)
   print("DALI mean response:", response_mean_shift)
   print("Numerical mean shift:", exact_mean_shift)

The MAP and posterior mean are distinct estimators and should be
compared separately.

DALI improves on the local Fisher approximation by retaining nonlinear
information, but its predictions remain approximate. Their accuracy
depends on the posterior geometry and the magnitude of the systematic
displacement. Direct optimization and posterior integration provide
independent references for assessing that accuracy.
