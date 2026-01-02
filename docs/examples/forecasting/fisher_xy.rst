X–Y Gaussian Fisher matrix
=========================

In some forecasting problems, the model prediction for the observed data depends not
only on the parameters of interest but also on *measured inputs* that carry
uncertainty. Treating those inputs as exact (standard Fisher on the outputs only)
can lead to overly optimistic constraints.

The **X–Y Gaussian Fisher matrix** extends the standard Gaussian Fisher formalism to
the case where **both inputs and outputs are noisy and may be correlated**, without
explicitly introducing the inputs as latent nuisance parameters.

Model and covariance structure
------------------------------

We consider a model that predicts an output vector :math:`y` given measured inputs
:math:`x` and parameters :math:`\theta`:

.. math::

   y \sim \mathcal{N}\!\bigl(\mu_{xy}(x, \theta),\, C\bigr).

Measurement errors on :math:`x` and :math:`y` are described by a joint Gaussian
covariance with block structure:

.. math::

   C =
   \begin{pmatrix}
       C_{xx} & C_{xy} \\
       C_{xy}^\mathsf{T} & C_{yy}
   \end{pmatrix}.

Here:

* :math:`C_{xx}` is the covariance of the input measurements,
* :math:`C_{yy}` is the covariance of the output measurements,
* :math:`C_{xy}` is the cross-covariance between input and output measurement errors.

Propagating input uncertainty
-----------------------------

Rather than marginalizing over the unknown true inputs explicitly, the X–Y formalism
propagates input uncertainty into the output space using a local linearization of the
model mean around the measured inputs :math:`x_0`:

.. math::

   \mu_{xy}(x, \theta)
   \approx
   \mu_{xy}(x_0, \theta) + T (x - x_0),

with sensitivity matrix

.. math::

   T \equiv \left.\frac{\partial \mu_{xy}}{\partial x}\right|_{(x_0,\theta)}.

This yields an **effective output covariance**:

.. math::

   R(\theta)
   =
   C_{yy}
   - C_{xy}^\mathsf{T} T^\mathsf{T}
   - T C_{xy}
   + T C_{xx} T^\mathsf{T}.

The standard output covariance :math:`C_{yy}` is thus replaced by :math:`R(\theta)` in
the Gaussian likelihood.

Fisher matrix
-------------

With the effective covariance, the likelihood for :math:`y` has the standard Gaussian
form with mean :math:`\mu_{xy}(x_0,\theta)` and covariance :math:`R(\theta)`. The
corresponding Fisher matrix is:

.. math::

   F_{ij}
   =
   \mu_{,i}^{\mathsf{T}} R^{-1} \mu_{,j}
   +
   \frac{1}{2}\mathrm{Tr}\!\left[
       R^{-1} R_{,i} R^{-1} R_{,j}
   \right],

where :math:`\mu_{,i} \equiv \partial \mu_{xy}(x_0,\theta)/\partial\theta_i`. In the
implementation, the covariance blocks :math:`C_{xx}`, :math:`C_{xy}`, and :math:`C_{yy}`
are treated as fixed; parameter dependence enters through the sensitivity matrix
:math:`T(x_0,\theta)` and therefore through :math:`R(\theta)`.

Interpretation
--------------

* **Standard Fisher** treats inputs as exact and uses :math:`C_{yy}` directly.
* **X–Y Fisher** accounts for input uncertainty by inflating and reshaping the output
  covariance via the model response :math:`T`, which typically broadens and can rotate
  parameter constraints.

This formalism follows the generalized Fisher matrix treatment of Heavens et al.
(2014), https://arxiv.org/abs/1404.2854.


Demo: compare standard vs X–Y Fisher
====================================

This example compares:

1. A **standard Fisher matrix** built for :math:`\mu(\theta) = \mu_{xy}(x_0,\theta)`
   using only :math:`C_{yy}` (treating :math:`x` as exact).
2. An **X–Y Fisher matrix** that propagates input uncertainty through the effective
   covariance :math:`R(\theta)`.

Both Fisher matrices are converted to GetDist :class:`getdist.gaussian_mixtures.GaussianND`
objects and plotted together on a triangle plot.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from getdist import plots

   from derivkit.forecast_kit import ForecastKit
   from derivkit.forecasting.fisher_xy import build_xy_gaussian_fisher_matrix
   from derivkit.forecasting.getdist_fisher_samples import fisher_to_getdist_gaussiannd


   def mu_xy(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
   """Toy model: ``y = mu_xy(x, theta)`` in ``R^3``, with ``x`` in ``R^2``, ``theta`` in ``R^2``."""
       x = np.asarray(x)
       theta = np.asarray(theta)
       a, b = theta
       x0, x1 = x
       return np.array(
           [
               a * x0 + b,
               a * (x1**2) + 0.5 * b,
               a * (x0 + x1) + b * x0,
           ],
           dtype=float,
       )

   def mu_theta(theta: np.ndarray, *, x0: np.ndarray) -> np.ndarray:
   """Computes ``mu_xy(x0, theta)`` for fixed ``x0``."""
       return mu_xy(x0, theta)


   #  Fiducial (reference) parameter vector where we evaluate the Fisher matrix.
   # This is the expansion point in parameter space.
   theta0 = np.array([1.2, -0.3])

   # Reference input values at which we evaluate mu_xy(x0, theta) and linearize in x.
   # These are the measured inputs (noisy) used by the model.
   x0 = np.array([0.7, 1.1])  # x = (x0, x1)

   # Joint covariance blocks for (x, y)
   cxx = np.array([[0.04, 0.01], [0.01, 0.09]])
   cyy = np.array(
       [
           [0.20, 0.03, 0.00],
           [0.03, 0.15, 0.02],
           [0.00, 0.02, 0.10],
       ]
   )
   cxy = np.array(
       [
           [0.01, -0.005, 0.0],
           [0.0, 0.004, -0.006],
       ]
   )

   names = ["a", "b"]
   labels = [r"a", r"b"]

   fk = ForecastKit(function=lambda th: mu_theta(th, x0=x0), theta0=theta0, cov=cyy)
   fisher_std = fk.fisher()  # standard fixed-cov Fisher

   gnd_std = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_std,
       names=names,
       labels=labels,
       label="Standard Fisher (ignore X)",
   )

   fisher_xy = build_xy_gaussian_fisher_matrix(
       theta0=theta0,
       x0=x0,
       mu_xy=mu_xy,
       cxx=cxx,
       cxy=cxy,
       cyy=cyy,
       term="both",  # mean + cov terms (R depends on theta via T)
   )

   gnd_xy = fisher_to_getdist_gaussiannd(
       theta0=theta0,
       fisher=fisher_xy,
       names=names,
       labels=labels,
       label="X–Y Fisher (propagate X)",
   )

   # Plot comparison (GetDist triangle)
   g = plots.get_subplot_plotter()
   colors = ["#f21901", "#3b9ab2"]
   legend_labels = [
        f"Standard Fisher",
        f"X-Y Fisher",
        ]
   triangle_kwargs = {
        "contour_colors": colors,
        "filled": [False, False],
        "contour_ls": ["-", "-"],
        "contour_lws": [2, 2],
        }
   g.triangle_plot([gnd_std, gnd_xy], filled=True)
   plt.show()
   # g.export("demo_getdist_compare_standard_vs_xy_fisher.pdf")
   # out = "demo_getdist_compare_standard_vs_xy_fisher.png"
   # g.export(out)
   # print(f"Wrote {out}")


