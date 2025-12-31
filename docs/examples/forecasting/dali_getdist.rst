DALI sampling and GetDist integration
=====================================

DerivKit can convert a DALI-expanded posterior into GetDist-ready objects for
plotting (triangle/corner plots) and lightweight downstream analysis.

The public entry points are:

- :func:`derivkit.forecasting.getdist.dali_to_getdist_emcee` (MCMC sampling with ``emcee``)
- :func:`derivkit.forecasting.getdist.dali_to_getdist_importance` (fast importance sampling)

Both functions return a :class:`getdist.MCSamples` instance.

What is being sampled?
----------------------

DALI approximates the log-posterior around a fiducial point ``theta0`` using
the Fisher matrix ``F`` and higher-order tensors ``G`` (and optionally ``H``).
The target log-posterior is evaluated with
:func:`derivkit.forecasting.expansions.logposterior_dali`.

You provide:

- ``theta0``: fiducial parameters, shape ``(p,)``
- ``fisher``: Fisher matrix, shape ``(p, p)``
- ``g_tensor``: DALI ``G`` tensor, shape ``(p, p, p)``
- ``h_tensor``: optional DALI ``H`` tensor, shape ``(p, p, p, p)``

The returned GetDist object stores log-likelihoods in GetDist convention,
i.e. :attr:`getdist.MCSamples.loglikes` contains ``-log(posterior)`` up to
an additive constant.

MCMC sampling with emcee
------------------------

You can sample the DALI log-posterior with ``emcee`` and return the result as
GetDist :class:`getdist.MCSamples`.

Walkers are initialized from a Fisher–Gaussian cloud around ``theta0`` and then
evolved with :class:`emcee.EnsembleSampler`. Optional priors and support bounds
are applied through the target log-posterior.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecasting.getdist import dali_to_getdist_emcee

   # Example inputs (replace with your outputs from ForecastKit / build_dali)
   theta0 = np.array([0.3, 0.8])
   fisher = np.array([[100.0, 10.0],
                      [10.0,  50.0]])
   g = np.zeros((2, 2, 2))
   h = None

   names = ["Om", "s8"]
   labels = [r"\Omega_m", r"\sigma_8"]

   samples = dali_to_getdist_emcee(
       theta0=theta0,
       fisher=fisher,
       g_tensor=g,
       h_tensor=h,
       names=names,
       labels=labels,
       label="DALI (emcee)",
   )

   gplot = plots.get_subplot_plotter()
   gplot.triangle_plot([samples], params=names, filled=True)

Priors and parameter support
----------------------------

You can include priors in three ways:

- ``logprior(theta)``: provide a custom callable, OR
- ``prior_terms`` and/or ``prior_bounds``: build a prior with
  :func:`derivkit.forecasting.priors.core.build_prior`.

In addition, you may specify ``hard_bounds`` for explicit rejection /
initialization bounds. If both ``prior_bounds`` and ``hard_bounds`` are given,
DerivKit uses their intersection as the effective support.

.. code-block:: python

   import numpy as np
   from derivkit.forecasting.getdist import dali_to_getdist_emcee

   theta0 = np.array([0.3, 0.8])
   fisher = np.array([[100.0, 10.0],
                      [10.0,  50.0]])
   g = np.zeros((2, 2, 2))
   h = None

   names = ["Om", "s8"]
   labels = [r"\Omega_m", r"\sigma_8"]

   prior_bounds = [(0.05, 0.6), (0.4, 1.4)]  # top-hat support

   samples = dali_to_getdist_emcee(
       theta0=theta0,
       fisher=fisher,
       g_tensor=g,
       h_tensor=h,
       names=names,
       labels=labels,
       prior_bounds=prior_bounds,
   )

   gplot = plots.get_subplot_plotter()
   gplot.triangle_plot([samples], params=names, filled=True)

Importance sampling (quick plots)
---------------------------------

Importance sampling draws a cloud of samples from a Fisher–Gaussian proposal
(kernel) centered on ``theta0`` and reweights them to match the DALI posterior.
This is typically the fastest route to a usable triangle plot.

.. code-block:: python

   import numpy as np
   from getdist import plots

   from derivkit.forecasting.getdist import dali_to_getdist_importance

   theta0 = np.array([0.3, 0.8])
   fisher = np.array([[100.0, 10.0],
                      [10.0,  50.0]])
   g = np.zeros((2, 2, 2))
   h = np.zeros((2, 2, 2, 2))

   names = ["Om", "s8"]
   labels = [r"\Omega_m", r"\sigma_8"]

   samples = dali_to_getdist_importance(
       theta0=theta0,
       fisher=fisher,
       g_tensor=g,
       h_tensor=h,
       names=names,
       labels=labels,
       label="DALI (importance)",
   )

   gplot = plots.get_subplot_plotter()
   gplot.triangle_plot([samples], params=names, filled=True)

Notes and tips
--------------

- ``kernel_scale`` (importance sampling) controls the width of the Fisher–Gaussian
  proposal. If weights become extremely uneven, try increasing the scale slightly.
- When using bounds, make sure ``theta0`` lies inside the effective support; otherwise
  walker initialization or importance sampling rejection may remove everything.
- If you do not provide ``prior_terms``/``prior_bounds``/``logprior``, the default prior
  is flat (often improper). For plotting, it is usually better to specify at least
  bounded support via ``prior_bounds``.
