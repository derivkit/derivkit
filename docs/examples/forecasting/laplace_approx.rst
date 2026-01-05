Laplace Approximation
=====================

This section shows how to construct a Laplace approximation to a posterior using
:func:`derivkit.forecasting.laplace.laplace_approximation`.

The Laplace approximation replaces a posterior near its maximum with a local Gaussian.
It is defined by:

- an expansion point ``theta_map`` (typically the maximum a posteriori, MAP)
- a Hessian ``hessian`` of the negative log-posterior at ``theta_map``
- a covariance ``cov`` given by the inverse Hessian (up to numerical regularization)

For a conceptual overview of the Laplace approximation and its interpretation, see
:doc:`../../guide/forecasting`.

If you want to generate parameter samples or visualize confidence contours from the
Laplace approximation, see the example :doc:`laplace_contours`, which shows how to
construct GetDist objects and plot Laplace contours.


Basic usage
-----------

Provide a callable returning the negative log-posterior and a point ``theta_map``
around which you want the Gaussian approximation. In practice, ``theta_map`` is often
the MAP obtained from an optimizer, but for toy problems it may be known analytically.

The function must accept a 1D parameter vector and return a scalar.

.. doctest:: laplace_basic

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>>
   >>> # Toy 2D negative log-posterior: a Gaussian centered at mu
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>>
   >>> # For this toy model, the MAP is known: theta_map = mu
   >>> theta_map_in = np.array([1.3, -0.5])
   >>> out = laplace_approximation(neg_logposterior=neg_log_posterior, theta_map=theta_map_in)
   >>> theta_map = np.asarray(out["theta_map"], dtype=float)
   >>> cov = np.asarray(out["cov"], dtype=float)
   >>> print(theta_map.shape, cov.shape)
   (2,) (2, 2)
   >>> bool(np.all(np.isfinite(theta_map)) and np.all(np.isfinite(cov)))
   True


Interpreting the result
-----------------------

For ``p`` parameters:

- ``theta_map`` has shape ``(p,)``
- the Hessian ``hessian`` has shape ``(p, p)``
- the Laplace covariance ``cov`` has shape ``(p, p)``

The Laplace Gaussian is centered on ``theta_map`` by construction.

.. doctest:: laplace_shapes

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>>
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>>
   >>> out = laplace_approximation(
   ...     neg_logposterior=neg_log_posterior,
   ...     theta_map=np.array([1.3, -0.5]),
   ... )
   >>> theta_map = np.asarray(out["theta_map"], dtype=float)
   >>> cov = np.asarray(out["cov"], dtype=float)
   >>> hessian = np.asarray(out["hessian"], dtype=float)
   >>> bool(theta_map.shape == (2,) and cov.shape == (2, 2) and hessian.shape == (2, 2))
   True


Choosing a derivative backend
-----------------------------

The Laplace covariance is built from the Hessian of the negative log-posterior.
You can control how derivatives are computed by passing ``method`` and a dictionary
of derivative-engine options via ``dk_kwargs``.

.. doctest:: laplace_backend_control

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>>
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>>
   >>> out = laplace_approximation(
   ...     neg_logposterior=neg_log_posterior,
   ...     theta_map=np.array([1.3, -0.5]),
   ...     method="finite",
   ...     dk_kwargs={
   ...         "stepsize": 1e-2,
   ...         "num_points": 5,
   ...         "extrapolation": "ridders",
   ...         "levels": 4,
   ...     },
   ... )
   >>> bool(np.all(np.isfinite(out["cov"])) and np.all(np.isfinite(out["hessian"])))
   True


Notes
-----

- The Laplace approximation is local: it describes the posterior near ``theta_map``.
- If the posterior is strongly non-Gaussian away from the MAP, Laplace contours can be misleading.
- For curved degeneracies, consider DALI or a full sampler instead.
