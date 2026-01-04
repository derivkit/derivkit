Laplace Approximation
=====================

This section shows how to construct a Laplace approximation to a posterior using
:func:`derivkit.forecasting.laplace.laplace_approximation`.

The Laplace approximation replaces a posterior near its maximum with a local Gaussian.
It is defined by:

- a maximum a posteriori (MAP) point ``theta_map`` (the maximizer of the log-posterior)
- a covariance ``cov`` given by the inverse Hessian of the negative log-posterior
  at ``theta_map`` (up to numerical regularization)

For a conceptual overview of the Laplace approximation and its interpretation, see
:doc:`../../guide/forecasting`.

If you want to generate parameter samples or visualize confidence contours from the
Laplace approximation, see the example :doc:`laplace_contours`, which shows how to
construct GetDist objects and plot Laplace contours.


Basic usage
-----------

Provide a callable returning the **negative log-posterior** (or negative log-likelihood,
if you are doing pure likelihood). The function must return a scalar.

.. doctest:: laplace_basic

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>> # Toy 2D negative log-posterior with a nonzero MAP
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>> theta0 = np.array([0.0, 0.0])
   >>> out = laplace_approximation(function=neg_log_posterior, theta0=theta0)
   >>> theta_map = np.asarray(out["theta_map"], dtype=float)
   >>> cov = np.asarray(out["cov"], dtype=float)
   >>> print(theta_map.shape, cov.shape)
   (2,) (2, 2)
   >>> np.all(np.isfinite(theta_map)) and np.all(np.isfinite(cov))
   True


Interpreting the result
-----------------------

For ``p`` parameters:

- ``theta_map`` has shape ``(p,)``
- the Laplace covariance ``cov`` has shape ``(p, p)``

The Laplace Gaussian is centered on ``theta_map`` by construction.

.. doctest:: laplace_shapes

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>> out = laplace_approximation(function=neg_log_posterior, theta0=np.zeros(2))
   >>> theta_map = np.asarray(out["theta_map"], dtype=float)
   >>> cov = np.asarray(out["cov"], dtype=float)
   >>> theta_map.shape == (2,) and cov.shape == (2, 2)
   True


Choosing a derivative backend
-----------------------------

The Laplace covariance is built from the Hessian of the negative log-posterior.
You can control how derivatives are computed by passing ``method`` and backend-specific options.

All keyword arguments are forwarded to the derivative engines used internally.

.. doctest:: laplace_backend_control

   >>> import numpy as np
   >>> from derivkit.forecasting.laplace import laplace_approximation
   >>> def neg_log_posterior(theta):
   ...     theta = np.asarray(theta, dtype=float)
   ...     mu = np.array([1.3, -0.5])
   ...     cov = np.array([[0.10, -0.03], [-0.03, 0.05]])
   ...     prec = np.linalg.pinv(cov)
   ...     d = theta - mu
   ...     return float(0.5 * d @ prec @ d)
   >>> out = laplace_approximation(
   ...     function=neg_log_posterior,
   ...     theta0=np.zeros(2),
   ...     method="finite",
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> np.all(np.isfinite(out["cov"]))
   True


Notes
-----

- The Laplace approximation is local: it describes the posterior near ``theta_map``.
- If the posterior is strongly non-Gaussian away from the MAP, Laplace contours can be misleading.
- For curved degeneracies, consider DALI or a full sampler instead.
