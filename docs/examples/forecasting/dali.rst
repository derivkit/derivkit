DALI tensors
============

This section shows how to construct the DALI (Derivative Approximation for
LIkelihoods) tensors using :class:`derivkit.forecast_kit.ForecastKit`.

The DALI expansion extends the Fisher matrix by including higher-order
derivative information, allowing non-Gaussian structure in the likelihood to
be approximated locally around a fiducial parameter point ``theta0``.

In DerivKit, the doublet-DALI expansion returns two tensors, commonly denoted
``G`` and ``H``.

The model must map parameters ``theta`` to a 1D data vector, and the observable
covariance matrix must have shape ``(n, n)``, where ``n`` is the length of the
data vector.

For a conceptual overview of DALI forecasting and its interpretation, see
:doc:`../../guide/forecasting`.

If you want to visualize parameter contours from the DALI tensors, see the
example :doc:`dali_getdist`, which shows how to generate GetDist samples and
plot confidence regions from ``(G, H)``.


DALI tensor shapes
------------------

For ``p`` model parameters:

- the third-order tensor ``G`` has shape ``(p, p, p)``
- the fourth-order tensor ``H`` has shape ``(p, p, p, p)``

These tensors are evaluated at the fiducial parameter point ``theta0``.


Basic usage
-----------

The example below constructs the DALI tensors around a fiducial parameter
point ``theta0`` using a simple toy model.

.. doctest:: dali_basic

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> # Define a simple toy model: R^2 -> R^3
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> # Fiducial parameters and covariance
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> # Build ForecastKit and compute DALI tensors
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> g_tensor, h_tensor = fk.dali()
   >>> print(g_tensor.shape)
   (2, 2, 2)
   >>> print(h_tensor.shape)
   (2, 2, 2, 2)


Choosing a derivative backend
-----------------------------

As with Fisher forecasting, you can control how derivatives are computed by
passing ``method`` and backend-specific options.

All keyword arguments are forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

.. doctest:: dali_backend_control

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> # Define the model
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> # Fiducial setup
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> # Use a finite-difference derivative backend
   >>> g_tensor, h_tensor = fk.dali(
   ...     method="finite",
   ...     n_workers=2,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(g_tensor.shape)
   (2, 2, 2)
   >>> print(h_tensor.shape)
   (2, 2, 2, 2)


Parallel execution
------------------

DALI tensor components can be computed in parallel using ``n_workers``.

This parallelizes derivative evaluations across parameters and tensor entries.

.. doctest:: dali_parallel

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> # Define the model
   >>> def model(theta):
   ...     return np.array([theta[0], theta[1], theta[0] + 2.0 * theta[1]])
   >>> # Enable parallel execution
   >>> fk = ForecastKit(function=model, theta0=np.array([1.0, 2.0]), cov=np.eye(3))
   >>> g_tensor, h_tensor = fk.dali(n_workers=4)
   >>> print(g_tensor.shape)
   (2, 2, 2)
   >>> print(h_tensor.shape)
   (2, 2, 2, 2)


Notes
-----

- DALI tensors are evaluated locally at ``theta0``.
- The expansion captures non-Gaussian structure in parameter space through
  higher-order derivatives of the model.
- In the current implementation, the likelihood is assumed to be Gaussian
  in the data with fixed covariance.
- Higher-order derivatives increase computational cost relative to Fisher.
- Changing the derivative backend affects numerical accuracy but not the
  structure of the DALI expansion.
