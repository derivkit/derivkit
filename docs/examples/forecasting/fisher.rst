Fisher Matrix
=============

This section shows how to construct a Fisher information matrix using
:class:`derivkit.forecast_kit.ForecastKit`.

The Fisher matrix describes the local curvature of a Gaussian likelihood with
respect to the model parameters. It is commonly used for forecasting parameter
constraints and estimating parameter covariances.

In DerivKit, the Fisher matrix is built from:

- a model function mapping parameters ``theta`` to a 1D data vector
- a fixed observable covariance matrix ``cov``

The primary interface for Fisher forecasting is
:meth:`derivkit.forecast_kit.ForecastKit.fisher`.

For a conceptual overview of Fisher forecasting, including interpretation,
Fisher bias, and higher-order (DALI) extensions, see
:doc:`../../guide/forecasting`.

If you want to generate parameter samples or visualize confidence contours
from the Fisher matrix, see the example :doc:`fisher_getdist`, which shows how
to construct GetDist samples and plot Fisher ellipses.


Basic usage
-----------

The model must return a 1D array of length ``n`` for an input parameter vector
``theta`` of length ``p``. The covariance must have shape ``(n, n)``.

.. doctest:: fisher_basic

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
   >>> # Build ForecastKit and compute Fisher matrix
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> fisher = fk.fisher()
   >>> print(fisher)
   [[2. 2.]
    [2. 5.]]
   >>> print(fisher.shape)
   (2, 2)


Interpreting the result
-----------------------

For ``p`` parameters, the Fisher matrix has shape ``(p, p)``.

- The diagonal elements correspond to inverse variances (up to correlations).
- The inverse Fisher matrix approximates the parameter covariance near
  ``theta0``.

.. doctest:: fisher_inverse

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> # Same toy model as above
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> # Compute Fisher matrix and invert it
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> fisher = fk.fisher()
   >>> cov_theta = np.linalg.inv(fisher)
   >>> print(cov_theta.shape)
   (2, 2)
   >>> print(np.all(np.isfinite(cov_theta)))
   True


Choosing a derivative backend
-----------------------------

The Fisher matrix is constructed from derivatives of the model with respect to
the parameters. You can control how these derivatives are computed by passing
``method`` and backend-specific options.

All keyword arguments are forwarded to
:meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

.. doctest:: fisher_backend_control

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> np.set_printoptions(precision=8, suppress=True)
   >>> # Define the model
   >>> def model(theta):
   ...     a, b = theta
   ...     return np.array([a, b, a + 2.0 * b], dtype=float)
   >>> theta0 = np.array([1.0, 2.0])
   >>> cov = np.eye(3)
   >>> # Use a finite-difference derivative backend
   >>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
   >>> fisher = fk.fisher(
   ...     method="finite",
   ...     n_workers=2,
   ...     stepsize=1e-2,
   ...     num_points=5,
   ...     extrapolation="ridders",
   ...     levels=4,
   ... )
   >>> print(fisher)
   [[2. 2.]
    [2. 5.]]


Parallel execution
------------------

Fisher matrix elements can be computed in parallel using ``n_workers``.

This parallelizes derivative evaluations across parameters and model outputs.

.. doctest:: fisher_parallel

   >>> import numpy as np
   >>> from derivkit.forecast_kit import ForecastKit
   >>> # Model definition
   >>> def model(theta):
   ...     return np.array([theta[0], theta[1], theta[0] + 2.0 * theta[1]])
   >>> # Enable parallel execution
   >>> fk = ForecastKit(function=model, theta0=np.array([1.0, 2.0]), cov=np.eye(3))
   >>> fisher = fk.fisher(n_workers=4)
   >>> print(fisher.shape)
   (2, 2)


Notes
-----

- The Fisher matrix assumes a Gaussian likelihood with fixed covariance.
- Derivatives are evaluated at ``theta0``.
- The choice of derivative backend can affect numerical accuracy and cost.
- For likelihoods with parameter-dependent covariance, see the generalized
  Fisher matrix utilities.
