r"""Gaussian Fisher matrix utilities for models with uncertainty in both inputs and outputs.

This module implements the X–Y Gaussian Fisher formalism, where both the measured
inputs and outputs are noisy and may be correlated. The key idea is to account for
uncertainty in the inputs by propagating it into an effective covariance for the
outputs through a local linearization of the model. This allows standard Gaussian
Fisher matrix techniques to be applied without explicitly marginalizing over the
latent input variables.

Model and covariance structure
------------------------------

The model provides a mean prediction ``mu_xy(x, theta)`` for the observed output
``y`` as a function of inputs ``x`` and parameters ``theta``. Measurement errors
on ``x`` and ``y`` are described by a joint Gaussian covariance

.. math::

   C =
   \begin{pmatrix}
       C_{xx} & C_{xy} \\
       C_{xy}^{\mathsf{T}} & C_{yy}
   \end{pmatrix}.

Linearizing the model mean in the inputs around the measured values ``x_obs``,

.. math::

   \mu_{xy}(x, \theta) \approx \mu_{xy}(x_{\mathrm{obs}}, \theta) + T (x - x_{\mathrm{obs}}),

with

.. math::

   T = \left.\frac{\partial \mu_{xy}}{\partial x}\right|_{(x_{\mathrm{obs}}, \theta)},

yields an effective output covariance

.. math::

   R = C_{yy}
       - C_{xy}^{\mathsf{T}} T^{\mathsf{T}}
       - T C_{xy}
       + T C_{xx} T^{\mathsf{T}}.

This effective covariance replaces ``C_{yy}`` in the Gaussian likelihood and Fisher
matrix. The covariance blocks ``Cxx``, ``Cxy``, and ``Cyy`` are treated as fixed;
parameter dependence enters only through the local sensitivity matrix ``T``.

This formalism follows the generalized Fisher matrix treatment of
Heavens et al. (2014), https://arxiv.org/abs/1404.2854.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.forecasting.fisher_gaussian import build_gaussian_fisher_matrix
from derivkit.utils.validate import validate_covariance_matrix_shape

MuXY = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64] | float]

__all__ = [
    "mu_xy_given_theta",
    "mu_xy_given_x0",
    "build_mu_theta_from_mu_xy",
    "build_t_matrix",
    "build_effective_covariance_r",
    "effective_covariance_r_theta",
    "build_xy_gaussian_fisher_matrix",
]


def _as_1d_mu_output(y: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Converts a model output into a 1D array.

    This helper ensures the model mean can be treated as a single data vector.
    Scalars become length-1 arrays. If the model returns a vector or higher-rank
    tensor, it is flattened into 1D in row-major (C) order.

    Args:
        y: Model output. Can be a scalar, a vector, or a higher-rank array.

    Returns:
        The model output represented as a single 1D data vector.
    """
    arr = np.asarray(y, dtype=np.float64)

    if arr.ndim == 0:
        return arr[None]
    if arr.ndim == 1:
        return arr
    return arr.ravel(order="C")


def mu_xy_given_theta(
    x: NDArray[np.float64],
    *,
    theta: NDArray[np.float64],
    mu_xy: MuXY,
) -> NDArray[np.float64]:
    """Evaluates the model predicted mean as a function of ``x`` at fixed parameters.

    The input ``mu_xy`` is a callable that returns the model's mean prediction for the
    observed quantity ``y`` given input values ``x`` and parameters ``theta``. This
    wrapper holds ``theta`` fixed, so the resulting function depends only on ``x``.
    This form is used when varying the inputs while keeping the parameter point
    unchanged, such as when computing sensitivities with respect to ``x``.

    Args:
        x: Input values at which to evaluate the model.
        theta: Parameter values to hold fixed.
        mu_xy: Function that predicts the mean of ``y`` given ``x`` and ``theta``.

    Returns:
        The models mean prediction for ``y`` at ``(x, theta)``, returned as a single
        1D data vector.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    return _as_1d_mu_output(mu_xy(x, theta))


def mu_xy_given_x0(
    theta: NDArray[np.float64],
    *,
    x0: NDArray[np.float64],
    mu_xy: MuXY,
) -> NDArray[np.float64]:
    """Evaluates the model predicted mean as a function of ``theta`` at fixed inputs.

    The input ``mu_xy`` is a callable that returns the model's mean prediction for the
    observed quantity ``y`` given input values ``x`` and parameters ``theta``. This
    wrapper holds the inputs fixed at a chosen reference point ``x0``, so the resulting
    function depends only on ``theta``. This form is used when varying the parameters
    while treating the inputs as fixed.

    Args:
        theta: Parameter values at which to evaluate the model.
        x0: Input values to hold fixed.
        mu_xy: Function that predicts the mean of ``y`` given ``x`` and ``theta``.

    Returns:
        The models mean prediction for ``y`` at ``(x0, theta)``, returned as a single
        1D data vector.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    return _as_1d_mu_output(mu_xy(x0, theta))


def build_mu_theta_from_mu_xy(
    mu_xy: MuXY,
    *,
    x0: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Constructs a mean function that depends only on the model parameters.

    The input ``mu_xy`` predicts the model mean for the observed quantity given
    input values and parameters. This helper fixes the input values at a chosen
    reference point ``x0`` and returns a callable that depends only on the
    parameters. The resulting function represents the model mean evaluated at
    fixed inputs and is used when building parameter derivatives or Fisher
    matrices.

    Args:
        mu_xy: Function that predicts the mean of the observed quantity given
            input values and parameters.
        x0: Input values at which the model mean is evaluated.

    Returns:
        A callable that evaluates the model mean as a function of the parameters
        with the inputs held fixed.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    return partial(mu_xy_given_x0, x0=x0, mu_xy=mu_xy)


def build_t_matrix(
    mu_xy: MuXY,
    *,
    x0: NDArray[np.float64],
    theta: NDArray[np.float64],
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Computes the sensitivity of the model mean to changes in the inputs.

    The returned matrix describes how the model mean for the observed quantity changes
    when the input values are perturbed, evaluated at a reference input point ``x0``
    and parameter point ``theta``. In the X–Y Gaussian formulation, this sensitivity
    is used to propagate input uncertainty into an effective covariance for the
    observed quantity.

    Args:
        mu_xy: Function that predicts the mean of the observed quantity given input
            values and parameters.
        x0: Reference input values at which the sensitivity is evaluated.
        theta: Parameter values at which the sensitivity is evaluated.
        method: Optional derivative method name passed to the derivative engine.
        n_workers: Number of workers used for derivative evaluations.
        **dk_kwargs: Additional keyword arguments forwarded to the derivative engine.

    Returns:
        A matrix of sensitivities evaluated at ``(x0, theta)``, with one row per output
        component and one column per input component.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))

    mu_of_x = partial(mu_xy_given_theta, theta=theta, mu_xy=mu_xy)

    ckit = CalculusKit(mu_of_x, x0)
    jac = np.asarray(
        ckit.jacobian(method=method, n_workers=n_workers, **dk_kwargs),
        dtype=np.float64,
    )

    if jac.ndim == 1:
        jac = jac[None, :]
    return jac


def build_effective_covariance_r(
    *,
    cxx: NDArray[np.float64],
    cxy: NDArray[np.float64],
    cyy: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes an effective output covariance that includes input uncertainty.

    The X–Y Gaussian formulation allows both the inputs and outputs to be noisy, with
    possible correlations between input and output errors. This function combines the
    input covariance, output covariance, and cross-covariance with a local sensitivity
    matrix ``t`` (describing how the model mean changes with the inputs) to produce an
    effective covariance for the outputs. The result is the covariance used in the
    Gaussian likelihood and Fisher matrix after input uncertainty has been propagated
    to the output space.

    Args:
        cxx: Covariance matrix of the input measurements.
        cxy: Cross-covariance between input and output measurement errors.
        cyy: Covariance matrix of the output measurements.
        t: Sensitivity matrix of the model mean with respect to the inputs, evaluated
            at a chosen reference point.

    Returns:
        The effective covariance matrix for the output measurements.
    """
    cxx = np.asarray(cxx, dtype=np.float64)
    cxy = np.asarray(cxy, dtype=np.float64)
    cyy = np.asarray(cyy, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    nx = cxx.shape[0]
    ny = cyy.shape[0]

    validate_covariance_matrix_shape(cxx)
    validate_covariance_matrix_shape(cyy)

    if cxy.shape != (nx, ny):
        raise ValueError(f"cxy must have shape ({nx}, {ny}); got {cxy.shape}.")
    if t.shape != (ny, nx):
        raise ValueError(f"t must have shape ({ny}, {nx}); got {t.shape}.")

    return cyy - (cxy.T @ t.T) - (t @ cxy) + (t @ cxx @ t.T)


def effective_covariance_r_theta(
    theta: NDArray[np.float64],
    *,
    mu_xy: MuXY,
    x0: NDArray[np.float64],
    cxx: NDArray[np.float64],
    cxy: NDArray[np.float64],
    cyy: NDArray[np.float64],
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    """Evaluates the effective output covariance at a given parameter point.

    The block covariances for the input and output measurements are treated as fixed.
    The effective output covariance depends on the parameters through the local
    sensitivity of the model mean to the inputs, evaluated at the reference inputs
    ``x0``. This function recomputes that sensitivity at the supplied ``theta`` and
    returns the corresponding effective covariance.

    Args:
        theta: Parameter values at which the effective covariance is evaluated.
        mu_xy: Function that predicts the mean of the observed quantity given input
            values and parameters.
        x0: Reference input values used for the local sensitivity evaluation.
        cxx: Covariance matrix of the input measurements.
        cxy: Cross-covariance between input and output measurement errors.
        cyy: Covariance matrix of the output measurements.
        method: Optional derivative method name passed to the derivative engine.
        n_workers: Number of workers used for derivative evaluations.
        dk_kwargs: Additional keyword arguments forwarded to the derivative engine.

    Returns:
        The effective covariance matrix for the output measurements at ``theta``.
    """
    t_matrix = build_t_matrix(
        mu_xy,
        x0=x0,
        theta=theta,
        method=method,
        n_workers=n_workers,
        **dk_kwargs,
    )
    return build_effective_covariance_r(cxx=cxx, cxy=cxy, cyy=cyy, t=t_matrix)


def build_xy_gaussian_fisher_matrix(
    *,
    theta0: NDArray[np.float64],
    x0: NDArray[np.float64],
    mu_xy: MuXY,
    cxx: NDArray[np.float64],
    cxy: NDArray[np.float64],
    cyy: NDArray[np.float64],
    term: str = "both",
    method: str | None = None,
    n_workers: int = 1,
    rcond: float = 1e-12,
    symmetrize_dcov: bool = True,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Computes a Gaussian Fisher matrix when both inputs and outputs are noisy.

    This function supports the X–Y Gaussian case, where measurement uncertainty is
    present in the inputs and in the outputs, and the two may be correlated. Input
    uncertainty is incorporated by forming an effective covariance for the output
    measurements using a local sensitivity of the model mean to the inputs evaluated
    at the reference inputs ``x0``. The Fisher matrix is then constructed at the
    parameter point ``theta0`` using the model mean evaluated at ``x0`` and the
    effective output covariance.

    Args:
        theta0: Parameter values at which the Fisher matrix is evaluated.
        x0: Reference input values used for the local sensitivity evaluation.
        mu_xy: Function that predicts the mean of the observed quantity given input
            values and parameters.
        cxx: Covariance matrix of the input measurements.
        cxy: Cross-covariance between input and output measurement errors.
        cyy: Covariance matrix of the output measurements.
        term: Which contributions to include in the Fisher matrix. Supported values
            are ``"mean"``, ``"cov"``, and ``"both"``.
        method: Optional derivative method name passed to the derivative engine.
        n_workers: Number of workers used for derivative evaluations.
        rcond: Cutoff used when solving linear systems involving the covariance.
        symmetrize_dcov: Whether to symmetrize numerical covariance derivatives.
        **dk_kwargs: Additional keyword arguments forwarded to the derivative engine.

    Returns:
        The Fisher information matrix evaluated at ``theta0``.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=np.float64))
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))

    mu_theta = build_mu_theta_from_mu_xy(mu_xy, x0=x0)

    r_fn = partial(
        effective_covariance_r_theta,
        mu_xy=mu_xy,
        x0=x0,
        cxx=np.asarray(cxx, dtype=np.float64),
        cxy=np.asarray(cxy, dtype=np.float64),
        cyy=np.asarray(cyy, dtype=np.float64),
        method=method,
        n_workers=n_workers,
        dk_kwargs=dict(dk_kwargs),
    )

    r0 = np.asarray(r_fn(theta0), dtype=np.float64)

    return build_gaussian_fisher_matrix(
        theta0=theta0,
        cov=(r0, r_fn),
        function=None if term == "cov" else mu_theta,
        term=term,
        method=method,
        n_workers=n_workers,
        rcond=rcond,
        symmetrize_dcov=symmetrize_dcov,
        **dk_kwargs,
    )
