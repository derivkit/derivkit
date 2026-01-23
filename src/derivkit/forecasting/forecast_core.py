"""Core utilities for likelihoods-based forecasts.

This module provides functional helpers to

- compute first- and second-order derivatives of a model with respect to
  its parameters, and
- build Fisher and doublet-DALI forecast tensors from those derivatives
  and a covariance matrix.

These functions are the low-level building blocks used by higher-level
forecasting interfaces in DerivKit. For details on the DALI expansion,
see e.g. https://doi.org/10.1103/PhysRevD.107.103506.
"""

from typing import Any, Callable, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.concurrency import normalize_workers
from derivkit.utils.linalg import invert_covariance
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "get_forecast_tensors",
]


def get_forecast_tensors(
        function: Callable[[ArrayLike], float | NDArray[np.floating]],
        theta0: ArrayLike,
        cov: ArrayLike,
        *,
        forecast_order: int = 1,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Returns a set of tensors according to the requested order of the forecast.

    Args:
        function: The scalar or vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return either a scalar or a
            :class:`np.ndarray` of observable values.
        theta0: The points at which the
            derivative is evaluated. A 1D array or list of parameter values
            matching the expected input of the function.
        cov: The covariance matrix of the observables. Should be a square
            matrix with shape ``(n_observables, n_observables)``, where ``n_observables``
            is the number of observables returned by the function.
        forecast_order: The requested order D of the forecast:

                - D = 1 returns a Fisher matrix.
                - D = 2 returns the 3D and 4D tensors required for the
                  doublet-DALI approximation.
                - D = 3 would be the triplet-DALI approximation.

            Currently only D = 1, 2 are supported.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial). Inner batch evaluation is kept serial to
            avoid nested pools.
        **dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        If ``D = 1``: Fisher matrix of shape ``(P, P)``.
        If ``D = 2``: tuple ``(G, H)`` with shapes ``(P, P, P)`` and ``(P, P, P, P)``.

    Raises:
        ValueError: If `forecast_order` is not 1 or 2.

    Warns:
        RuntimeWarning: If `cov` is not symmetric (proceeds as-is, no symmetrization),
            is ill-conditioned (large condition number), or inversion
            falls back to the pseudoinverse.
    """
    if forecast_order not in (1, 2):
        raise ValueError(
            "Only Fisher (order 1) and doublet-DALI (order 2) forecasts are currently supported."
        )

    theta0_arr = np.atleast_1d(theta0)
    cov_arr = validate_covariance_matrix_shape(cov)
    n_observables = cov_arr.shape[0]

    y0 = np.atleast_1d(function(theta0_arr))
    if y0.shape[0] != n_observables:
        raise ValueError(
            f"Expected {n_observables} observables from model (from cov {cov.shape}), "
            f"but got {y0.shape[0]} (output shape {y0.shape})."
        )

    invcov = invert_covariance(cov_arr, warn_prefix="get_forecast_tensors")

    deriv_order1 = _get_derivatives(
        function,
        theta0_arr,
        cov_arr,
        order=1,
        n_workers=n_workers,
        method=method,
        **dk_kwargs,
    )

    if forecast_order == 1:
        return np.einsum("ai,ij,bj->ab",
                         deriv_order1,
                         invcov,
                         deriv_order1)  # Fisher

    deriv_order2 = _get_derivatives(
        function,
        theta0_arr,
        cov_arr,
        order=2,
        n_workers=n_workers,
        method=method,
        **dk_kwargs,
    )
    # G_abc = Σ_ij d2[a,b,i] invcov[i,j] d1[c,j]
    g_tensor = np.einsum("abi,ij,cj->abc", deriv_order2, invcov, deriv_order1)
    # H_abcd = Σ_ij d2[a,b,i] invcov[i,j] d2[c,d,j]
    h_tensor = np.einsum("abi,ij,cdj->abcd", deriv_order2, invcov, deriv_order2)
    return g_tensor, h_tensor


def _get_derivatives(
        function: Callable[[ArrayLike], float | NDArray[np.floating]],
        theta0: ArrayLike,
        cov: ArrayLike,
        *,
         order: int,
         method: str | None = None,
         n_workers: int = 1,
         **dk_kwargs: Any,
         ) -> NDArray[np.float64]:
    """Returns derivatives of the observables of the requested order.

    Args:
        function: The scalar or vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return either a scalar or a
            :class:`np.ndarray` of observable values.
        theta0: The points at which the
            derivative is evaluated. A 1D array or list of parameter values
            matching the expected input of the function.
        cov: The covariance matrix of the observables. Should be a square
            matrix with shape ``(n_observables, n_observables)``, where ``n_observables``
            is the number of observables returned by the function.
        order: The requested order of the derivatives:

            - A value of ``1`` returns first-order derivatives.
            - A value of ``2`` returns second-order derivatives.

            Currently supported values are ``1`` and ``2``.

        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``). If ``None``,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers for per-parameter parallelization
         (threads). Default ``1`` (serial).
        **dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Array of derivative values. For ``order == 1``, the
        shape is ``(n_parameters, n_observables)`` (first-order derivatives).
        For ``order == 2``, the shape is
        ``(n_parameters, n_parameters, n_observables)`` (second-order derivatives).

    Raises:
        ValueError: An error occurred if a derivative was requested of
            higher order than 2.
        RuntimeError: An error occurred if a ValueError was not raised
            after calling the function.
    """
    if order not in (1, 2):
        raise ValueError("Only first- and second-order derivatives are currently supported.")

    theta0_arr = np.atleast_1d(theta0)
    cov_arr = validate_covariance_matrix_shape(cov)

    n_parameters = theta0_arr.shape[0]
    n_observables = cov_arr.shape[0]

    n_workers = normalize_workers(n_workers)
    ckit = CalculusKit(function, theta0_arr)

    if order == 1:
        j_raw = np.asarray(
            ckit.jacobian(
                method=method,
                n_workers=n_workers,  # allow outer parallelism across params
                **dk_kwargs,
            ),
            dtype=float,
        )
        # Accept (N, P) or (P, N); return (P, N)
        if j_raw.shape == (n_observables, n_parameters):
            return j_raw.T
        elif j_raw.shape == (n_parameters, n_observables):
            return j_raw
        else:
            raise ValueError(
                f"build_jacobian returned unexpected shape {j_raw.shape}; "
                f"expected ({n_observables},{n_parameters}) or "
                f"({n_parameters},{n_observables})."
            )

    elif order == 2:
        # Build Hessian tensor once (shape expected (n_observables, n_parameters, n_parameters)),
        # then return as (n_parameters, n_parameters, n_observables) for downstream einsum.
        h_raw = np.asarray(
            ckit.hessian(
                method=method,
                n_workers=n_workers,  # allow outer parallelism across params
                **dk_kwargs,
            ),
            dtype=float,
        )
        if h_raw.shape == (n_observables, n_parameters, n_parameters):
            return np.moveaxis(h_raw,[1,2],[0,1])
        elif h_raw.shape == (n_parameters, n_parameters, n_observables):
            return h_raw
        else:
            raise ValueError(
                f"build_hessian_tensor returned unexpected shape {h_raw.shape}; "
                f"expected ({n_observables},{n_parameters},{n_parameters}) or "
                f"({n_parameters},{n_parameters},{n_observables})."
            )
    else:
        raise ValueError(f"Unsupported value of {order}.")
