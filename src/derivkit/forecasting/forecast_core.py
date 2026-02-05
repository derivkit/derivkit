"""Core utilities for likelihoods-based forecasts.

This module provides functional helpers to

- compute first-, second-, and third-order derivatives of a model with
  respect to its parameters, and
- build Fisher, doublet-DALI, and triplet-DALI forecast tensors from those
  derivatives and a covariance matrix.

These functions are the low-level building blocks used by higher-level
forecasting interfaces in DerivKit. For details on the DALI expansion,
see e.g. https://doi.org/10.1103/PhysRevD.107.103506.
"""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.concurrency import normalize_workers
from derivkit.utils.linalg import invert_covariance
from derivkit.utils.types import ArrayLike1D, ArrayLike2D
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "SUPPORTED_FORECAST_ORDERS",
    "get_forecast_tensors",
]


#: The supported orders of the DALI expansion.
#:
#: A value of 1 corresponds to the Fisher matrix.
#: A value of 2 corresponds to the DALI doublet.
#: A value of 3 corresponds to the DALI triplet.
SUPPORTED_FORECAST_ORDERS = (1, 2, 3)

SUPPORTED_DERIVATIVE_ORDERS = (1, 2, 3)


def get_forecast_tensors(
    function: Callable[[ArrayLike1D], float | NDArray[np.floating]],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    *,
    forecast_order: int = 1,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> dict[int, tuple[NDArray[np.float64], ...]]:
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
        forecast_order: The requested order of the forecast.
            Currently supported values and their meaning are given in
            :data:`SUPPORTED_FORECAST_ORDERS`.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial). Inner batch evaluation is kept serial to
            avoid nested pools.
        **dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A dict mapping ``order -> tensors`` for all ``order = 1..forecast_order``.

        The tensors are grouped by the *forecast order at which they first appear*:

        - order 1: ``(F,)``
        - order 2: ``(D_{(2,1)}, D_{(2,2)})``
        - order 3: ``(T_{(3,1)}, T_{(3,2)}, T_{(3,3)})``

        Here ``D_{(k,l)}`` and ``T_{(k,l)}`` denote tensors obtained by contracting
        the ``k``-th order derivative with the ``l``-th order derivative via the
        inverse covariance.

        Each tensor axis has length ``p = len(theta0)``. Shapes are:

        - ``F``: ``(p, p)``
        - ``D_{(2,1)}``: ``(p, p, p)``
        - ``D_{(2,2)}``: ``(p, p, p, p)``
        - ``T_{(3,1)}``: ``(p, p, p, p)``
        - ``T_{(3,2)}``: ``(p, p, p, p, p)``
        - ``T_{(3,3)}``: ``(p, p, p, p, p, p)``

    Raises:
        ValueError: If ``forecast_order`` is not in :data:`SUPPORTED_FORECAST_ORDERS`.

    Warns:
        RuntimeWarning: If ``cov`` is not symmetric (proceeds as-is, no symmetrization),
            is ill-conditioned (large condition number), or inversion
            falls back to the pseudoinverse.
    """
    try:
        forecast_order = int(forecast_order)
    except Exception as e:
        raise TypeError(f"forecast_order must be an int;"
                        f" got {type(forecast_order)}.") from e

    if forecast_order not in SUPPORTED_FORECAST_ORDERS:
        raise ValueError(
            f"forecast_order={forecast_order} is not supported. "
            f"Supported values: {SUPPORTED_FORECAST_ORDERS}."
        )

    theta0_arr = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError("theta0 must be non-empty 1D.")

    cov_arr = validate_covariance_matrix_shape(cov)
    n_observables = cov_arr.shape[0]

    y0 = np.asarray(function(theta0_arr), dtype=float)
    y0_flat = y0.reshape(-1)

    if y0_flat.size != n_observables:
        raise ValueError(
            f"Expected {n_observables} observables from model "
            f"(from cov {cov_arr.shape}), "
            f"but got {y0_flat.size} (output shape {y0.shape})."
        )

    invcov = invert_covariance(cov_arr, warn_prefix="get_forecast_tensors")

    forecast_tensors: dict[int, tuple[NDArray[np.float64], ...]] = {}
    derivatives: dict[int, NDArray[np.float64]] = {}

    contractions = {
        1: {1: "ia,ij,jb->ab"},
        2: {1: "iab,ij,jc->abc",
            2: "iab,ij,jcd->abcd"},
        3: {1: "iabc,ij,jd->abcd",
            2: "iabc,ij,jde->abcde",
            3: "iabc,ij,jdef->abcdef"},
    }

    for order1 in range(1, 1 + forecast_order):
        derivatives[order1] = _get_derivatives(
            function,
            theta0_arr,
            cov_arr,
            order=order1,
            n_workers=n_workers,
            method=method,
            **dk_kwargs,
        )

        tensors_at_order: list[NDArray[np.float64]] = []
        for order2 in contractions[order1]:
            tensors_at_order.append(
                np.einsum(
                    contractions[order1][order2],
                    derivatives[order1],
                    invcov,
                    derivatives[order2],
                ).astype(np.float64, copy=False)
            )

        forecast_tensors[order1] = tuple(tensors_at_order)

    expected_keys = set(range(1, forecast_order + 1))
    if set(forecast_tensors.keys()) != expected_keys:
        raise RuntimeError(
            f"internal error: forecast_tensors keys {sorted(forecast_tensors.keys())} "
            f"!= expected {sorted(expected_keys)}."
        )

    return forecast_tensors


def _get_derivatives(
    function: Callable[[ArrayLike1D], float | NDArray[np.floating]],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
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
        order: The requested order of the derivatives. The value determines
            the order of the derivative that is returned. Currently supported
            values are given in :data:`SUPPORTED_DERIVATIVE_ORDERS`.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``). If ``None``,
            the DerivativeKit default ("adaptive") is used.
        n_workers: Number of workers for per-parameter parallelization
         (threads). Default ``1`` (serial).
        **dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Array of derivative values. For ``order == 1``, the
        shape is ``(n_observables, n_parameters)`` (first-order derivatives).
        For ``order == 2``, the shape is
        ``(n_observables, n_parameters, n_parameters)`` (second-order derivatives).
        For ``order == 3``, the shape is
        ``(n_observables, n_parameters, n_parameters, n_parameters)`` (third-order derivatives).

    Raises:
        ValueError: An error occurred if a derivative was requested of
            higher order than 3.
        RuntimeError: An error occurred if a ValueError was not raised
            after calling the function.
    """
    if order not in SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            f"Requested derivative order={order} is not supported. "
            f"Supported values: {SUPPORTED_DERIVATIVE_ORDERS}."
        )

    theta0_arr = np.atleast_1d(theta0)
    cov_arr = np.asarray(cov, dtype=float)

    n_parameters = theta0_arr.shape[0]
    n_observables = cov_arr.shape[0]

    n_workers = normalize_workers(n_workers)

    def _vectorize_model_output(theta: ArrayLike1D) -> NDArray[np.float64]:
        """Returns model output as a 1D float64 vector."""
        y = np.asarray(function(theta), dtype=np.float64)
        if y.ndim > 1:
            raise TypeError(
                "model must return a scalar or 1D vector of observables; "
                f"got shape {y.shape}."
            )
        return np.atleast_1d(y)

    ckit = CalculusKit(_vectorize_model_output, theta0_arr)

    if order == 1:
        j_raw = np.asarray(
            ckit.jacobian(
                method=method,
                n_workers=n_workers,  # allow outer parallelism across params
                **dk_kwargs,
            ),
            dtype=float,
        )
        if j_raw.shape == (n_observables, n_parameters):
            return j_raw
        else:
            raise ValueError(
                f"jacobian returned unexpected shape {j_raw.shape}; "
                f"expected ({n_observables},{n_parameters})."
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
            return h_raw
        else:
            raise ValueError(
                f"hessian returned unexpected shape {h_raw.shape}; "
                f"expected ({n_observables},{n_parameters},{n_parameters})."
            )


    elif order == 3:
        hh_raw = np.asarray(
            ckit.hyper_hessian(
                method=method,
                n_workers=n_workers,
                **dk_kwargs,
            ),
            dtype=float,
        )
        if hh_raw.shape == (n_observables, n_parameters, n_parameters, n_parameters):
            return hh_raw
        else:
            raise ValueError(
                f"hyper_hessian returned unexpected shape {hh_raw.shape}; "
                f"expected ({n_observables},{n_parameters},{n_parameters},{n_parameters})."
            )
 

    else:
        raise ValueError(f"Unsupported value of {order}.")
