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
        function: Callable[[ArrayLike], float | NDArray[np.floating]],
        theta0: ArrayLike,
        cov: ArrayLike,
        *,
        forecast_order: int = 1,
        method: str | None = None,
        n_workers: int = 1,
        single_forecast_order: bool = True,
        **dk_kwargs: Any,
) -> Union[
        NDArray[np.float64],
        tuple[NDArray[np.float64]],
        dict[int, tuple[NDArray[np.float64],...]],
    ]:
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
        single_forecast_order: If set to ``True``, the function will return only
            the requested order. If set to ``False``, the function will return
            the tensors up to the requested order.
        **dk_kwargs: Additional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        If ``single_order == True`` the function returns a tuple containing
        the tensors of the requested order.

        If ``single_order == False`` the function returns a dict with entries
        of the form ``order: (tensor_1,...,tensor_n)``, where ``order`` is an
        ``int` that determines the order in the DALI approximation and the
        ``tensor_n`` are the tensors corresponding to the order ``order`` in the
        DALI approximation.

        The number of axes of the tensors in the tuple at order ``order``
        ranges from ``order+1`` to ``2*order`` in order of increasing number.
        All axes have as length ``len(theta0)``.

    Raises:
        ValueError: If ``forecast_order`` is not in :data:`SUPPORTED_FORECAST_ORDER`.

    Warns:
        RuntimeWarning: If ``cov`` is not symmetric (proceeds as-is, no symmetrization),
            is ill-conditioned (large condition number), or inversion
            falls back to the pseudoinverse.
    """
    if forecast_order not in SUPPORTED_FORECAST_ORDERS:
        raise ValueError(
            "A forecast_order was requested with a higher value "
            "than currently supported."
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

    forecast_tensors = {}
    derivatives = {}
    contractions = {
        1: {1: "ai,ij,bj->ab"},
        2: {1: "abi,ij,cj->abc",
            2: "abi,ij,cdj->abcd"},
        3: {1: "iabc,ij,dj->abcd",
            2: "iabc,ij,dej->abcde",
            3: "iabc,ij,jdef->abcdef"},
    }
    for order1 in range(1, 1+forecast_order):
        derivatives[order1] = _get_derivatives(
            function,
            theta0_arr,
            cov_arr,
            order=order1,
            n_workers=n_workers,
            method=method,
            **dk_kwargs,
        )
        tensors_at_order = []
        for order2 in contractions[order1].keys():
            if order2 > order1:
                raise ValueError(
                    f"Requested a derivative of order {order2} "
                    f"in a DALI forecast of order {order1}"
                )
            tensors_at_order.append(np.einsum(
                contractions[order1][order2],
                derivatives[order1],
                invcov,
                derivatives[order2]
            ))
        forecast_tensors[order1] = tuple(tensors_at_order)

    if single_forecast_order:
        forecast_tensors = forecast_tensors[forecast_order]
        # The expected behaviour of the function is to return
        # the Fisher matrix directly (not as a tuple). The Fisher
        # matrix is the only DALI singlet, so if forecast_tensors
        # at this point has length one it is guaranteed to contain
        # the Fisher matrix.
        if len(forecast_tensors) == 1:
            forecast_tensors = forecast_tensors[0]

    return forecast_tensors


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
        shape is ``(n_parameters, n_observables)`` (first-order derivatives).
        For ``order == 2``, the shape is
        ``(n_parameters, n_parameters, n_observables)`` (second-order derivatives).
        For ``order == 3``, the shape is
        ``(n_parameters, n_parameters, n_parameters, n_observables)`` (third-order derivatives).

    Raises:
        ValueError: An error occurred if a derivative was requested of
            higher order than 3.
        RuntimeError: An error occurred if a ValueError was not raised
            after calling the function.
    """
    if order not in SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            "Requested derivative order is higher than what is "
            "currently supported."
        )

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

    elif order == 3:
        hh_raw = np.asarray(
            ckit.hyper_hessian(
                method=method,
                n_workers=n_workers,  # allow outer parallelism across params
                **dk_kwargs,
            ),
            dtype=float,
        )
        if hh_raw.shape == (n_observables, n_parameters, n_parameters, n_parameters):
            return hh_raw
        elif hh_raw.shape == (n_parameters, n_parameters, n_parameters, n_observables):
            return np.moveaxis(hh_raw, [0,1,2], [1,2,3])
        else:
            raise ValueError(
                f"build_hessian_tensor returned unexpected shape {hh_raw.shape}; "
                f"expected ({n_observables},{n_parameters},{n_parameters},{n_parameters}) or "
                f"({n_parameters},{n_parameters},{n_parameters},{n_observables})."
            )

    else:
        raise ValueError(f"Unsupported value of {order}.")
