"""Generalized Fisher matrix construction for parameter-dependent mean and covariance."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.forecasting.forecast_core import get_forecast_tensors
from derivkit.utils.concurrency import normalize_workers
from derivkit.utils.linalg import solve_or_pinv
from derivkit.utils.validate import (
    flatten_matrix_c_order,
    resolve_covariance_input,
    validate_covariance_matrix_shape,
)

__all__ = [
    "build_gaussian_fisher_matrix"
]


def build_gaussian_fisher_matrix(
    theta0: NDArray[np.float64],
    cov: NDArray[np.float64]
        | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    function: Callable[[NDArray[np.float64]], float | NDArray[np.float64]],
    *,
    method: str | None = None,
    n_workers: int = 1,
    rcond: float = 1e-12,
    symmetrize_dcov: bool = True,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Computes the Gaussian Fisher matrix.

    This implements the standard Fisher matrix for a Gaussian likelihood with
    parameter-dependent mean and covariance (see e.g. Eq. (2) of arXiv:1404.2854).

    For Gaussian-distributed data `d` with mean `mu(theta)` and covariance `C(theta)`,
    the generalized Fisher matrix evaluated at `theta0` is::

        F_ij = mu_i^T C^{-1} mu_j + 0.5 * Tr[C^{-1} C_i C^{-1} C_j].

    Args:
        function: Callable returning the model mean ``mu(theta)`` as a scalar (only if
            ``n_obs == 1``) or 1D array of observables with shape ``(n_obs,)``.
        cov: Covariance matrix. Provide either a fixed covariance array or
            a callable covariance function. Supported forms are:

            - ``cov=C0``: fixed covariance matrix ``C(theta_0)`` with shape
              ``(n_obs, n_obs)``. Here ``n_obs`` is the number of observables.
              In this case the covariance-derivative Fisher term will not be computed.
            - ``cov=cov_fn``: callable ``cov_fn(theta)`` returning the covariance
              matrix ``C(theta)`` evaluated at the parameter vector ``theta``,
              with shape ``(n_obs, n_obs)``.

            The callable form is evaluated at ``theta0`` to determine ``n_obs`` and (unless
            ``C0`` is provided) to define ``C0 = C(theta0)``.
        theta0: Fiducial parameter vector where the Fisher matrix is evaluated.
        method: Derivative method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit` default is used.
        n_workers: Number of workers for per-parameter parallelisation. Default is ``1`` (serial).
        rcond: Regularization cutoff for pseudoinverse fallback.
        symmetrize_dcov: If ``True``, symmetrize each covariance derivative via
            ``0.5 * (C_i + C_i.T)``. Default is ``True``.
        **dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Fisher matrix with shape ``(p, p)`` where ``p`` is the number of parameters.

    Raises:
        ValueError: If ``function(theta0)`` does not match the implied observable dimension.
    """
    n_workers = normalize_workers(n_workers)
    theta0 = np.atleast_1d(theta0).astype(np.float64)
    n_parameters = int(theta0.size)

    cov0, cov_fn = resolve_covariance_input(
        cov, theta0=theta0, validate=validate_covariance_matrix_shape
    )
    n_observables = int(cov0.shape[0])

    # Validate that function(theta0) matches the
    # observable dimension implied by cov0.
    _mu0 = np.asarray(function(theta0), dtype=np.float64)
    if _mu0.ndim == 0:
        # Scalar mean is only valid for a single observable.
        if n_observables != 1:
            raise ValueError(
                "function(theta0) returned a scalar, "
                "but cov implies n_observables={n_observables}. "
                "Return a 1D mean vector with length n_observables."
            )
    elif _mu0.ndim != 1 or _mu0.shape[0] != n_observables:
        raise ValueError(
            f"function(theta0) must return shape ({n_observables},); "
            "got {_mu0.shape}."
        )

    # Term with derivatives of covariance matrices:
    # (1/2) Tr[C^{-1} C_{,i} C^{-1} C_{,j}]
    fisher_cov = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    if cov_fn is not None:

        def cov_flat_function(th: NDArray[np.float64]) -> NDArray[np.float64]:
            """Flattened covariance function for derivative computation."""
            return flatten_matrix_c_order(cov_fn,
                                          th,
                                          n_observables=n_observables)

        cov_ckit = CalculusKit(cov_flat_function, theta0)

        dcov_flat = np.asarray(
            cov_ckit.jacobian(method=method, n_workers=n_workers, **dk_kwargs),
            dtype=np.float64,
        )

        expected_shape = (n_observables * n_observables, n_parameters)
        if dcov_flat.shape != expected_shape:
            raise ValueError(
                f"dcov_flat must have shape {expected_shape}; "
                "got {dcov_flat.shape}."
            )

        dcov = dcov_flat.T.reshape(n_parameters,
                                   n_observables,
                                   n_observables,
                                   order="C")
        if symmetrize_dcov:
            dcov = 0.5 * (dcov + np.swapaxes(dcov, -1, -2))

        covinv_dcov = np.empty_like(dcov)  # (p, n_obs, n_obs)
        for i in range(n_parameters):
            covinv_dcov[i] = solve_or_pinv(
                cov0,
                dcov[i],
                rcond=rcond,
                assume_symmetric=True,
                warn_context=f"C^{-1} dC solve (i={i})",
            )
        fisher_cov = 0.5 * np.einsum("iab,jba->ij", covinv_dcov, covinv_dcov)

    # Term with derivatives of model mean functions:
    # mu_{,i}^T C^{-1} mu_{,j}
    fisher_mean = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    fisher_mean = np.asarray(
        get_forecast_tensors(
            function=function,
            theta0=theta0,
            cov=cov0,
            forecast_order=1,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        ),
        dtype=np.float64,
    )

    fisher = fisher_mean + fisher_cov
    return 0.5 * (fisher + fisher.T)
