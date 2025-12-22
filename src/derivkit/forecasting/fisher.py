"""Fisher forecasting utilities."""

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.forecasting.forecast_core import get_forecast_tensors
from derivkit.utils.concurrency import normalize_workers
from derivkit.utils.linalg import solve_or_pinv
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "build_fisher_matrix",
    "build_fisher_bias",
    "build_delta_nu",
]


def build_fisher_matrix(
    function: Callable[[ArrayLike], float | NDArray[np.floating]],
    theta0: ArrayLike,
    cov: ArrayLike,
    *,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Computes the Fisher information matrix for a given model and covariance.

    Args:
        function: The scalar or vector-valued model function. It should accept
            a 1D array-like of parameter values and return either a scalar or
            an array of observables.
        theta0: 1D array-like of fiducial parameters (single expansion point).
            This helper currently assumes a single expansion point; if you need
            multiple expansion points with different covariances, call this
            function in a loop or work directly with ForecastKit.
        cov: Covariance matrix of the observables. Must be square with shape
            ``(n_observables, n_observables)``.
        method: Derivative method name or alias (e.g., ``"adaptive"``,
            ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default is used.
        n_workers: Number of workers for per-parameter parallelisation.
            Default is ``1`` (serial).
        **dk_kwargs: Additional keyword arguments forwarded to
            :meth:`DerivativeKit.differentiate`.

    Returns:
        Fisher matrix with shape ``(n_parameters, n_parameters)``.
    """
    fisher = get_forecast_tensors(
        function=function,
        theta0=theta0,
        cov=cov,
        forecast_order=1,
        method=method,
        n_workers=n_workers,
        **dk_kwargs,
    )
    return fisher


def build_fisher_bias(
        function: Callable[[ArrayLike], float | NDArray[np.floating]],
        theta0: ArrayLike,
        cov: ArrayLike,
        *,
        fisher_matrix: NDArray[np.floating],
        delta_nu: NDArray[np.floating],
        n_workers: int = 1,
        method: str | None = None,
        rcond: float = 1e-12,
        **dk_kwargs: Any,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""Estimates parameter bias using the stored model, expansion point, and covariance.

    This function takes a model, an expansion point, a covariance matrix,
    a Fisher matrix, and a data-vector difference ``delta_nu`` and maps that
    difference into parameter space. A common use case is the classic
    “Fisher bias” setup, where one asks how a systematic-induced change in
    the data would shift inferred parameters.

    Internally, the function evaluates the model response at the expansion
    point and uses the covariance and Fisher matrix to compute both the
    parameter-space bias vector and the corresponding shifts. See
    https://arxiv.org/abs/0710.5171 for details.

    Args:
        function: The scalar or vector-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return either a scalar or a
            :class:`np.ndarray` of observable values.
        theta0: 1D array-like of fiducial parameters (single expansion point).
            This helper currently assumes a single expansion point; if you need
            multiple expansion points with different covariances, call this
            function in a loop or work directly with ForecastKit.
        cov: The covariance matrix of the observables. Must be a square
            matrix with shape ``(n_observables, n_observables)``, where
            ``n_observables`` is the number of observables returned by the
            function.
        fisher_matrix: Square matrix describing information about the parameters.
            Its shape must be ``(p, p)``, where ``p`` is the number of parameters.
        delta_nu: Difference between a biased and an unbiased data vector,
            for example :math:`\\Delta\nu = \nu_{\\mathrm{with\\,sys}} - \nu_{\\mathrm{without\\,sys}}`.
            Accepts a 1D array of length n or a 2D array that will be flattened in
            row-major order (“C”) to length n, where n is the number of observables.
            If supplied as a 1D array, it must already follow the same row-major (“C”)
            flattening convention used throughout the package.
        n_workers: Number of workers used by the internal derivative routine when
            forming the Jacobian.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        rcond: Regularization cutoff for pseudoinverse. Default is ``1e-12``.
        **dk_kwargs: Additional keyword arguments passed to ``DerivativeKit.differentiate``.

    Returns:
        A tuple ``(bias_vec, delta_theta)`` of 1D arrays with length ``p``,
        where ``bias_vec`` is the parameter-space bias vector
        and ``delta_theta`` are the corresponding parameter shifts.

    Raises:
      ValueError: If input shapes are inconsistent with the stored model, covariance,
        or the Fisher matrix dimensions.
      FloatingPointError: If the difference vector contains at least one ``NaN``.
    """
    n_workers = normalize_workers(n_workers)

    theta0 = np.atleast_1d(theta0)
    cov = validate_covariance_matrix_shape(cov)

    n_parameters = theta0.shape[0]
    n_observables = cov.shape[0]

    fisher_matrix = np.asarray(fisher_matrix, dtype=float)
    if fisher_matrix.ndim != 2 or fisher_matrix.shape[0] != fisher_matrix.shape[1]:
        raise ValueError(f"fisher_matrix must be square; got shape {fisher_matrix.shape}.")

    # Compute the Jacobian with shape (n_obs, n_params), so that rows correspond
    # to observables and columns to parameters. This convention is used
    # throughout the forecasting utilities and is assumed by the Fisher/bias
    # algebra below.
    ckit = CalculusKit(function, theta0)
    j_matrix = np.asarray(
        ckit.jacobian(
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )
    )

    n_obs, n_params = n_observables, n_parameters
    if j_matrix.shape != (n_obs, n_params):
        raise ValueError(
            f"build_jacobian must return shape (n_obs, n_params)=({n_obs},{n_params}); "
            f"got {j_matrix.shape}."
        )

    if cov.shape != (j_matrix.shape[0], j_matrix.shape[0]):
        raise ValueError(
            f"covariance shape {cov.shape} must be (n, n) = "
            f"{(j_matrix.shape[0], j_matrix.shape[0])} from the Jacobian."
        )
    if fisher_matrix.shape != (j_matrix.shape[1], j_matrix.shape[1]):
        raise ValueError(
            f"fisher_matrix shape {fisher_matrix.shape} must be (p, p) = "
            f"{(j_matrix.shape[1], j_matrix.shape[1])} from the Jacobian."
        )

    # Make delta_nu a 1D array of length n; 2D inputs are flattened in row-major ("C") order.
    delta_nu = np.asarray(delta_nu, dtype=float)
    if delta_nu.ndim == 2:
        delta_nu = delta_nu.ravel(order="C")
    if delta_nu.ndim != 1 or delta_nu.size != n_obs:
        raise ValueError(f"delta_nu must have length n={n_obs}; got shape {delta_nu.shape}.")
    if not np.isfinite(delta_nu).all():
        raise FloatingPointError("Non-finite values found in delta_nu.")

    # GLS weighting by the inverse covariance:
    # If C is diagonal, compute invcov * delta_nu by elementwise division (fast).
    # Otherwise solve with a symmetric solver; on ill-conditioning/failure,
    # fall back to a pseudoinverse and emit a warning.
    off = cov.copy()
    np.fill_diagonal(off, 0.0)
    is_diag = not np.any(off)

    if is_diag:
        diag = np.diag(cov)
        if np.all(diag > 0):
            cinv_delta = delta_nu / diag
        else:
            cinv_delta = solve_or_pinv(
                cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
            )
    else:
        cinv_delta = solve_or_pinv(
            cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
        )

    bias_vec = j_matrix.T @ cinv_delta
    delta_theta = solve_or_pinv(
        fisher_matrix, bias_vec, rcond=rcond, assume_symmetric=True, warn_context="Fisher solve"
    )

    return bias_vec, delta_theta


def build_delta_nu(
        cov: ArrayLike,
        *,
        data_with: NDArray[np.floating],
        data_without: NDArray[np.floating],
        dtype: DTypeLike = np.float64,
) -> NDArray[np.floating]:
    """Computes the difference between two data vectors.

    This helper is used in Fisher-bias calculations and any other workflow
    where two data vectors are compared: it takes a pair of vectors (for example,
    a version with a systematic and one without) and returns their difference as a
    1D array whose length matches the number of observables implied by ``cov``.
    It works with both 1D inputs and 2D arrays (for example, correlation-by-ell)
    and flattens 2D inputs using NumPy's row-major ("C") order, which is the
    standard convention throughout the DerivKit package.

    Args:
        cov: The covariance matrix of the observables. Should be a square
            matrix with shape ``(n_observables, n_observables)``, where
            ``n_observables`` is the number of observables returned by the
            function.
        data_with: Data vector that includes the systematic effect. Can be
            1D or 2D. If 1D, it must follow the NumPy's row-major (“C”)
            flattening convention used throughout the package.
        data_without: Reference data vector without the systematic. Can be
            1D or 2D. If 1D, it must follow the NumPy's row-major (“C”)
            flattening convention used throughout the package.
        dtype: NumPy dtype for the output array (defaults to ``np.float64``,
            i.e. NumPy's default floating type).

    Returns:
        A 1D NumPy array of length ``n_observables`` representing the mismatch
        between the two input data vectors. This is simply the element-wise
        difference between the input with systematic and the input without systematic,
        flattened if necessary to match the expected observable ordering.

    Raises:
      ValueError: If input shapes differ, inputs are not 1D/2D, or the flattened
        length does not match ``n_observables``.
      FloatingPointError: If non-finite values are detected in the result.
    """
    n_observables = cov.shape[0]

    a = np.asarray(data_with, dtype=dtype)
    b = np.asarray(data_without, dtype=dtype)

    if a.shape != b.shape:
        raise ValueError(f"Shapes must match: got {a.shape} vs {b.shape}.")

    if a.ndim == 1:
        delta_nu = a - b
    elif a.ndim == 2:
        delta_nu = (a - b).ravel(order="C")
    else:
        raise ValueError(f"Only 1D or 2D inputs are supported; got ndim={a.ndim}.")

    if delta_nu.size != n_observables:
        raise ValueError(
            f"Flattened length {delta_nu.size} != expected n_observables {n_observables}."
        )

    if not np.isfinite(delta_nu).all():
        raise FloatingPointError("Non-finite values found in delta vector.")

    return delta_nu
