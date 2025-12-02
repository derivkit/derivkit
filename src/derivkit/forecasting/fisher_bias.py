"""Provides tools for facilitating experimental forecasts.

The user must specify the observables, fiducial values and covariance matrix
at which the derivative should be evaluated. Derivatives of the first order
are Fisher derivatives. Derivatives of second order are evaluated using the
derivative approximation for likelihoods (DALI) technique as described in
https://doi.org/10.1103/PhysRevD.107.103506.

More details about available options can be found in the documentation of
the methods.
"""

from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.linalg import invert_covariance, solve_or_pinv


def build_fisher_bias(
        expansion, # : LikelihoodExpansion?
        fisher_matrix: NDArray[np.floating],
        delta_nu: NDArray[np.floating],
        n_workers: int = 1,
        method: str | None = None,
        rcond: float = 1e-12,
        **dk_kwargs: Any,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""Estimate parameter bias using the stored model, expansion point, and covariance.

    This method quantifies how differences between two data sets—typically an
    "unbiased" or reference data vector and a "biased" one including a given
    systematic—propagate into parameter biases when interpreted through a Fisher
    forecast. It evaluates the model response internally and uses it, together
    with the stored covariance and provided Fisher matrix, to estimate both the
    bias vector and the resulting shift in parameter values.
    For more information, see https://arxiv.org/abs/0710.5171.

    Args:
        fisher_matrix: Square matrix describing information about the parameters.
        Its shape must be (p, p), where p is the number of parameters.
        delta_nu: Difference between a "biased" and an "unbiased" data vector,
        for example :math:`\Delta\nu = \nu_{\mathrm{with\,sys}} - \nu_{\mathrm{without\,sys}}`.
        Accepts a 1D array of length n or a 2D array that will be flattened in
        row-major order (“C”) to length n, where n is the number of observables.
        If supplied as a 1D array, it must already follow the same row-major (“C”)
        flattening convention used throughout the package.
        n_workers: Number of workers used by the internal derivative routine when
        forming the Jacobian.
        method: Method name or alias (e.g., "adaptive", "finite").
        If None, the DerivativeKit default ("adaptive") is used.
        **dk_kwargs: Additional keyword arguments passed to DerivativeKit.differentiate.
        rcond: Regularization cutoff for pseudoinverse. Default is 1e-12.

    Returns:
        A tuple ``(bias_vec, delta_theta)`` where both entries are 1D arrays of length ``p``:
        - bias_vec: parameter-space bias vector.
        - delta_theta: estimated parameter shifts.

    Raises:
        ValueError: If input shapes are inconsistent with the stored model, covariance,
        or the Fisher matrix dimensions.
        FloatingPointError: If the difference vector contains NaNs.
    """
    n_workers = expansion._normalize_workers(n_workers)

    fisher_matrix = np.asarray(fisher_matrix, dtype=float)
    if fisher_matrix.ndim != 2 or fisher_matrix.shape[0] != fisher_matrix.shape[1]:
        raise ValueError(f"fisher_matrix must be square; got shape {fisher_matrix.shape}.")

    # Jacobian — we are enforcing (n_obs, n_params) throughout the package
    ckit = CalculusKit(expansion.function, expansion.theta0)
    j_matrix = np.asarray(
        ckit.jacobian(
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        ),
        dtype=float,
    )
    n_obs, n_params = expansion.n_observables, expansion.n_parameters
    if j_matrix.shape != (n_obs, n_params):
        raise ValueError(
            f"build_jacobian must return shape (n_obs, n_params)=({n_obs},{n_params}); "
            f"got {j_matrix.shape}."
        )

    # Shape checks consistent with J
    if expansion.cov.shape != (j_matrix.shape[0], j_matrix.shape[0]):
        raise ValueError(
            f"covariance shape {expansion.cov.shape} must be (n, n) = "
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
    off = expansion.cov.copy()
    np.fill_diagonal(off, 0.0)
    is_diag = not np.any(off)  # True iff all off-diagonals are exactly zero

    if is_diag:
        diag = np.diag(expansion.cov)
        if np.all(diag > 0):
            cinv_delta = delta_nu / diag
        else:
            cinv_delta = solve_or_pinv(
                expansion.cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
            )
    else:
        cinv_delta = solve_or_pinv(
            expansion.cov, delta_nu, rcond=rcond, assume_symmetric=True, warn_context="covariance solve"
        )

    bias_vec = j_matrix.T @ cinv_delta
    delta_theta = solve_or_pinv(
        fisher_matrix, bias_vec, rcond=rcond, assume_symmetric=True, warn_context="Fisher solve"
    )

    return bias_vec, delta_theta

def build_delta_nu(
        expansion, # : LikelihoodExpansion?
        data_with: NDArray[np.floating],
        data_without: NDArray[np.floating],
        *,
        dtype: type | np.dtype = float,
) -> NDArray[np.floating]:
    """Compute the difference between two data vectors.

    This function is typically used for Fisher-bias estimates, taking two data vectors—
    one with a systematic included and one without—and returning their difference as a
    1D array that matches the expected number of observables in this instance. It works
    with both 1D inputs and 2D arrays (for example, correlation × ell) and flattens 2D
    arrays using NumPy's row-major ("C") order, our standard convention throughout the package.

    We standardize on row-major (“C”) flattening of 2D arrays, where the last
    axis varies fastest. The user must ensure that any data vectors and associated covariances
    are constructed with the same convention for consistent results.

    Args:
        data_with: Data vector that includes the systematic effect. Can be 1D or 2D.
        If 1D, it must follow the NumPy's row-major (“C”) flattening convention used
        throughout the package.
        data_without: Reference data vector without the systematic. Can be 1D or 2D. If 1D,
        it must follow the NumPy's row-major (“C”) flattening convention used throughout
        the package.
        dtype: Data type of the output array (defaults to float).

    Returns:
        A 1D NumPy array of length ``self.n_observables`` representing the data
        mismatch (delta_nu = data_with − data_without).

    Raises:
        ValueError: If input shapes differ, inputs are not 1D/2D, or the flattened
        length does not match ``self.n_observables``.
        FloatingPointError: If non-finite values are detected in the result.
    """
    # define flattening orders for numpy
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

    if delta_nu.size != expansion.n_observables:
        raise ValueError(
            f"Flattened length {delta_nu.size} != expected self.n_observables {expansion.n_observables}."
        )

    if not np.isfinite(delta_nu).all():
        raise FloatingPointError("Non-finite values found in delta vector.")

    return delta_nu
