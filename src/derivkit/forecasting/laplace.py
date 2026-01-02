"""Laplace approximation utilities.

Implements a Gaussian approximation to a posterior around a point ``theta_map``.
This is typically the *maximum a posteriori* (MAP). It uses a second-order Taylor expansion
of the negative log-posterior::

    neg_logpost(theta) = -logposterior(theta)
    neg_logpost(theta) ≈ neg_logpost(theta_map) + 0.5 * d^T H d

where ``d = theta - theta_map`` and ``H`` is the Hessian of ``neg_logpost`` evaluated
at ``theta_map``. The approximate covariance is ``cov ≈ H^{-1}``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.linalg import solve_or_pinv
from derivkit.utils.linalg_laplace import make_spd_by_jitter, symmetrize_matrix
from derivkit.utils.validate_laplace import (
    validate_square_matrix_finite,
    validate_theta_1d_finite,
)

__all__ = [
    "negative_logposterior",
    "laplace_hessian",
    "laplace_covariance",
    "laplace_approximation",
]


def negative_logposterior(
    theta: Sequence[float] | NDArray[np.float64],
    *,
    logposterior: Callable[[NDArray[np.float64]], float],
) -> float:
    """Computes the negative log-posterior at ``theta``.

    This converts a user-provided log-posterior (often returned as a log-likelihood
    plus log-prior) into the objective most optimizers and curvature-based methods
    work with: the negative log-posterior.

    In practice, this is the scalar function whose Hessian at the *maximum a posteriori* (MAP)
    point defines the Laplace (Gaussian) approximation to the posterior, and it is also
    the quantity minimized by MAP estimation routines.

    Args:
        theta: 1D array-like parameter vector.
        logposterior: Callable that accepts a 1D float64 array and returns a scalar float.

    Returns:
        Negative log-posterior value as a float.

    Raises:
        ValueError: If ``theta`` is not a finite 1D vector or if the negative log-posterior
            evaluates to a non-finite value.
    """
    theta_array = validate_theta_1d_finite(theta, name="theta")
    negative_log_posterior_value = -float(logposterior(theta_array))

    if not np.isfinite(negative_log_posterior_value):
        raise ValueError("Negative log-posterior evaluated to a non-finite value.")

    return negative_log_posterior_value


def laplace_hessian(
    *,
    neg_logposterior: Callable[[NDArray[np.float64]], float],
    theta_map: Sequence[float] | NDArray[np.float64],
    method: str | None = None,
    n_workers: int = 1,
    dk_kwargs: Mapping[str, Any] | None = None,
) -> NDArray[np.float64]:
    """Computes the Hessian of the negative log-posterior at ``theta_map``.

    The Hessian at ``theta_map`` measures the local curvature of the posterior peak.
    In the Laplace approximation, this Hessian plays the role of a local precision
    matrix, and its inverse provides a fast Gaussian estimate of parameter
    uncertainties and correlations.

    Internally, this uses :class:`derivkit.calculus_kit.CalculusKit` (and therefore
    DerivKit's Hessian construction machinery) to differentiate the scalar objective
    ``neg_logposterior(theta)``.

    Args:
        neg_logposterior: Callable returning the scalar negative log-posterior.
        theta_map: Point where the curvature is evaluated (typically the MAP).
        method: Derivative method name/alias forwarded to the calculus machinery.
        n_workers: Outer parallelism forwarded to Hessian construction.
        dk_kwargs: Extra keyword arguments forwarded to :meth:`DerivativeKit.differentiate`.

    Returns:
        A symmetric 2D array with shape ``(p, p)`` giving the Hessian of
        ``neg_logposterior`` evaluated at ``theta_map``.

    Raises:
        TypeError: If ``neg_logposterior`` is not scalar-valued (Hessian is not 2D).
        ValueError: If inputs are invalid or the Hessian is not a finite square matrix.
    """
    theta_map_array = validate_theta_1d_finite(theta_map, name="theta_map")

    derivative_options = dict(dk_kwargs) if dk_kwargs is not None else {}
    calculus_kit = CalculusKit(neg_logposterior, x0=theta_map_array)

    hessian_raw = np.asarray(
        calculus_kit.hessian(method=method, n_workers=n_workers, **derivative_options),
        dtype=np.float64,
    )

    if hessian_raw.ndim != 2:
        raise TypeError(
            "laplace_hessian requires a scalar negative log-posterior; "
            f"got Hessian with ndim={hessian_raw.ndim} and shape {hessian_raw.shape}."
        )

    hessian_matrix = validate_square_matrix_finite(hessian_raw, name="Hessian")

    expected_shape = (theta_map_array.size, theta_map_array.size)
    if hessian_matrix.shape != expected_shape:
        raise ValueError(f"Hessian must have shape {expected_shape}, got {hessian_matrix.shape}.")

    return symmetrize_matrix(hessian_matrix)


def laplace_covariance(
    hessian: NDArray[np.float64],
    *,
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """Computes the Laplace covariance matrix from a Hessian.

    In the Laplace (Gaussian) approximation, the Hessian of the negative
    log-posterior at the expansion point acts like a local precision matrix.
    The approximate posterior covariance is the matrix inverse of that Hessian.

    This helper inverts the Hessian using DerivKit's robust linear-solve routine
    (with a pseudoinverse fallback) and returns a symmetrized covariance matrix
    suitable for downstream use (e.g., Gaussian approximations, uncertainty
    summaries, or proposal covariances).

    Args:
        hessian: 2D square Hessian matrix (typically of the negative log-posterior).
        rcond: Cutoff for small singular values used by the pseudoinverse fallback.

    Returns:
        A 2D symmetric covariance matrix with the same shape as ``hessian``.

    Raises:
        ValueError: If ``hessian`` is not a finite square matrix.
    """
    hessian_matrix = validate_square_matrix_finite(hessian, name="Hessian")
    hessian_matrix = symmetrize_matrix(hessian_matrix)

    n_parameters = hessian_matrix.shape[0]
    identity_matrix = np.eye(n_parameters, dtype=np.float64)

    covariance_matrix = solve_or_pinv(
        hessian_matrix,
        identity_matrix,
        rcond=rcond,
        assume_symmetric=True,
        warn_context="Laplace covariance",
    )

    covariance_matrix = validate_square_matrix_finite(covariance_matrix, name="covariance")
    return symmetrize_matrix(covariance_matrix)


def laplace_approximation(
    *,
    neg_logposterior: Callable[[NDArray[np.float64]], float],
    theta_map: Sequence[float] | NDArray[np.float64],
    method: str | None = None,
    n_workers: int = 1,
    dk_kwargs: Mapping[str, Any] | None = None,
    ensure_spd: bool = True,
    rcond: float = 1e-12,
) -> dict[str, Any]:
    """Computes a Laplace (Gaussian) approximation around ``theta_map``.

    The Laplace approximation replaces the posterior near its peak with a Gaussian.
    It does this by measuring the local curvature of the negative log-posterior
    using its Hessian at ``theta_map``. The Hessian acts like a local precision
    matrix, and its inverse is the approximate covariance.

    This is useful when a fast, local summary of the posterior is needed without
    running a full MCMC sampler. The output includes the expansion point,
    negative log-posterior value there, the Hessian (local precision), and
    the covariance matrix (approximate inverse Hessian).

    Args:
        neg_logposterior: Callable that accepts a 1D float64 parameter vector and
            returns a scalar negative log-posterior value.
        theta_map: Expansion point for the approximation (typically the MAP).
        method: Derivative method name/alias forwarded to the Hessian builder.
        n_workers: Outer parallelism forwarded to Hessian construction.
        dk_kwargs: Extra keyword arguments forwarded to :meth:`derivkit.DerivativeKit.differentiate`.
        ensure_spd: If True, attempt to regularize the Hessian to be SPD by adding
            diagonal jitter (required for a valid Gaussian covariance).
        rcond: Cutoff for small singular values used by the pseudoinverse fallback
            when computing the covariance.

    Returns:
        A dictionary with keys:
          - "theta_map": 1D float64 array of the expansion point.
          - "neg_logposterior_at_map": float negative log-posterior at the expansion point.
          - "hessian": (p, p) float64 Hessian of the negative log-posterior (local precision).
          - "cov": (p, p) float64 covariance matrix (approximate inverse Hessian).
          - "jitter": float amount of diagonal jitter added (0.0 if none).

    Raises:
        TypeError: If ``neg_logposterior`` is not scalar-valued.
        ValueError: If inputs are invalid or non-finite values are encountered.
        np.linalg.LinAlgError: If ``ensure_spd=True`` and the Hessian cannot be
            regularized to be SPD.
    """
    theta_map_array = validate_theta_1d_finite(theta_map, name="theta_map")

    negative_logposterior_at_map = float(neg_logposterior(theta_map_array))
    if not np.isfinite(negative_logposterior_at_map):
        raise ValueError("Negative log-posterior evaluated to a non-finite value at theta_map.")

    hessian_matrix = laplace_hessian(
        neg_logposterior=neg_logposterior,
        theta_map=theta_map_array,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )

    diagonal_jitter = 0.0
    if ensure_spd:
        hessian_matrix, diagonal_jitter = make_spd_by_jitter(hessian_matrix)

    covariance_matrix = laplace_covariance(hessian_matrix, rcond=rcond)

    return {
        "theta_map": theta_map_array,
        "neg_logposterior_at_map": float(negative_logposterior_at_map),
        "hessian": hessian_matrix,
        "cov": covariance_matrix,
        "jitter": float(diagonal_jitter),
    }
