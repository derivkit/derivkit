"""Validation utilities for DerivativeKit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.utils.sandbox import get_partial_function

__all__ = [
    "is_finite_and_differentiable",
    "check_scalar_valued",
    "validate_tabulated_xy",
    "validate_covariance_matrix",
    "validate_fisher_shapes",
    "validate_dali_shapes",
]

def is_finite_and_differentiable(
    function: Callable[[float], Any],
    x: float,
    delta: float = 1e-5,
) -> bool:
    """Check that ``function`` is finite at ``x`` and ``x + delta``.

    Evaluates without exceptions and returns finite values at both points.

    Args:
      function: Callable ``f(x)`` returning a scalar or array-like.
      x: Probe point.
      delta: Small forward step.

    Returns:
      True if finite at both points; otherwise False.
    """
    f0 = np.asarray(function(x))
    f1 = np.asarray(function(x + delta))
    return np.isfinite(f0).all() and np.isfinite(f1).all()


def check_scalar_valued(function, theta0: np.ndarray, i: int, n_workers: int):
    """Helper used by ``build_gradient`` and ``build_hessian``.

    Args:
        function (callable): The scalar-valued function to
            differentiate. It should accept a list or array of parameter
            values as input and return a scalar observable value.
        theta0: The points at which the derivative is evaluated.
            A 1D array or list of parameter values matching the expected
            input of the function.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)

    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "build_gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )


def validate_tabulated_xy(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Validates and converts tabulated ``x`` and ``y`` arrays into NumPy arrays.

    Requirements:
      - ``x`` is 1D and strictly increasing.
      - ``y`` has at least 1 dimension.
      - ``y.shape[0] == x.shape[0]``, but ``y`` may have arbitrary trailing
        dimensions (scalar, vector, or ND output).

    Args:
        x: 1D array-like of x values (must be strictly increasing).
        y: Array-like of y values with ``y.shape[0] == len(x)``.

    Returns:
        Tuple of (x_array, y_array) as NumPy arrays.

    Raises:
        ValueError: If input arrays do not meet the required conditions.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1D.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length along axis 0.")
    if not np.all(np.diff(x_arr) > 0):
        raise ValueError("x must be strictly increasing.")
    if y_arr.ndim < 1:
        raise ValueError("y must be at least 1D.")

    return x_arr, y_arr


def validate_covariance_matrix(cov: ArrayLike) -> NDArray[np.floating]:
    """Validates and converts a covariance matrix into a NumPy array."""
    cov_arr = np.asarray(cov, dtype=float)

    if cov_arr.ndim > 2:
        raise ValueError(
            f"cov must be at most two-dimensional; got ndim={cov_arr.ndim}."
        )
    if cov_arr.ndim == 2 and cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"cov must be square; got shape={cov_arr.shape}.")

    return cov_arr


def validate_fisher_shapes(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
) -> None:
    """Validates shapes for Fisher forecasting inputs.

    Checks that:
      - ``theta0`` is a 1D parameter vector with shape ``(p,)``.
      - ``fisher`` is a square matrix with shape ``(p, p)``, where ``p = len(theta0)``
      with ``p`` being the number of parameters.

    Args:
      theta0: Expansion point (fiducial parameters) as a 1D array of length ``p``.
      fisher: Fisher information matrix as a 2D array with shape ``(p, p)``.

    Raises:
      ValueError: If ``theta0`` is not 1D, or if ``fisher`` does not have shape
        ``(p, p)``.
    """
    theta0 = np.asarray(theta0)
    fisher = np.asarray(fisher)

    if theta0.ndim != 1:
        raise ValueError(f"theta0 must be 1D, got {theta0.shape}")

    p = theta0.shape[0]
    if fisher.shape != (p, p):
        raise ValueError(f"fisher must have shape {(p, p)}, got {fisher.shape}")


def validate_dali_shapes(
    theta0: NDArray[np.floating],
    fisher: NDArray[np.floating],
    g_tensor: NDArray[np.floating],
    h_tensor: NDArray[np.floating] | None,
) -> None:
    """Validates shapes for DALI expansion inputs.

    Checks that:
      - ``theta0`` is a 1D parameter vector with shape ``(p,)``.
      - ``fisher`` has shape ``(p, p)``.
      - ``g_tensor`` (third-derivative tensor) has shape ``(p, p, p)``.
      - ``h_tensor`` (fourth-derivative tensor), if provided, has shape
        ``(p, p, p, p)``
        with ``p`` being the number of parameters.

    Args:
      theta0: Expansion point (fiducial parameters) as a 1D array of length ``p``.
      fisher: Fisher information matrix as a 2D array with shape ``(p, p)``.
      g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
      h_tensor: Optional DALI quartic tensor with shape ``(p, p, p, p)``. If
        ``None``, no quartic shape check is performed.

    Raises:
      ValueError: If any input does not have the expected dimensionality/shape.
    """
    validate_fisher_shapes(theta0, fisher)

    theta0 = np.asarray(theta0)
    g_tensor = np.asarray(g_tensor)

    p = theta0.shape[0]
    if g_tensor.shape != (p, p, p):
        raise ValueError(f"g_tensor must have shape {(p, p, p)}, got {g_tensor.shape}")

    if h_tensor is not None:
        h_tensor = np.asarray(h_tensor)
        if h_tensor.shape != (p, p, p, p):
            raise ValueError(
                f"h_tensor must have shape {(p, p, p, p)}, got {h_tensor.shape}"
            )
