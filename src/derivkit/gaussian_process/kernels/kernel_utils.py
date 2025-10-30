"""Common helpers for GP kernels."""

from __future__ import annotations

from typing import Mapping

import numpy as np

_EPS = 1e-12  # avoid zeros/tiny scales


def to_2d(x: np.ndarray) -> np.ndarray:
    """Returns a 2D array with shape (n_points, n_dims).

    For 1D input, this treats entries as samples of a single feature and returns
    a column matrix of shape (n_points, 1). For 0D input, returns shape (1, 1).

    Args:
        x: Input array (0D, 1D, or 2D).

    Returns:
        A float array with shape (n_points, n_dims).

    Raises:
        ValueError: If the input has more than 2 dimensions, is empty, or contains
            non-finite values (NaN or inf).
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim > 2:
        raise ValueError("Input array must have at most 2 dimensions.")
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input array must contain only finite values.")

    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        # Interpret 1D as (n_points,) -> column vector (n_points, 1)
        return arr.reshape(-1, 1)
    return arr

def output_variance(params: Mapping[str, float | np.ndarray]) -> float:
    """Returns the output variance (squared output scale).

    Args:
        params: Kernel parameters containing an ``"output_scale"`` entry.

    Returns:
        The squared amplitude as a float.

    Raises:
        KeyError: If ``"output_scale"`` is missing.
        TypeError: If ``"output_scale"`` is not a scalar float or 0-d ndarray (ints allowed).
        ValueError: If ``"output_scale"`` is not finite or not > 0.
    """
    if "output_scale" not in params:
        raise KeyError("Missing 'output_scale' in params.")

    val = params["output_scale"]

    if isinstance(val, np.ndarray):
        if val.ndim != 0:
            raise TypeError("'output_scale' must be a scalar float or 0-d ndarray (int/float allowed).")
        val = float(val)
    elif isinstance(val, (int, float, np.floating)):
        val = float(val)
    else:
        raise TypeError("'output_scale' must be a scalar float or 0-d ndarray (int/float allowed).")

    if not np.isfinite(val):
        raise ValueError("'output_scale' must be finite.")
    if val <= 0.0:
        raise ValueError("'output_scale' must be > 0.")

    return val * val


def axis_length_scale_squared(params: Mapping[str, float | np.ndarray], axis: int) -> float:
    """Returns the squared length scale for a given axis.

    Works with either a single (isotropic) length scale or an
    Automatic Relevance Determination (ARD) vector.

    Args:
        params: Kernel parameters containing a ``"length_scale"`` entry.
        axis: Zero-based axis index used when ``length_scale`` is a vector.

    Returns:
        The squared length scale as a float.

    Raises:
        KeyError: If ``"length_scale"`` is missing.
        TypeError: If ``"length_scale"`` is not a scalar float/0-d ndarray/1-d ndarray,
            or if ``axis`` is not an integer.
        ValueError: If the chosen length scale is not finite and > 0,
            or if ``axis`` is out of range for a 1-d vector.
    """
    if "length_scale" not in params:
        raise KeyError("Missing 'length_scale' in params.")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("'axis' must be an integer.")

    ls = params["length_scale"]

    # Normalize types
    if isinstance(ls, np.ndarray):
        if ls.ndim == 0:
            val = float(ls)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError("'length_scale' must be finite and > 0.")
            return val * val
        elif ls.ndim == 1:
            if axis < 0 or axis >= ls.shape[0]:
                raise ValueError(f"'axis'={axis} is out of range for length_scale of shape {ls.shape}.")
            val = float(ls[axis])
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError("Selected ARD 'length_scale' entry must be finite and > 0.")
            return val * val
        else:
            raise TypeError("'length_scale' ndarray must be 0-d (scalar) or 1-d (ARD).")
    elif isinstance(ls, (int, float, np.floating)):
        val = float(ls)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError("'length_scale' must be finite and > 0.")
        return val * val
    else:
        raise TypeError("'length_scale' must be a scalar float or an ndarray (0-d or 1-d).")


def rbf_similarity(xa: np.ndarray, xb: np.ndarray, params: Mapping[str, float | np.ndarray]) -> np.ndarray:
    """Returns the RBF (squared exponential) similarity between two point sets.

    Computes exp(-0.5 * squared_distance / length_scale^2) with either a single
    shared length scale (isotropic) or one per dimension (automatic relevance determination, ARD).

    Args:
        xa: Array of samples with shape (n_a, n_dims), (n_a,), or scalar.
        xb: Array of samples with shape (n_b, n_dims), (n_b,), or scalar.
        params: Mapping that must contain key ``"length_scale"``. This can be a
            positive float (isotropic) or a 1D array of positive floats with
            length equal to the number of columns in the inputs (ARD).

    Returns:
        Array of shape (n_a, n_b) with pairwise RBF similarities.

    Raises:
        KeyError: If ``"length_scale"`` is missing from ``params``.
        TypeError: If ``length_scale`` has an invalid type.
        ValueError: If inputs have incompatible shapes or if ``length_scale`` is non-positive
            or has the wrong shape.
    """
    xa = to_2d(xa)
    xb = to_2d(xb)

    ell = np.asarray(params["length_scale"], dtype=float)
    if ell.ndim == 0:
        # Isotropic => promote to ARD vector of shape (n_dims,)
        ell = np.full(xa.shape[1], float(ell), dtype=float)

    if ell.ndim != 1 or ell.shape[0] != xa.shape[1]:
        raise ValueError("length_scale must be scalar or 1D with length equal to n_dims.")
    if not np.all(np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError("All length_scale entries must be positive and finite.")

    diff = xa[:, None, :] - xb[None, :, :]
    z = diff / ell.reshape(1, 1, -1)  # per-dimension scaling
    sqdist = np.einsum("nij,nij->ni", z, z, optimize=True)
    return np.exp(-0.5 * sqdist)
