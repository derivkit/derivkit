"""Validation utilities for DerivativeKit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from derivkit.utils.sandbox import get_partial_function

__all__ = [
    "is_finite_and_differentiable",
    "check_scalar_valued",
    "validate_tabulated_xy",
    "validate_covariance_matrix_shape",
    "validate_symmetric_psd",
    "validate_fisher_shapes",
    "validate_dali_shapes",
    "validate_square_matrix",
    "ensure_finite",
    "normalize_theta",
    "validate_theta_1d_finite",
    "validate_square_matrix_finite",
    "resolve_covariance_input",
    "flatten_matrix_c_order",
    "require_callable",
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
      A boolean which is ``True`` if the input is finite at both points
      and ``False`` otherwise.
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
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
            This does not parallelize across parameters.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    partial_vec = get_partial_function(function, i, theta0)
    _ = n_workers

    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "build_gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )


def validate_tabulated_xy(
    x: Any,
    y: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def validate_covariance_matrix_shape(cov: Any) -> NDArray[np.float64]:
    """Validates covariance input shape: allows 0D/1D/2D; if 2D requires square."""
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim > 2:
        raise ValueError(f"cov must be at most two-dimensional; got ndim={cov_arr.ndim}.")
    if cov_arr.ndim == 2 and cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"cov must be square; got shape={cov_arr.shape}.")
    return cov_arr


def validate_symmetric_psd(
    matrix: Any,
    *,
    sym_atol: float = 1e-12,
    psd_atol: float = 1e-12,
) -> NDArray[np.float64]:
    """Validates that an input is a symmetric positive semidefinite (PSD) matrix.

    This is intended for strict validation (e.g., inputs passed to GetDist, or any
    code path where an indefinite covariance-like matrix should hard-fail). This
    is an important validation because many algorithms assume PSD inputs, and
    invalid inputs can lead to silent failures or nonsensical results.

    Policy:
      - Requires 2D square shape.
      - Requires near-symmetry within ``sym_atol`` (raises if violated).
      - After the symmetry check passes, checks PSD by computing eigenvalues of the
        symmetrized matrix ``S = 0.5 * (A + A.T)`` for numerical robustness, and
        requires ``min_eig(S) >= -psd_atol``.

    Args:
        matrix: Array-like input expected to be a covariance-like matrix.
        sym_atol: Absolute tolerance for symmetry check.
        psd_atol: Absolute tolerance for PSD check. Allows small negative eigenvalues
            down to ``-psd_atol``.

    Returns:
        A NumPy array view/copy of the input, converted to ``float`` (same values as input).

    Note:
        The input must be symmetric within ``sym_atol``; this function does not
        modify or symmetrize the returned matrix. The positive semi-definite check uses the
        symmetrized form ``0.5*(A + A.T)`` only to reduce roundoff sensitivity
        after the symmetry check passes.

    Raises:
        ValueError: If ``matrix`` is not 2D, square, is too asymmetric, contains non-finite
            values, is not PSD within tolerance, if `max(|A - A.T|) > sym_atol``,
            if ``min_eig(0.5*(A + A.T)) < -psd_atol``, or if eigenvalue computation fails.
    """
    a = np.asarray(matrix, dtype=np.float64)

    if a.ndim != 2:
        raise ValueError(f"matrix must be 2D; got ndim={a.ndim}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square; got shape={a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix contains non-finite values.")

    skew = a - a.T
    max_abs_skew = float(np.max(np.abs(skew))) if skew.size else 0.0
    if max_abs_skew > sym_atol:
        raise ValueError(
            f"matrix must be symmetric within sym_atol={sym_atol:.2e}; "
            f"max(|A-A^T|)={max_abs_skew:.2e}."
        )

    s = 0.5 * (a + a.T)
    try:
        evals = np.linalg.eigvalsh(s)
    except np.linalg.LinAlgError as e:
        raise ValueError("eigenvalue check failed for matrix (LinAlgError).") from e

    min_eig = float(np.min(evals)) if evals.size else 0.0
    if min_eig < -psd_atol:
        raise ValueError(
            f"matrix is not positive semi-definite within psd_atol={psd_atol:.2e}; min eigenvalue={min_eig:.2e}."
        )

    return a


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
      ``(p, p, p, p)``  with ``p`` being the number of parameters.

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


def validate_square_matrix(a: Any, *, name: str = "matrix") -> NDArray[np.float64]:
    """Validates that the input is a 2D square matrix and return it as float array."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got ndim={arr.ndim}.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square; got shape={arr.shape}.")
    return arr


def ensure_finite(arr: Any, *, msg: str) -> None:
    """Ensures that all values in an array are finite.

    Args:
        arr: Input array-like to check.
        msg: Error message for the exception if non-finite values are found.

    Raises:
        FloatingPointError: If any value in ``arr`` is non-finite.
    """
    if not np.isfinite(np.asarray(arr)).all():
        raise FloatingPointError(msg)


def normalize_theta(theta0: Any) -> NDArray[np.float64]:
    """Ensures that data vector is a non-empty 1D float array.

    Args:
        theta0: Input array-like to validate and convert.

    Returns:
        1D float array.

    Raises:
        ValueError: if ``theta0`` is empty.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")
    return theta


def validate_theta_1d_finite(theta: Any, *, name: str = "theta") -> NDArray[np.float64]:
    """Validates that ``theta`` is a finite, non-empty 1D parameter vector and returns it as a float64 NumPy array.

    Args:
        theta: Array-like parameter vector.
        name: Name used in error messages.

    Returns:
        A 1D float64 NumPy array containing the validated parameter vector.

    Raises:
        ValueError: If ``theta`` is not 1D, is empty, or contains non-finite values.
    """
    t = np.asarray(theta, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {t.shape}.")
    if t.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{name} contains non-finite values.")
    return t.astype(np.float64, copy=False)


def validate_square_matrix_finite(
    a: Any, *, name: str = "matrix"
) -> NDArray[np.float64]:
    """Validates that ``a`` is a finite 2D square matrix and returns it as a float64 NumPy array.

    Args:
        a: Array-like matrix.
        name: Name used in error messages.

    Returns:
        A 2D float64 NumPy array containing the validated square matrix.

    Raises:
        ValueError: If ``a`` is not 2D, is not square, or contains non-finite values.
    """
    m = np.asarray(a, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be 2D; got ndim={m.ndim}.")
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be square; got shape {m.shape}.")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} contains non-finite values.")
    return m.astype(np.float64, copy=False)


def resolve_covariance_input(
    cov: NDArray[np.float64]
        | Callable[[NDArray[np.float64]], NDArray[np.float64]]
        | tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]]],
    *,
    theta0: NDArray[np.float64],
    validate: Callable[[Any], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]] | None]:
    """Processes a covariance input and return a fixed covariance and an optional callable.

    Accepts a fixed covariance array, a covariance function, or a tuple containing both,
    and returns the covariance at ``theta0`` together with the callable (if provided).

    Args:
        cov: Covariance input.
            You can pass:

              - A fixed square covariance array (constant covariance). In this case the
                returned callable is ``None``.
              - A callable that takes ``theta`` and returns a square covariance array.
                In this case the function evaluates it at ``theta0`` to get ``cov0`` and
                returns the callable as ``cov_fn``.
              - A tuple ``(cov0, cov_fn)`` where ``cov0`` is the covariance at ``theta0``
                and ``cov_fn`` is the callable covariance function. This avoids evaluating
                the callable at ``theta0`` again.
        theta0: Fiducial parameter vector. Only used when ``cov`` is a callable covariance
            function (or when a callable is provided in the tuple form). Ignored for fixed
            covariance arrays.
        validate: A function that converts a covariance-like input into a NumPy array and checks its
            basic shape (and any other rules the caller wants). ``resolve_covariance_input`` exists
            to handle the different input types for ``cov`` (array vs callable vs (array, callable))
            and to consistently produce ``(cov0, cov_fn)``; ``validate`` is only used to check/coerce
            the arrays that come out of that process.

    Returns:
        A tuple with two items:

        - ``cov0``: The validated covariance at ``theta0`` (or the provided fixed covariance).
        - ``cov_fn``: The callable covariance function if provided, otherwise ``None``.

    Raises:
        TypeError: If the tuple form does not have exactly two items, or if the
            second item is not callable.
    """
    if isinstance(cov, tuple):
        if len(cov) != 2:
            raise TypeError("cov tuple form must be (cov0, cov_fn).")
        cov0, cov_fn = cov
        if not callable(cov_fn):
            raise TypeError("cov tuple form must be (cov0, cov_fn) with cov_fn callable.")
        return validate(cov0), cov_fn

    if callable(cov):
        cov0 = validate(cov(theta0))
        return cov0, cov

    return validate(cov), None


def flatten_matrix_c_order(
    cov_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    theta: NDArray[np.float64],
    *,
    n_observables: int,
) -> NDArray[np.float64]:
    """Validates the output of a covariance function and flattens it to 1D.

    This function uses the convention of flattening 2D arrays in row-major ("C") order.
    The flattening is necessary when computing derivatives of covariance matrices with respect to
    parameters, as the derivative routines typically operate on 1D arrays.

    Args:
        cov_function: Callable that takes a parameter vector and returns a covariance matrix.
        theta: Parameter vector at which to evaluate the covariance function.
        n_observables: Number of observables, used to validate the shape of the covariance matrix.

    Returns:
        A 1D NumPy array representing the flattened covariance matrix.

    Raises:
        ValueError: If the output of ``cov_function`` does not have the expected shape.
    """
    cov = validate_covariance_matrix_shape(cov_function(theta))
    if cov.shape != (n_observables, n_observables):
        raise ValueError(
            f"cov_function(theta) must return shape {(n_observables, n_observables)}; got {cov.shape}."
        )
    return np.asarray(cov, dtype=np.float64).ravel(order="C")


def require_callable(
    func: Callable[..., Any] | None,
    *,
    name: str = "function",
    context: str | None = None,
    hint: str | None = None,
) -> Callable[..., Any]:
    """Ensures a required callable is provided.

    This is a small helper to validate inputs.
    If ``func`` is ``None``, it raises a ``ValueError`` with a clear message (and an
    optional context/hint to make debugging easier). If ``func`` is provided, it is
    returned unchanged so the caller can use it directly.

    Args:
        func: Callable to validate.
        name: Name shown in the error message.
        context: Optional context prefix (e.g. "ForecastKit.fisher").
        hint: Optional hint appended to the error message.

    Returns:
        The input callable.

    Raises:
        ValueError: If ``func`` is None.
    """
    if func is None:
        prefix = f"{context}: " if context else ""
        msg = f"{prefix}{name} must be provided."
        if hint:
            msg += f" {hint}"
        raise ValueError(msg)
    return func
