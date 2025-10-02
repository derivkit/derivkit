"""Lightweight utility functions used across DerivKit.

These helpers have no heavy dependencies or side effects and are safe to import
from anywhere (library code, tests, notebooks). They cover small conveniences
for logging, quick sanity checks, simple finite-difference heuristics, grid
symmetry checks, and example/test function generators.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "log_debug_message",
    "is_finite_and_differentiable",
    "normalize_derivative",
    "central_difference_error_estimate",
    "is_symmetric_grid",
    "generate_test_function",
    "get_partial_function",
    "solve_or_pinv",
]


def log_debug_message(
    message: str,
    debug: bool = False,
    log_file: str | None = None,
    log_to_file: bool | None = None,
) -> None:
    """Optionally print and/or append a debug message.

    Args:
        message: Text to log.
        debug: If True, print to stdout.
        log_file: Path to a log file (used if ``log_to_file`` is True).
        log_to_file: If True, append to ``log_file`` when provided.
    """
    if not debug and not log_to_file:
        return
    if debug:
        print(message)
    if log_to_file and log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[log_debug_message] Failed to write to log file: {e}")


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


def normalize_derivative(
    derivative: float | np.ndarray,
    reference: float | np.ndarray,
) -> np.ndarray:
    """Convert a derivative to a dimensionless relative deviation.

    Computes the signed relative difference with respect to a reference scale:
    ``(derivative - reference) / (abs(reference) + 1e-12)``. This centers the
    result at zero (when ``derivative == reference``) and expresses deviations
    in units of the reference magnitude. The small epsilon prevents blow-ups
    when ``reference`` is near zero.

    Args:
      derivative: Value(s) to normalize.
      reference: Reference scale (same broadcastable shape as ``derivative``).

    Returns:
      Normalized value(s) as a NumPy array.
    """
    return (np.asarray(derivative) - np.asarray(reference)) / (
        np.abs(reference) + 1e-12
    )


def central_difference_error_estimate(step_size, order: int = 1):
    """Rule-of-thumb truncation error for central differences.

    This estimate comes from the leading term in the Taylor expansion of
    central-difference formulas. It gives the expected order of magnitude of
    the truncation error but is not an exact bound—hence “heuristic.”

    Args:
      step_size: Grid spacing.
      order: Derivative order (1–4 supported).

    Returns:
      Estimated truncation error scale.

    Raises:
      ValueError: If ``order`` is not in {1, 2, 3, 4}.
    """
    if order == 1:
        return step_size**2 / 6
    if order == 2:
        return step_size**2 / 12
    if order == 3:
        return step_size**2 / 20
    if order == 4:
        return step_size**2 / 30
    raise ValueError("Only derivative orders 1–4 are supported.")


def is_symmetric_grid(x_vals):
    """Return True if ``x_vals`` are symmetric about zero (within tolerance)."""
    x_vals = np.sort(np.asarray(x_vals))
    n = len(x_vals)
    mid = n // 2
    return np.allclose(x_vals[:mid], -x_vals[:mid:-1])


def generate_test_function(name: str = "sin"):
    """Return (f, f', f'') tuple for a named test function.

    Args:
        name: One of {"sin"}; more may be added.

    Returns:
        Tuple of callables (f, df, d2f) for testing.
    """
    if name == "sin":
        return lambda x: np.sin(x), lambda x: np.cos(x), lambda x: -np.sin(x)
    raise ValueError(f"Unknown test function: {name!r}")


def get_partial_function(
    full_function: Callable,
    variable_index: int,
    fixed_values: list | np.ndarray,
) -> Callable:
    """Returns a single-variable version of a multivariate function.

    A single parameter must be specified by index. All others parameters
    are held fixed.

    Args:
        full_function (callable): A function that takes a list of
            n_parameters parameters and returns a vector of n_observables
            observables.
        variable_index (int): The index of the parameter to treat as the
            variable.
        fixed_values (list or np.ndarray): The list of parameter values to
            use as fixed inputs for all parameters except the one being
            varied.

    Returns:
        callable: A function of a single variable, suitable for use in
            differentiation.

    Raises:
        ValueError: If ``fixed_values`` is not 1D or if `variable_index`` is out of bounds.
        TypeError: If ``variable_index`` is not an integer.
        IndexError: If ``variable_index`` is out of bounds for the size of ``fixed_values``.
    """
    fixed_arr = np.asarray(fixed_values, dtype=float)
    if fixed_arr.ndim != 1:
        raise ValueError(
            f"fixed_values must be 1D; got shape {fixed_arr.shape}."
        )
    if not isinstance(variable_index, (int, np.integer)):
        raise TypeError(
            f"variable_index must be an integer; got {type(variable_index).__name__}."
        )
    if variable_index < 0 or variable_index >= fixed_arr.size:
        raise IndexError(
            f"variable_index {variable_index} out of bounds for size {fixed_arr.size}."
        )

    def partial_function(x):
        params = fixed_arr.copy()
        params[variable_index] = x
        return np.atleast_1d(full_function(params))

    return partial_function


def solve_or_pinv(matrix: np.ndarray, vector: np.ndarray, *, rcond: float = 1e-12,
                  assume_symmetric: bool = True, warn_context: str = "linear solve") -> np.ndarray:
    """Solve ``system_matrix @ x = rhs`` with pseudoinverse fallback.

    If ``assume_symmetric`` is True (e.g., Fisher matrices), attempt a
    Cholesky-based solve. On failure (not SPD / singular), warn and fall
    back to ``pinv(system_matrix, rcond) @ rhs``.

    Args:
      matrix: Coefficient matrix of shape ``(n, n)``.
      vector: Right-hand side vector or matrix of shape ``(n,)`` or ``(n, k)``.
      rcond: Cutoff for small singular values used by ``np.linalg.pinv``.
      assume_symmetric: If True, prefer a Cholesky solve (fast path for SPD).
      warn_context: Short label included in the warning message.

    Returns:
      Solution array ``x`` with shape matching ``rhs`` (``(n,)`` or ``(n, k)``).

    Raises:
      ValueError: If shapes of ``system_matrix`` and ``rhs`` are incompatible.
    """
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)

    try:
        if assume_symmetric:
            # Fast path for symmetric positive definite matrices
            spd_matrix = np.linalg.cholesky(matrix)
            # Cholesky: matrix = L @ L.T
            # Solve in two steps: (1) L y = b, (2) L.T x = y
            y = np.linalg.solve(spd_matrix, vector)
            return np.linalg.solve(spd_matrix.T, y)
        else:
            return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        # Fall back to pseudo inverse with a helpful warning
        try:
            cond = np.linalg.cond(matrix)
            cond_msg = f" (cond≈{cond:.2e})"
        except Exception:
            cond_msg = ""
        warnings.warn(
            f"In {warn_context}, the matrix was not SPD or was singular. "
            f"Falling back to pseudoinverse with rcond={rcond}{cond_msg}.",
            RuntimeWarning,
        )
        return np.linalg.pinv(matrix, rcond=rcond) @ vector
