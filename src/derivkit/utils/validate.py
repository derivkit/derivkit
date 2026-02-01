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
    "validate_fisher_shape",
    "validate_dali_shape",
    "resolve_dali_introduced_multiplet",
    "resolve_dali_assembled_multiplet",
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


def validate_fisher_shape(
    theta0: NDArray[np.floating],
    fisher: Any,
    *,
    check_finite: bool = False,
) -> None:
    """Validates Fisher matrix shape (and optionally finiteness).

    Requirements:
      - ``theta0`` is a non-empty 1D array of length ``p``.
      - ``fisher`` is a 2D array with shape ``(p, p)``.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)``.
        fisher: Fisher matrix with shape ``(p, p)``.
        check_finite: If ``True``, require all entries of ``fisher`` to be finite.

    Raises:
        ValueError: If ``theta0`` is empty or if ``fisher`` does not have shape ``(p, p)``.
        FloatingPointError: If ``check_finite=True`` and ``fisher`` contains non-finite values.
    """
    theta0_arr = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError(
            f"theta0 must be non-empty 1D; got shape {np.asarray(theta0).shape}."
        )
    p = int(theta0_arr.size)

    f_arr = np.asarray(fisher, dtype=np.float64)
    if f_arr.ndim != 2 or f_arr.shape != (p, p):
        raise ValueError(
            f"fisher must have shape {(p, p)}; got {f_arr.shape}.")

    if check_finite and not np.isfinite(f_arr).all():
        raise FloatingPointError("fisher contains non-finite values.")


def validate_dali_shape(
    theta0: NDArray[np.floating],
    dali: Any,
    *,
    check_finite: bool = False,
) -> None:
    """Validates forecast tensor shapes.

    The accepted input forms match the conventions used by
    :func:`derivkit.forecasting.get_forecast_tensors`:

    - A dict mapping ``order -> multiplet`` for consecutive orders starting at 1.
    - A single multiplet tuple.

    With ``p = len(theta0)``, the required shapes are:

    - order 1 multiplet: ``(F,)`` with ``F`` of shape ``(p, p)``.
    - order 2 multiplet: ``(D_{(2,1)}, D_{(2,2)})`` with shapes
      ``(p, p, p)`` and ``(p, p, p, p)``.
    - order 3 multiplet: ``(T_{(3,1)}, T_{(3,2)}, T_{(3,3)})`` with shapes
      ``(p, p, p, p)``, ``(p, p, p, p, p)``, and ``(p, p, p, p, p, p)``.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)``.
        dali: Forecast tensors to validate. Must be either:

            - ``dict[int, tuple[...]]`` where each value is a multiplet for that order, or
            - ``tuple[...]`` which is a single multiplet.

        check_finite: If ``True``, require all validated arrays to be finite.

    Raises:
        TypeError: If ``dali`` has an unsupported type, if dict keys are not ints,
            or if any multiplet is not a tuple.
        ValueError: If dict keys are not consecutive starting at 1, if a multiplet
            has the wrong length for its order, or if any tensor has the wrong
            dimension/shape.
        FloatingPointError: If ``check_finite=True`` and any validated array contains
            non-finite values.
    """
    theta0_arr = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError(
            f"theta0 must be non-empty 1D; got shape {np.asarray(theta0).shape}."
        )
    p = int(theta0_arr.size)

    def _require_tensor(arr_like: Any, *, idx: int, expected_ndim: int) -> None:
        """Validate a single forecast tensor against the expected ``(p,)*ndim`` shape.

        This helper enforces the core tensor contract used by forecast tensors:
        each axis has length ``p = len(theta0)`` and the tensor has a fixed rank
        determined by its position in the multiplet.

        Args:
            arr_like: Array-like object to validate.
            idx: Index of the tensor within its multiplet. Used only for error messages.
            expected_ndim: Expected tensor rank.

        Raises:
            ValueError: If the array does not have ``expected_ndim`` dimensions or does
                not have shape ``(p,) * expected_ndim``.
            FloatingPointError: If ``check_finite=True`` in the enclosing scope and the
                array contains non-finite values.
        """
        arr = np.asarray(arr_like, dtype=np.float64)
        if arr.ndim != expected_ndim:
            raise ValueError(
                f"DALI tensor at position {idx} must have ndim"
                f"={expected_ndim}; got ndim={arr.ndim}."
            )
        expected_shape = (p,) * expected_ndim
        if arr.shape != expected_shape:
            raise ValueError(
                f"DALI tensor at position {idx} must have shape"
                f" {expected_shape}; got {arr.shape}."
            )
        if check_finite and not np.isfinite(arr).all():
            raise FloatingPointError(
                f"DALI tensor at position {idx} contains non-finite values."
            )

    def _validate_order_multiplet(order: int, m: Any) -> None:
        """Validate the multiplet structure for a specific forecast order.

        This helper checks that ``m`` is a tuple of the correct length for the given
        ``order`` and validates the shape of each tensor in the tuple.

        Conventions with ``p = len(theta0)``:

          - ``order == 1``: ``m == (F,)`` with ``F`` of shape ``(p, p)``.
          - ``order == 2``: ``m == (D_{(2,1)}, D_{(2,2)})`` with shapes
            ``(p, p, p)`` and ``(p, p, p, p)``.
          - ``order == 3``: ``m == (T_{(3,1)}, T_{(3,2)}, T_{(3,3)})`` with shapes
            ``(p, p, p, p)``, ``(p, p, p, p, p)``, and ``(p, p, p, p, p, p)``.

        Args:
            order: Forecast order associated with this multiplet.
            m: Candidate multiplet tuple to validate.

        Raises:
            TypeError: If ``m`` is not a tuple.
            ValueError: If ``order`` is unsupported or if the tuple length does not
                match the expected structure for that order, or if any tensor shape
                is invalid.
            FloatingPointError: If ``check_finite=True`` in the enclosing scope and any
                tensor contains non-finite values.
        """
        if not isinstance(m, tuple):
            raise TypeError(f"dali[order={order}] must be a tuple; got {type(m)}.")

        if order == 1:
            if len(m) != 1:
                raise ValueError(f"dali[1] must be a 1-tuple (F,); got length {len(m)}.")
            validate_fisher_shape(theta0_arr, m[0], check_finite=check_finite)
            return

        if order == 2:
            if len(m) != 2:
                raise ValueError(f"dali[2] must be a 2-tuple (D21, D22); got length {len(m)}.")
            _require_tensor(m[0], idx=0, expected_ndim=3)
            _require_tensor(m[1], idx=1, expected_ndim=4)
            return

        if order == 3:
            if len(m) != 3:
                raise ValueError(
                    f"dali[3] must be a 3-tuple (T31, T32, T33); got length {len(m)}."
                )
            _require_tensor(m[0], idx=0, expected_ndim=4)
            _require_tensor(m[1], idx=1, expected_ndim=5)
            _require_tensor(m[2], idx=2, expected_ndim=6)
            return

        raise ValueError(f"Unsupported forecast order={order}. Expected 1, 2, or 3.")

    def _validate_tuple_multiplet(m: tuple[Any, ...]) -> None:
        """Validate a single multiplet tuple and infer its forecast order from structure.

        The order is inferred using the tuple length together with the rank of the first
        entry:

          - ``(F,)``: length 1 and ``ndim(F) == 2``.
          - ``(D_{(2,1)}, D_{(2,2)})``: length 2 and ``ndim(D_{(2,1)}) == 3``.
          - ``(T_{(3,1)}, T_{(3,2)}, T_{(3,3)})``: length 3 and ``ndim(T_{(3,1)}) == 4``.

        This helper exists to accept tuple inputs in a way that is consistent with the
        per-order multiplet convention.

        Args:
            m: Candidate multiplet tuple.

        Raises:
            ValueError: If ``m`` is empty or does not match one of the supported tuple
                structures.
            TypeError/FloatingPointError: Propagated from the order-specific validation
                helpers when shapes or finiteness checks fail.
        """
        if len(m) == 0:
            raise ValueError("DALI tuple must be non-empty.")

        first_ndim = np.asarray(m[0], dtype=np.float64).ndim

        # Disambiguate strictly by (len, ndim of first tensor).
        if len(m) == 1 and first_ndim == 2:
            _validate_order_multiplet(1, m)
            return
        if len(m) == 2 and first_ndim == 3:
            _validate_order_multiplet(2, m)
            return
        if len(m) == 3 and first_ndim == 4:
            _validate_order_multiplet(3, m)
            return

        raise ValueError(
            "Unrecognized DALI tuple form."
            " Expected (F,)"
            " or (D21,D22) or"
            " (T31,T32,T33)."
        )

    # dict[int, tuple[...]]: get_forecast_tensors output
    if isinstance(dali, dict):
        if len(dali) == 0:
            raise ValueError("DALI dict is empty.")

        # keys must be int, consecutive, starting at 1
        keys: list[int] = []
        for k in dali.keys():
            if not isinstance(k, int):
                raise TypeError(f"DALI dict keys must be int;"
                                f" got {k!r} ({type(k)}).")
            keys.append(k)

        keys_sorted = sorted(keys)
        if keys_sorted[0] != 1:
            raise ValueError(f"DALI dict must start at key=1;"
                             f" got keys {keys_sorted}.")
        if keys_sorted != list(range(1, keys_sorted[-1] + 1)):
            raise ValueError(
                f"DALI dict keys must be consecutive 1..K; got keys {
                keys_sorted}."
            )

        for order in keys_sorted:
            _validate_order_multiplet(order, dali[order])
        return

    # tuple: single introduced-at-order multiplet
    if isinstance(dali, tuple):
        _validate_tuple_multiplet(dali)
        return

    raise TypeError(
        "Invalid DALI type. Expected dict[int, tuple[...]] or tuple[...] "
        "containing an introduced-at-order multiplet."
    )


def resolve_dali_introduced_multiplet(
    theta0: NDArray[np.floating],
    dali: Any,
    *,
    forecast_order: int | None = None,
    check_finite: bool = False,
) -> tuple[int, tuple[NDArray[np.float64], ...]]:
    """"Returns ``(order, multiplet)`` from any accepted forecast tensor output.

    The accepted input forms match the conventions used by
     :func:`derivkit.forecasting.get_forecast_tensors`:

    - A dict mapping ``order -> multiplet`` for consecutive orders starting at 1.
    - A single multiplet tuple.

    If ``dali`` is a dict and ``forecast_order`` is not provided, the highest
    available order is selected. If ``forecast_order`` is provided, it must be
    present in the dict.

    If ``dali`` is a tuple, the order is inferred from the tuple structure.

    Args:
        theta0: Fiducial parameter vector with shape ``(p,)``.
        dali: Forecast tensors in one of the accepted forms.
        forecast_order: Optional order selector when ``dali`` is a dict.
        check_finite: If ``True``, require all selected arrays to be finite.

    Returns:
        Tuple ``(order, multiplet)`` where ``multiplet`` is a tuple of float64 arrays.

    Raises:
        TypeError/ValueError/FloatingPointError: If ``dali`` is invalid, if the
            selected order does not exist, or if array shapes/values do not satisfy
            the validation rules.
    """
    theta0_arr = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError(
            f"theta0 must be non-empty 1D; got shape {np.asarray(theta0).shape}."
        )

    validate_dali_shape(theta0_arr, dali, check_finite=check_finite)

    if isinstance(dali, dict):
        available = sorted(dali.keys())
        chosen = available[-1] if forecast_order is None else int(forecast_order)
        if chosen not in dali:
            raise ValueError(f"forecast_order={chosen} not in DALI dict keys {
            available}.")
        multiplet = tuple(np.asarray(x, dtype=np.float64) for x in dali[chosen])
        return chosen, multiplet

    # tuple: infer order from strict (len, first_ndim)
    m = dali  # validated as tuple by validate_dali_shape
    first_ndim = np.asarray(m[0], dtype=np.float64).ndim

    if len(m) == 1 and first_ndim == 2:
        order = 1
    elif len(m) == 2 and first_ndim == 3:
        order = 2
    elif len(m) == 3 and first_ndim == 4:
        order = 3
    else:
        # Should be unreachable because validate_dali_shape already enforced.
        raise RuntimeError("internal error: could not infer order from validated tuple.")

    if forecast_order is not None and int(forecast_order) != order:
        raise ValueError(
            f"forecast_order={int(forecast_order)} does not match inferred order={order}."
        )

    multiplet = tuple(np.asarray(x, dtype=np.float64) for x in m)
    return order, multiplet


def resolve_dali_assembled_multiplet(
    theta0: NDArray[np.floating],
    dali: Any,
    *,
    forecast_order: int | None = None,
    check_finite: bool = False,
) -> tuple[int, tuple[NDArray[np.float64], ...]]:
    """Return ``(order, multiplet)`` where multiplet is assembled up to ``order``.

    Accepted inputs (matching get_forecast_tensors):
      - dict[int, tuple[...]]: per-order "introduced-at-order" multiplets
      - tuple[...]: a single introduced-at-order multiplet

    Returned multiplets are *assembled* as:
      - order 1: (F,)
      - order 2: (F, D1, D2)
      - order 3: (F, D1, D2, T1, T2, T3)

    Notes:
      - Tuple inputs cannot be assembled for order>1 because they do not include F.
        For order>1 evaluation, pass the dict form from get_forecast_tensors.
    """
    theta0_arr = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError(
            f"theta0 must be non-empty 1D; got shape {np.asarray(theta0).shape}."
        )

    validate_dali_shape(theta0_arr, dali, check_finite=check_finite)

    if isinstance(dali, dict):
        available = sorted(dali.keys())
        chosen = available[-1] if forecast_order is None else int(forecast_order)
        if chosen not in dali:
            raise ValueError(f"forecast_order={chosen} not in DALI dict keys {available}.")
        if chosen not in (1, 2, 3):
            raise ValueError(f"forecast_order must be 1, 2, or 3; got {chosen}.")

        # Always include Fisher
        f = np.asarray(dali[1][0], dtype=np.float64)

        if chosen == 1:
            return 1, (f,)

        d1 = np.asarray(dali[2][0], dtype=np.float64)
        d2 = np.asarray(dali[2][1], dtype=np.float64)

        if chosen == 2:
            return 2, (f, d1, d2)

        t1 = np.asarray(dali[3][0], dtype=np.float64)
        t2 = np.asarray(dali[3][1], dtype=np.float64)
        t3 = np.asarray(dali[3][2], dtype=np.float64)
        return 3, (f, d1, d2, t1, t2, t3)

    # tuple input: can only safely support Fisher-only (because order>1 tuples have no F)
    m = dali  # validated as tuple
    first_ndim = np.asarray(m[0], dtype=np.float64).ndim

    if len(m) == 1 and first_ndim == 2:
        if forecast_order is not None and int(forecast_order) != 1:
            raise ValueError(
                "forecast_order>1 requires the dict form from get_forecast_tensors "
                "(tuple multiplets do not include Fisher for order>1)."
            )
        f = np.asarray(m[0], dtype=np.float64)
        return 1, (f,)

    # If it's an introduced-at-order tuple of order 2 or 3, we refuse assembly.
    if (len(m) == 2 and first_ndim == 3) or (len(m) == 3 and first_ndim == 4):
        raise ValueError(
            "Order>1 evaluation requires the dict form from get_forecast_tensors, "
            "because introduced-at-order tuples do not include the Fisher matrix."
        )

    # Should be unreachable because validate_dali_shape already enforced allowed tuple forms.
    raise RuntimeError("internal error: could not infer order from validated tuple.")


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
        | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    theta0: NDArray[np.float64],
    validate: Callable[[Any], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]] | None]:
    """Returns the covariance-like input after validation.

    Args:
        cov: Covariance input. You can pass:

              - A fixed square covariance array (constant covariance).
                In this case the returned callable is ``None``.
              - A callable that takes ``theta`` and returns a square
                covariance array. In this case the function evaluates
                it at ``theta0`` to get ``cov0`` and returns the callable
                as ``cov_fn``.

        theta0: Fiducial parameter vector. Only used when ``cov`` is a callable
            covariance function (or when a callable is provided in the tuple
            form). Ignored for fixed covariance arrays.
        validate: A function that converts a covariance-like input into a NumPy
            array and checks its basic shape (and any other rules the caller
            wants). ``resolve_covariance_input`` exists to handle the different
            input types for ``cov`` (array vs callable) and to consistently
            produce ``(cov0, cov_fn)``; ``validate`` is only used to check or
            coerce the arrays that come out of that process.

    Returns:
        A tuple with two items:

        - ``cov0``: The validated covariance at ``theta0`` (or the provided
          fixed covariance).
        - ``cov_fn``: The callable covariance function if provided,
          otherwise ``None``.
    """
    if callable(cov):
        return validate(cov(theta0)), cov

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
