"""Contains functions used in constructing the Hessian of a scalar-valued function."""

from collections.abc import Callable
from functools import partial
from typing import Any, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function

__all__ = [
    "build_hessian",
    "build_hessian_diag",
]


def build_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Returns the full Hessian of a function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Parallel tasks across output components / Hessian entries.
        **dk_kwargs: Extra options forwarded to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Always returns the full Hessian with shape:

        - (p, p) if ``function(theta0)`` is scalar with ``p`` the number of parameters.
        - (``*out_shape``, p, p) if ``function(theta0)`` has shape ``out_shape``.

        The output shape is fixed; use ``build_hessian_diag()`` if only the diagonal is needed.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If a single output component (flattened scalar subpath) does not return a scalar.
    """
    return _build_hessian_internal(
        function, theta0, method=method, n_workers=n_workers, diag=False, **dk_kwargs
    )


def build_hessian_diag(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the diagonal of the Hessian of a function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Parallel tasks across output components / Hessian entries.
        **dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
            You may optionally pass ``inner_workers=<int>`` here to override
            the inner policy.

    Returns:
        Returns only the diagonal entries of the Hessian.

        - (p,) if ``function(theta0)`` is scalar.
        - (``*out_shape``, p) if ``function(theta0)`` has shape ``out_shape``.

        This reduction in rank is intentional to avoid computing or storing off-diagonal terms.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If evaluating a single output component does not return a scalar.
    """
    return _build_hessian_internal(
        function, theta0, method=method, n_workers=n_workers, diag=True, **dk_kwargs
    )


def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError


def _build_hessian_full(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: np.ndarray,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the full Hessian for a scalar- or vector-valued function.
    
    The shape of the Hessian contains two additional axes, each of which
    have length ``len(theta)``. These axes are appended to the shape of the
    original function, so e.g. a function of 3 parameters returning a vector
    with 4 components would have shape ``(4, 3, 3)``, while a scalar-valued
    function with 2 components would have shape ``(2, 2)``.

    Args:
        function: The function to be differentiated.
        theta: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
        outer_workers: Number of outer parallel workers for Hessian entries.
        inner_workers: Optional inner parallelism for the differentiation engine.
        **dk_kwargs: Additional keyword arguments for
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        An ND array representing the Hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
    """
    p = int(theta.size)

    # Here we build a list of tasks for all unique Hessian entries (i, j).
    # We only compute the upper triangle and diagonal, then mirror the results.
    # This reduces computation by nearly half.
    tasks: list[Tuple[Any, ...]] = [(function, theta, i, i, method, inner_workers, dk_kwargs) for i in range(p)]
    tasks += [
        (function, theta, i, j, method, inner_workers, dk_kwargs)
        for i in range(p) for j in range(i + 1, p)
    ]

    vals_list = parallel_execute(
        _hessian_component_worker,
        tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )
    y0 = np.asarray(function(theta))
    if y0.ndim == 0:
        hess = np.empty((1, p, p), dtype=float)
    else:
        out_shape = y0.shape
        o = int(out_shape[0])
        hess = np.empty((o, p, p), dtype=float)
    k = 0
    vals = np.array(vals_list, dtype=float)
    for i in range(p):
        hess[:,i, i] = vals[k]
        k += 1
    for i in range(p):
        for j in range(i + 1, p):
            hij = vals[k]
            k += 1
            hess[:,i, j] = hij
            hess[:,j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    if y0.ndim == 0:
        return hess[0,:,:]
    else:
        return hess


def _build_hessian_diag(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: np.ndarray,
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Returns the diagonal of the Hessian for a scalar- or vector-valued function.

    Args:
        function: The function to be differentiated.
        theta: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
        outer_workers: Number of outer parallel workers for diagonal entries.
        inner_workers: Optional inner parallelism for the differentiation engine.
        **dk_kwargs: Additional keyword arguments for
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A 1D array representing the diagonal of the Hessian.

    Raises:
        FloatingPointError: If non-finite values are encountered.
    """
    p = int(theta.size)

    tasks: list[Tuple[Any, ...]] = [(function, theta, i, i, method, inner_workers, dk_kwargs) for i in range(p)]
    vals = parallel_execute(
        _hessian_component_worker,
        tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    diag = np.asarray(vals, dtype=float)
    if not np.isfinite(diag).all():
        raise FloatingPointError("Non-finite values encountered in Hessian diagonal.")
    return diag


def _hessian_component_worker(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict,
) -> float:
    """Returns one entry of the Hessian for a scalar- or vector-valued function.

    Args:
        function: A function that returns a scalar or vector value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        inner_workers: Optional inner parallelism for the differentiation engine.
        dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A number or vector showing how the rate of change of the function
        depends on the parameters.
    """
    val = _hessian_component(
        function=function,
        theta0=theta0,
        i=i,
        j=j,
        method=method,
        n_workers=inner_workers or 1,
        **dk_kwargs,
    )
    val_arr = np.asarray(val, dtype=float)
    return val_arr


def _hessian_component(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> float:
    """Returns one entry of the Hessian for a scalar- or vector-valued function.

    This function measures how the rate of change in one parameter depends
    on another. It handles both the pure and mixed second derivatives:
      - If i == j, this is the second derivative with respect to a single parameter.
      - If i != j, this is the mixed derivative, computed by first finding
        how the function changes with parameter i while holding parameter j fixed,
        and then differentiating that result with respect to parameter j.

    Args:
        function: A function that returns a scalar or vector value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Optional inner parallelism for the differentiation engine.
        **dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A number or vector showing how the rate of change in one parameter
        depends on another.
    """
    # Mixed derivative path: define a helper that computes how the function changes with parameter i
    # when parameter j is temporarily set to a specific value.
    # Then we take the derivative of that helper with respect to parameter j.
    if i == j:
        partial_vec1 = get_partial_function(function, i, theta0)
        kit1 = DerivativeKit(partial_vec1, float(theta0[i]))
        return kit1.differentiate(order=2, method=method, n_workers=n_workers, **dk_kwargs)

    path = partial(
        _mixed_partial_value,
        function=function,
        theta0=theta0,
        i=i,
        j=j,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    kit2 = DerivativeKit(path, float(theta0[j]))
    return kit2.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)


def _mixed_partial_value(
    y: float,
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    i: int,
    j: int,
    method: str | None,
    n_workers: int | None,
    dk_kwargs: dict,
) -> float:
    """Returns the first derivative with respect to parameter i while temporarily setting parameter j to a given value.

    This helper does not compute the second derivative itself. It only returns
    the first derivative of the function with respect to one parameter while
    holding another fixed. The caller then takes the derivative of this result
    with respect to that fixed parameter to get the mixed second derivative.

    Args:
        y: The value to set for parameter j.
        function: A function that returns a scalar or vector value.
        theta0: The parameter values where the derivative is evaluated.
        i: Index of the first parameter.
        j: Index of the second parameter.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit` default
            (``"adaptive"``) is used.
        n_workers: Optional inner parallelism for the differentiation engine.
        dk_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        The value of the partial derivative with respect to parameter i
        when parameter j is set to y.
    """
    theta = theta0.copy()
    theta[j] = float(y)
    partial_vec1 = get_partial_function(function, i, theta)
    kit1 = DerivativeKit(partial_vec1, float(theta[i]))
    val = kit1.differentiate(
        order=1,
        method=method,
        n_workers=n_workers,
        **dk_kwargs,
    )
    val_arr = np.asarray(val, dtype=float)
    return val_arr


def _build_hessian_internal(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    *,
    method: str | None,
    n_workers: int,
    diag: bool,
    **dk_kwargs: Any,
) -> np.ndarray:
    """Core Hessian builder (internal).

    Computes either the full Hessian or only its diagonal at ``theta0``.
    Used internally by:

    - ``build_hessian(...)`` → ``diag=False`` (full)
    - ``build_hessian_diag(...)`` → ``diag=True`` (diagonal only)

    Args:
        function:
            Callable mapping parameters to a scalar or tensor.
        theta0:
            Parameter vector (1D array) at which the Hessian is evaluated.
        method:
            Derivative method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, uses the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``).
        n_workers:
            Number of outer parallel workers (across output components / Hessian entries).
            You may pass ``inner_workers=<int>`` in ``dk_kwargs`` to override inner parallelism.
        diag:
            If ``True``, compute only the diagonal entries.
            If ``False``, compute the full Hessian.
        **dk_kwargs:
            Additional keyword arguments forwarded to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        If ``function(theta0)`` is scalar:

        - ``diag=False``: array with shape ``(p, p)``  (full Hessian)
        - ``diag=True``: array with shape ``(p,)``    (diagonal only)

        If ``function(theta0)`` has shape ``out_shape``:

        - ``diag=False``: array with shape ``(*out_shape, p, p)``
        - ``diag=True``: array with shape ``(*out_shape, p)``

    Raises:
        FloatingPointError:
            If non-finite values are encountered.
        ValueError:
            If ``theta0`` is empty.
        TypeError:
            If evaluating a single output component does not return a scalar.

    Notes:
    - When ``diag=True``, mixed partials are skipped for speed and memory efficiency.
    - The inner worker count defaults to ``resolve_inner_from_outer(n_workers)`` unless
      explicitly overridden via ``inner_workers`` in ``dk_kwargs``.
    """
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    inner_override = dk_kwargs.pop("inner_workers", None)
    outer = int(n_workers) if n_workers is not None else 1
    inner = int(inner_override) if inner_override is not None else resolve_inner_from_outer(outer)

    if diag:
        arr = _build_hessian_diag(function, theta, method, outer, inner, **dk_kwargs)
        if not np.isfinite(arr).all():
            raise FloatingPointError("Non-finite values encountered in Hessian.")
        return arr
    else:
        arr = _build_hessian_full(function, theta, method, outer, inner, **dk_kwargs)
        if not np.isfinite(arr).all():
            raise FloatingPointError("Non-finite values encountered in Hessian.")
        return arr
