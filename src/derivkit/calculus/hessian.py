"""Build Hessians for scalar- or tensor-valued functions.

This implementation is backend-agnostic with respect to DerivKit engines and
avoids taking "derivatives of derivatives" for mixed partials.

Core idea
---------

For a scalar function f(theta) with Hessian H, the second directional derivative
along a direction vector v is::

    d^2/dt^2 f(theta + t v) at t=0  =  v^T H v

Using the directions v_plus = e_i + e_j and v_minus = e_i - e_j::

    v_plus^T H v_plus   =  H_ii + 2 H_ij + H_jj
    v_minus^T H v_minus =  H_ii - 2 H_ij + H_jj

Therefore::

    H_ij = ( (v_plus^T H v_plus) - (v_minus^T H v_minus) ) / 4

We compute:

- Diagonal terms H_ii via an order-2 derivative of ``t -> f(theta + t e_i)``.
- Off-diagonal terms H_ij via two order-2 derivatives of::

      t -> f(theta + t (e_i + e_j))
      t -> f(theta + t (e_i - e_j))

This is:

- backend-agnostic: uses whatever DerivKit engine you select
- direct: no nested differentiation
- cache-friendly: repeated evaluations at identical ``theta`` can be reused

Tensor-valued outputs
---------------------

If ``f(theta)`` returns an array, we flatten it and compute one scalar Hessian
per output component. To avoid recomputing the forward model for identical
parameter displacements, the forward model is wrapped with the theta-caching
helper before taking derivatives.

Parallelism
-----------

- Outer workers parallelize across output components (tensor case) or across
  Hessian entries (scalar case).
- Inner workers are forwarded to DerivKit for its internal parallelism.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus.calculus_core import cache_theta_function
from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)

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
        method: Method name or alias forwarded to DerivativeKit (e.g. "adaptive",
            "finite", "fornberg", "local_polynomial", etc.). If None, uses the
            DerivativeKit default.
        n_workers: Parallel tasks across output components / Hessian entries.
        **dk_kwargs: Extra options forwarded to DerivativeKit.differentiate.

    Returns:
        Always returns the full Hessian with shape:

        - ``(p, p)`` if function(theta0) is scalar with p parameters.
        - ``(*out_shape, p, p)`` if function(theta0) has shape out_shape.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If theta0 is empty.
        TypeError: If a flattened scalar component path does not return a scalar.
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
) -> NDArray[np.floating]:
    """Returns the diagonal of the Hessian of a function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the Hessian is evaluated.
        method: Method name or alias forwarded to DerivativeKit.
        n_workers: Parallel tasks across output components / diagonal entries.
        **dk_kwargs: Extra options forwarded to DerivativeKit.differentiate.

            You may optionally pass inner_workers=<int> here.

    Returns:
        Returns only the diagonal entries:

        - ``(p,)`` if function(theta0) is scalar with ``p`` parameters.
        - ``(*out_shape, p)`` if function(theta0) has shape out_shape.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If theta0 is empty.
        TypeError: If evaluating a single output component does not return a scalar.
    """
    return _build_hessian_internal(
        function, theta0, method=method, n_workers=n_workers, diag=True, **dk_kwargs
    )


def gauss_newton_hessian(*args: Any, **kwargs: Any) -> None:
    """Placeholder for a Gauss-Newton Hessian computation function."""
    _, _ = args, kwargs
    raise NotImplementedError


def _component_scalar_eval(
    theta_vec: NDArray[np.floating],
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    idx: int,
) -> float:
    """Returns one scalar component from a (possibly tensor) function output."""
    th = np.asarray(theta_vec, dtype=np.float64).reshape(-1)
    val = np.asarray(function(th), dtype=float)
    return float(val.ravel()[idx])


def _directional_second_derivative(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.floating],
    direction: NDArray[np.floating],
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> float:
    """Compute d^2/dt^2 f(theta0 + t*direction) at t=0 using DerivativeKit.

    Args:
        function: Scalar-valued function of theta.
        theta0: Base point (p,).
        direction: Direction vector (p,). Not required to be unit length.
        method: DerivativeKit method name/alias.
        n_workers: Passed to DerivativeKit.differentiate.
        dk_kwargs: Passed through to DerivativeKit.differentiate.

    Returns:
        Second derivative w.r.t. t at t=0 as a float.

    Raises:
        TypeError: If function(theta) is not scalar along this scalar wrapper path.
    """
    th0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    v = np.asarray(direction, dtype=np.float64).reshape(-1)

    def g(t: float) -> float:
        th = th0 + float(t) * v
        y = np.asarray(function(th), dtype=float)
        if y.size != 1:
            raise TypeError("Directional wrapper expects a scalar-valued function.")
        return float(y.item())

    kit = DerivativeKit(g, x0=0.0)
    val = kit.differentiate(order=2, method=method, n_workers=n_workers, **dk_kwargs)
    arr = np.asarray(val, dtype=float)
    if arr.size != 1:
        raise TypeError(f"Expected scalar second derivative; got shape {arr.shape}.")
    return float(arr.item())


def _mixed_from_directionals(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.floating],
    i: int,
    j: int,
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> float:
    """Compute H_ij via directional second derivatives, avoiding nested derivatives.

    Uses:
        H_ij = (D2(e_i + e_j) - D2(e_i - e_j)) / 4
    where D2(v) = d^2/dt^2 f(theta + t v) at t=0.
    """
    p = int(theta0.size)
    e_i = np.zeros(p, dtype=float)
    e_j = np.zeros(p, dtype=float)
    e_i[i] = 1.0
    e_j[j] = 1.0

    v_plus = e_i + e_j
    v_minus = e_i - e_j

    d2_plus = _directional_second_derivative(
        function=function,
        theta0=theta0,
        direction=v_plus,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    d2_minus = _directional_second_derivative(
        function=function,
        theta0=theta0,
        direction=v_minus,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    return 0.25 * (d2_plus - d2_minus)


def _build_hessian_scalar_diag(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.floating],
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Returns the diagonal of the Hessian for a scalar-valued function."""
    p = int(theta.size)
    inner = int(inner_workers) if inner_workers is not None else 1
    dk_local = dict(dk_kwargs)

    def _diag_worker(i: int) -> float:
        e_i = np.zeros(p, dtype=float)
        e_i[int(i)] = 1.0
        return _directional_second_derivative(
            function=function,
            theta0=theta,
            direction=e_i,
            method=method,
            n_workers=inner,
            dk_kwargs=dict(dk_local),
        )

    vals = parallel_execute(
        _diag_worker,
        [(i,) for i in range(p)],
        outer_workers=outer_workers,
        inner_workers=inner,
    )

    diag = np.asarray(vals, dtype=float)
    if not np.isfinite(diag).all():
        raise FloatingPointError("Non-finite values encountered in Hessian diagonal.")
    return diag


def _build_hessian_scalar_full(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.floating],
    method: str | None,
    outer_workers: int,
    inner_workers: int | None,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Returns the full (p, p) Hessian for a scalar-valued function."""
    p = int(theta.size)
    inner = int(inner_workers) if inner_workers is not None else 1
    dk_local = dict(dk_kwargs)

    # Diagonal first (needed anyway; also useful for debugging/stability).
    diag = _build_hessian_scalar_diag(
        function, theta, method, outer_workers=outer_workers, inner_workers=inner, **dk_local
    )

    dk_local = dict(dk_kwargs)

    def _offdiag_worker(i: int, j: int) -> tuple[int, int, float]:
        hij = _mixed_from_directionals(
            function=function,
            theta0=theta,
            i=int(i),
            j=int(j),
            method=method,
            n_workers=inner,
            dk_kwargs=dict(dk_local),
        )
        return int(i), int(j), float(hij)

    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    vals = parallel_execute(
        _offdiag_worker,
        pairs,
        outer_workers=outer_workers,
        inner_workers=inner,
    )

    hess = np.empty((p, p), dtype=float)
    hess[np.diag_indices(p)] = diag
    for i, j, hij in vals:
        hess[i, j] = hij
        hess[j, i] = hij

    if not np.isfinite(hess).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    return hess


def _compute_component_hessian(
    idx: int,
    theta: NDArray[np.floating],
    method: str | None,
    inner_workers: int | None,
    return_diag: bool,
    dk_kwargs: dict[str, Any],
    function: Callable[[ArrayLike], float | np.ndarray],
) -> NDArray[np.floating]:
    """Compute Hessian (or diagonal) for one output component using cached function."""
    g = partial(_component_scalar_eval, function=function, idx=int(idx))

    if return_diag:
        return _build_hessian_scalar_diag(g, theta, method, 1, inner_workers, **dk_kwargs)
    return _build_hessian_scalar_full(g, theta, method, 1, inner_workers, **dk_kwargs)


def _build_hessian_internal(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: np.ndarray,
    *,
    method: str | None,
    n_workers: int,
    diag: bool,
    **dk_kwargs: Any,
) -> NDArray[np.floating]:
    """Core Hessian builder (internal).

    Shapes of the returned array:

    - Scalar output (function(theta0) is a scalar):
      - diag=False -> ``(p, p)``
      - diag=True  -> ``(p,)``

    - Tensor output (function(theta0) has shape ``out_shape``):
      - diag=False -> ``out_shape + (p, p)``
      - diag=True  -> ``out_shape + (p,)``
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # Never mutate caller kwargs (and don't share dicts across tasks).
    dk_kwargs_local = dict(dk_kwargs)

    inner_override = dk_kwargs_local.pop("inner_workers", None)
    outer = int(n_workers) if n_workers is not None else 1
    inner = int(inner_override) if inner_override is not None else resolve_inner_from_outer(outer)

    # Cache full forward-model evaluations at identical theta.
    function_cached = cache_theta_function(function)

    y0 = np.asarray(function_cached(theta), dtype=float)
    out_shape = y0.shape

    # Scalar output.
    if y0.ndim == 0:
        if diag:
            return _build_hessian_scalar_diag(
                function_cached, theta, method, outer, inner, **dk_kwargs_local
            )
        return _build_hessian_scalar_full(
            function_cached, theta, method, outer, inner, **dk_kwargs_local
        )

    # Tensor output: flatten and compute per-component Hessians.
    m = int(y0.size)
    tasks = [
        (i, theta, method, inner, diag, dict(dk_kwargs_local), function_cached)
        for i in range(m)
    ]
    vals = parallel_execute(
        _compute_component_hessian,
        tasks,
        outer_workers=outer,
        inner_workers=inner,
    )

    arr = np.stack(vals, axis=0)
    arr = arr.reshape(out_shape + arr.shape[1:])
    if not np.isfinite(arr).all():
        raise FloatingPointError("Non-finite values encountered in Hessian.")
    return arr
