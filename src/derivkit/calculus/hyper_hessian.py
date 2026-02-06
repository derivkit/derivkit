"""Construct third-derivative tensors ("hyper-Hessians") for scalar- or tensor-valued functions.

This implementation is backend-agnostic with respect to DerivKit engines and
avoids taking derivatives of derivatives for mixed partials.

Core idea
---------

For a scalar function f(theta) with a symmetric third-derivative tensor T, the
third directional derivative along a direction vector v is::

    D3(v) = d^3/dt^3 f(theta + t v) evaluated at t = 0

This directional derivative can be written as a weighted sum over the tensor
entries T[a, b, c] multiplied by the components of v.

We recover the unique tensor entries using a small set of directional identities.

Cases
-----

1) Pure terms (all indices equal)

For index i::

    T[i, i, i] = D3(e_i)

2) Two indices equal (i, i, k with i != k)

Define::

    A  = D3(e_i)
    B  = D3(e_k)
    C+ = D3(e_i + e_k)
    C- = D3(e_i - e_k)

Then::

    T[i, i, k] = (C+ - C- - 2 B) / 6
    T[i, k, k] = (C+ + C- - 2 A) / 6

When storing only sorted indices (i <= j <= k), the (i, i, k) case uses
T[i, i, k].

3) All indices distinct (i < j < k)

Define::

    S1 = D3(e_i + e_j + e_k)
    S2 = D3(e_i + e_j - e_k)
    S3 = D3(e_i - e_j + e_k)
    S4 = D3(e_j + e_k - e_i)

Then::

    T[i, j, k] = (S1 - S2 - S3 - S4) / 12

All required quantities are obtained from one-dimensional third-derivative
calls through DerivKit on functions of the form::

    t -> f(theta + t v)

Any DerivKit backend that supports third-order derivatives can be used.

Tensor-valued outputs
---------------------

If ``f(theta)`` returns an array, the output is flattened and one scalar
hyper-Hessian is computed per component. The full forward model is wrapped
with the theta-caching helper so evaluations at identical parameter values
are reused across components.

Parallelism
-----------

- Outer workers parallelize across output components (tensor case) or across
  unique index triplets (scalar case).
- Inner workers are forwarded to DerivKit for its internal parallelism.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus.calculus_core import (
    cache_theta_function,
    component_scalar_eval,
)
from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.validate import ensure_finite

__all__ = [
    "build_hyper_hessian",
]


def build_hyper_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64] | Sequence[float],
    *,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns the third-derivative tensor ("hyper-Hessian") of a function.

    Args:
        function: Function to differentiate.
        theta0: 1D parameter vector where the derivatives are evaluated.
        method: Derivative method name or alias forwarded to DerivativeKit. If
            None, DerivativeKit's default is used.
        n_workers: Outer parallelism across output components (tensor outputs) or
            across unique triplets (scalar outputs).
        **dk_kwargs: Extra keyword args forwarded to DerivativeKit.differentiate.
            You may pass inner_workers=<int> here to override inner parallelism.

    Returns:
        Third-derivative tensor.

        - Scalar output: shape ``(p, p, p)`` with ``p = theta0.size`` parameters
        - Tensor output with shape out_shape: shape ``(*out_shape, p, p, p)``

    Raises:
        ValueError: If theta0 is empty.
        FloatingPointError: If non-finite values are encountered.
        TypeError: If a scalar path does not return a scalar.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # Never mutate caller kwargs (and don't share dicts across tasks).
    dk_kwargs_local: dict[str, Any] = dict(dk_kwargs)

    inner_override = dk_kwargs_local.pop("inner_workers", None)
    outer_workers = int(n_workers) if n_workers is not None else 1
    inner_workers = (
        int(inner_override)
        if inner_override is not None
        else resolve_inner_from_outer(outer_workers)
    )

    # Cache full forward-model evaluations at identical theta.
    function_cached = cache_theta_function(function)

    y0 = np.asarray(function_cached(theta), dtype=np.float64)
    ensure_finite(y0, msg="Non-finite values in model output at theta0.")

    if y0.ndim == 0:
        out = _build_hyper_hessian_scalar(
            function=function_cached,
            theta=theta,
            method=method,
            inner_workers=inner_workers,
            outer_workers=outer_workers,
            **dk_kwargs_local,
        )
        ensure_finite(out, msg="Non-finite values encountered in hyper-Hessian.")
        return out

    # Tensor output: compute per-component scalar hyper-Hessian, then reshape.
    out_shape = y0.shape
    m = int(y0.size)

    tasks = [
        (idx, theta, method, inner_workers, dict(dk_kwargs_local), function_cached)
        for idx in range(m)
    ]
    vals = parallel_execute(
        _compute_component_hyper_hessian,
        tasks,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    arr0 = np.asarray(vals[0])
    arr = np.stack(vals, axis=0).reshape(out_shape + arr0.shape)

    ensure_finite(arr, msg="Non-finite values encountered in hyper-Hessian.")
    return arr


def _directional_third_derivative(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64],
    direction: NDArray[np.float64],
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> float:
    """Compute D3(direction) = d^3/dt^3 f(theta0 + t*direction) at t=0.

    Args:
        function: Scalar-valued function of theta.
        theta0: Base point (p,).
        direction: Direction vector (p,).
        method: DerivativeKit method name/alias.
        n_workers: Passed to DerivativeKit.differentiate.
        dk_kwargs: Passed through to DerivativeKit.differentiate.

    Returns:
        Third derivative w.r.t. t at t=0 as a float.

    Raises:
        TypeError: If the wrapped path is not scalar-valued.
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
    val = kit.differentiate(order=3, method=method, n_workers=n_workers, **dk_kwargs)
    arr = np.asarray(val, dtype=float)
    if arr.size != 1:
        raise TypeError(f"Expected scalar third derivative; got shape {arr.shape}.")
    return float(arr.item())


def _third_entry_from_directionals(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64],
    i: int,
    j: int,
    k: int,
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> float:
    """Compute T[i, j, k] from directional third derivatives."""
    i, j, k = int(i), int(j), int(k)
    p = int(theta0.size)

    if i == j == k:
        e = np.zeros(p, dtype=float)
        e[i] = 1.0
        return _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )

    # Two equal: (i, i, k) with i<k (sorted i<=j<=k).
    if i == j and j != k:
        e_i = np.zeros(p, dtype=float)
        e_k = np.zeros(p, dtype=float)
        e_i[i] = 1.0
        e_k[k] = 1.0

        # Only B is needed for T_{iik}.
        B = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_k,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        C_plus = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_i + e_k,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        C_minus = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_i - e_k,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        return (C_plus - C_minus - 2.0 * B) / 6.0

    # Two equal: (i, k, k) occurs as sorted (i<j==k).
    if i != j and j == k:
        e_i = np.zeros(p, dtype=float)
        e_k = np.zeros(p, dtype=float)
        e_i[i] = 1.0
        e_k[k] = 1.0

        # Only A is needed for T_{ikk}.
        A = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_i,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        C_plus = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_i + e_k,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        C_minus = _directional_third_derivative(
            function=function,
            theta0=theta0,
            direction=e_i - e_k,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )
        return (C_plus + C_minus - 2.0 * A) / 6.0

    # All distinct: i<j<k.
    e_i = np.zeros(p, dtype=float)
    e_j = np.zeros(p, dtype=float)
    e_k = np.zeros(p, dtype=float)
    e_i[i] = 1.0
    e_j[j] = 1.0
    e_k[k] = 1.0

    S1 = _directional_third_derivative(
        function=function,
        theta0=theta0,
        direction=e_i + e_j + e_k,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    S2 = _directional_third_derivative(
        function=function,
        theta0=theta0,
        direction=e_i + e_j - e_k,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    S3 = _directional_third_derivative(
        function=function,
        theta0=theta0,
        direction=e_i - e_j + e_k,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    S4 = _directional_third_derivative(
        function=function,
        theta0=theta0,
        direction=-e_i + e_j + e_k,
        method=method,
        n_workers=n_workers,
        dk_kwargs=dk_kwargs,
    )
    return (S1 - S2 - S3 - S4) / 12.0


def _build_hyper_hessian_scalar(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    outer_workers: int,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar-valued function.

    Computes only unique entries i<=j<=k via directional identities and then
    symmetrizes to fill the full (p, p, p) tensor.
    """
    probe = np.asarray(function(theta), dtype=np.float64)
    if probe.ndim != 0:
        raise TypeError(
            "Scalar hyper-Hessian path expects a scalar-valued function; "
            f"got output with shape {probe.shape}."
        )

    p = int(theta.size)
    iw = int(inner_workers or 1)

    # IMPORTANT: treat kwargs as read-only
    dk_local: dict[str, Any] = dict(dk_kwargs)

    # Cache directional third derivatives D3(v) within this build.
    # Key as a tuple of ints because our directions are integer combos of basis
    # vectors (-1, 0, +1). This massively reduces repeated DerivativeKit calls.
    d3_cache: dict[tuple[int, ...], float] = {}

    def d3(direction: NDArray[np.float64]) -> float:
        key = tuple(int(x) for x in np.asarray(direction, dtype=int).ravel())
        try:
            return d3_cache[key]
        except KeyError:
            val = _directional_third_derivative(
                function=function,
                theta0=theta,
                direction=np.asarray(direction, dtype=np.float64),
                method=method,
                n_workers=iw,
                dk_kwargs=dict(dk_local),  # defensive: ensure no mutation leaks
            )
            d3_cache[key] = float(val)
            return float(val)

    def entry_value(i: int, j: int, k: int) -> float:
        """Compute T[i, j, k] for sorted indices i<=j<=k."""
        i, j, k = int(i), int(j), int(k)

        if i == j == k:
            e = np.zeros(p, dtype=float)
            e[i] = 1.0
            return d3(e)

        if i == j and j != k:
            e_i = np.zeros(p, dtype=float)
            e_k = np.zeros(p, dtype=float)
            e_i[i] = 1.0
            e_k[k] = 1.0
            B = d3(e_k)
            C_plus = d3(e_i + e_k)
            C_minus = d3(e_i - e_k)
            return (C_plus - C_minus - 2.0 * B) / 6.0

        if i != j and j == k:
            e_i = np.zeros(p, dtype=float)
            e_k = np.zeros(p, dtype=float)
            e_i[i] = 1.0
            e_k[k] = 1.0
            A = d3(e_i)
            C_plus = d3(e_i + e_k)
            C_minus = d3(e_i - e_k)
            return (C_plus + C_minus - 2.0 * A) / 6.0

        # all distinct (i<j<k)
        e_i = np.zeros(p, dtype=float)
        e_j = np.zeros(p, dtype=float)
        e_k = np.zeros(p, dtype=float)
        e_i[i] = 1.0
        e_j[j] = 1.0
        e_k[k] = 1.0

        S1 = d3(e_i + e_j + e_k)
        S2 = d3(e_i + e_j - e_k)
        S3 = d3(e_i - e_j + e_k)
        S4 = d3(-e_i + e_j + e_k)
        return (S1 - S2 - S3 - S4) / 12.0

    triplets: list[tuple[int, int, int]] = [
        (i, j, k) for i in range(p) for j in range(i, p) for k in range(j, p)
    ]

    # Make directional cache effective (shared within this call).
    # For scalar outputs, parallelism here defeats caching because each worker
    # has its own local d3_cache.
    outer_workers = 1

    vals = parallel_execute(
        entry_value,
        arg_tuples=triplets,
        outer_workers=outer_workers,
        inner_workers=iw,
    )

    tens = np.zeros((p, p, p), dtype=np.float64)
    for (i, j, k), v in zip(triplets, vals, strict=True):
        vv = float(np.asarray(v, dtype=float).item())
        for a, b, c in set(permutations((i, j, k), 3)):
            tens[a, b, c] = vv

    ensure_finite(tens, msg="Non-finite values encountered in hyper-Hessian.")
    return tens


def _compute_component_hyper_hessian(
    idx: int,
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict[str, Any],
    function: Callable[[ArrayLike], float | np.ndarray],
) -> NDArray[np.float64]:
    """Computes the hyper-Hessian for one output component of a tensor-valued function."""
    g = partial(component_scalar_eval, function=function, idx=int(idx))
    # Parallelism over components happens outside, so keep scalar builder outer_workers=1 here.
    return _build_hyper_hessian_scalar(
        function=g,
        theta=theta,
        method=method,
        inner_workers=inner_workers,
        outer_workers=1,
        **dk_kwargs,
    )
