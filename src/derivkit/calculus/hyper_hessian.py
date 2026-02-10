"""Construct third-derivative tensors ("hyper-Hessians") for scalar- or vector-valued functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function
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

    This function computes all third-order partial derivatives of a scalar- or
    vector-valued function with respect to its parameters, evaluated at a single
    point in parameter space. The resulting tensor generalizes the Hessian to
    third order and is useful for higher-order Taylor expansions, non-Gaussian
    approximations, and sensitivity analyses beyond quadratic order.

    Args:
        function: Function to differentiate.
        theta0: 1D Parameter vector where the derivatives are evaluated.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        n_workers: Outer parallelism across output components (tensor outputs only).
        **dk_kwargs: Extra keyword args forwarded to :meth:`derivkit.DerivativeKit.differentiate`.
            You may pass ``inner_workers=<int>`` here to override inner parallelism.

    Returns:
        Third-derivative tensor. For scalar outputs, the result has shape
        ``(p, p, p)``, where ``p`` is the number of parameters. For tensor-valued
        outputs with shape ``out_shape``, the result has shape
        ``(*out_shape, p, p, p)``.

    Raises:
        ValueError: If ``theta0`` is empty.
        FloatingPointError: If non-finite values are encountered.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    probe = np.asarray(function(theta), dtype=np.float64)
    ensure_finite(probe, msg="Non-finite values in model output at theta0.")

    if probe.ndim not in (0, 1):
        raise TypeError(
            "Hyper-Hessian expects a scalar- or vector-valued function; "
            f"got output with shape {probe.shape}."
        )

    out_shape = probe.shape

    inner_override = dk_kwargs.pop("inner_workers", None)
    outer_workers = int(n_workers) if n_workers is not None else 1
    inner_workers = (
        int(inner_override)
        if inner_override is not None
        else resolve_inner_from_outer(outer_workers)
    )

    out = _build_hyper_hessian(
        function=function,
        theta=theta,
        out_shape=out_shape,
        method=method,
        inner_workers=inner_workers,
        outer_workers=outer_workers,
        **dk_kwargs,
    )

    return out


def _build_hyper_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    out_shape: tuple[int, ...],
    method: str | None,
    inner_workers: int | None,
    outer_workers: int,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar- or vector-valued function.

    Args:
        function: Scalar- or vector-valued function to differentiate.
        theta: 1D parameter vector where the derivatives are evaluated.
        out_shape: Shape of the output array.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        inner_workers: Number of inner workers for :class:`derivkit.DerivativeKit` calls.
        outer_workers: Number of outer workers for parallelism over entries.
        **dk_kwargs: Extra keyword args forwarded to :meth:`derivkit.DerivativeKit.differentiate`.

    Returns:
        The full hyper-Hessian array for the scalar- or vector-valued function.

    Raises:
        TypeError: If ``function`` does not return a scalar or a vector.
    """
    p = int(theta.size)
    iw = int(inner_workers or 1)

    # Compute only unique entries i<=j<=k, then symmetrize.
    triplets: list[tuple[int, int, int]] = [
        (i, j, k) for i in range(p) for j in range(i, p) for k in range(j, p)
    ]

    def entry_worker(i: int, j: int, k: int) -> np.ndarray:
        """Worker to compute one hyper-Hessian entry.

        Args:
            i: First parameter index.
            j: Second parameter index.
            k: Third parameter index.

        Returns:
            Value of the third order derivative of the function at theta0.
        """
        return _third_derivative_entry(
            function=function,
            theta0=theta,
            i=i,
            j=j,
            k=k,
            method=method,
            n_workers=iw,
            dk_kwargs=dk_kwargs,
        )

    vals = parallel_execute(
        entry_worker,
        arg_tuples=triplets,
        outer_workers=outer_workers,
        inner_workers=iw,
    )

    hess = np.empty((*out_shape, p, p, p), dtype=float)

    def _perm_indices(i: int, j: int, k: int) -> list[tuple[int, int, int]]:
        if i == j == k:
            return [(i, j, k)]
        if i == j != k:
            return [(i, i, k), (i, k, i), (k, i, i)]
        if i != j == k:
            return [(i, j, j), (j, i, j), (j, j, i)]
        return [
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ]

    for (i, j, k), v in zip(triplets, vals, strict=True):
        v = np.asarray(v, dtype=float)
        for a, b, c in _perm_indices(i, j, k):
            hess[..., a, b, c] = v

    ensure_finite(hess, msg="Non-finite values encountered in hyper-Hessian.")
    return hess


def _third_derivative_entry(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64],
    i: int,
    j: int,
    k: int,
    method: str | None,
    n_workers: int,
    dk_kwargs: dict[str, Any],
) -> np.ndarray:
    """Computes the third order derivative of ``function`` at ``theta0`` with respect to parameters ``i``, ``j``, ``k``.

    Args:
        function: Scalar- or vector-valued function to differentiate.
        theta0: 1D parameter vector at which the derivative is evaluated.
        i: First parameter index.
        j: Second parameter index.
        k: Third parameter index.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        n_workers: Number of workers for :class:`derivkit.DerivativeKit` calls.
        dk_kwargs: Extra keyword args forwarded to :meth:`derivkit.DerivativeKit.differentiate`.

    Returns:
        Value of the third order derivative of the function at ``theta0``.
    """
    i, j, k = int(i), int(j), int(k)
    ii, jj, kk = sorted((i, j, k))

    if ii == jj == kk:
        f1 = get_partial_function(function, ii, theta0)
        kit = DerivativeKit(f1, float(theta0[ii]))
        val = kit.differentiate(order=3, method=method, n_workers=n_workers, **dk_kwargs)
        return np.asarray(val, dtype=float)

    def g_func(t: float) -> float:
        """Function for derivative with respect to parameter kk.

        Args:
            t: Value of parameter kk.

        Returns:
            Second derivative with respect to ii and jj at theta0 with kk=t.

        """
        th = theta0.copy()
        th[kk] = float(t)

        if ii == jj:
            f1 = get_partial_function(function, ii, th)
            kit2 = DerivativeKit(f1, float(th[ii]))
            v2 = kit2.differentiate(order=2, method=method, n_workers=n_workers, **dk_kwargs)
            return np.asarray(v2, dtype=float)

        # mixed second derivative via nested 1D partials
        def h_func(yj: float) -> float:
            """Function for derivative with respect to parameter jj.

            Args:
                yj: Value of parameter jj.

            Returns:
                First derivative with respect to ii at theta0 with jj=yj and kk fixed.
            """
            th2 = th.copy()
            th2[jj] = float(yj)
            f_ii = get_partial_function(function, ii, th2)
            kit1 = DerivativeKit(f_ii, float(th2[ii]))
            v1 = kit1.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)
            return np.asarray(v1, dtype=float)

        kitm = DerivativeKit(h_func, float(th[jj]))
        vm = kitm.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)
        return np.asarray(vm, dtype=float)

    kit3 = DerivativeKit(g_func, float(theta0[kk]))
    v3 = kit3.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)
    return np.asarray(v3, dtype=float)
