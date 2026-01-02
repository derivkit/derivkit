"""Construct third-derivative tensors ("hyper-Hessians") for scalar- or tensor-valued functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.calculus.calculus_core import (
    component_scalar_eval,
    dispatch_tensor_output,
)
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

    For scalar outputs, returns an array with shape ``(p, p, p)`` with ``p``
    the number of parameters.
    For tensor outputs with shape ``out_shape``, returns ``(*out_shape, p, p, p)``.

    Args:
        function: Function to differentiate.
        theta0: 1D Parameter vector where the derivatives are evaluated.
        method: Derivative method name or alias. If ``None``, DerivativeKit default is used.
        n_workers: Outer parallelism across output components (tensor outputs only).
        **dk_kwargs: Extra keyword args forwarded to :meth:`DerivativeKit.differentiate`.
            You may pass ``inner_workers=<int>`` here to override inner parallelism.

    Returns:
        Third-derivative tensor as described above.

    Raises:
        ValueError: If ``theta0`` is empty.
        FloatingPointError: If non-finite values are encountered.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    y0 = np.asarray(function(theta))
    ensure_finite(y0, msg="Non-finite values in model output at theta0.")

    inner_override = dk_kwargs.pop("inner_workers", None)
    outer_workers = int(n_workers) if n_workers is not None else 1
    inner_workers = (
        int(inner_override)
        if inner_override is not None
        else resolve_inner_from_outer(outer_workers)
    )

    if y0.ndim == 0:
        out = _build_hyper_hessian_scalar(
            function=function,
            theta=theta,
            method=method,
            inner_workers=inner_workers,
            outer_workers=outer_workers,
            **dk_kwargs,
        )

        ensure_finite(out, msg="Non-finite values encountered in hyper-Hessian.")
        return out

    # Tensor output: compute per-component scalar hyper-Hessian and reshape it back.
    return dispatch_tensor_output(
        function=function,
        theta=theta,
        method=method,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
        dk_kwargs=dk_kwargs,
        build_component=_compute_component_hyper_hessian,
    )


def _build_hyper_hessian_scalar(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    outer_workers: int,
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar-valued function.

    Args:
        function: Scalar-valued function to differentiate.
        theta: Parameter vector (1D).
        method: Derivative method name or alias.
        inner_workers: Number of inner workers for DerivativeKit calls.
        outer_workers: Number of outer workers for parallelism over entries.
        **dk_kwargs: Extra keyword args forwarded to :meth:`DerivativeKit.differentiate`.

    Returns:
        Hyper-Hessian array with shape ``(p, p, p)`` with ``p`` the number of parameters.

    Raises:
        TypeError: If ``function`` does not return a scalar.
    """
    probe = np.asarray(function(theta), dtype=np.float64)
    if probe.ndim != 0:
        raise TypeError(
            "Scalar hyper-Hessian path expects a scalar-valued function; "
            f"got output with shape {probe.shape}."
        )

    p = int(theta.size)
    iw = int(inner_workers or 1)

    # Compute only unique entries i<=j<=k, then symmetrize.
    triplets: list[tuple[int, int, int]] = [
        (i, j, k) for i in range(p) for j in range(i, p) for k in range(j, p)
    ]

    def entry_worker(i: int, j: int, k: int) -> float:
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

    hess = np.zeros((p, p, p), dtype=np.float64)
    for (i, j, k), v in zip(triplets, vals, strict=True):
        v = float(np.asarray(v).item())
        for a, b, c in set(permutations((i, j, k), 3)):
            hess[a, b, c] = v

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
) -> float:
    """Computes the third order derivative of `function` at `theta0` with respect to parameters i, j, k.

    Args:
        function: Scalar-valued function to differentiate.
        theta0: Parameter vector (1D).
        i: First parameter index.
        j: Second parameter index.
        k: Third parameter index.
        method: Derivative method name or alias.
        n_workers: Number of workers for DerivativeKit calls.
        dk_kwargs: Extra keyword args forwarded to :meth:`DerivativeKit.differentiate`.

    Returns:
        Value of the third order derivative of the function at theta0.
    """
    i, j, k = int(i), int(j), int(k)
    ii, jj, kk = sorted((i, j, k))

    if ii == jj == kk:
        f1 = get_partial_function(function, ii, theta0)
        kit = DerivativeKit(f1, float(theta0[ii]))
        val = kit.differentiate(order=3, method=method, n_workers=n_workers, **dk_kwargs)
        return float(np.asarray(val).item())

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
            return float(np.asarray(v2).item())

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
            return float(np.asarray(v1).item())

        kitm = DerivativeKit(h_func, float(th[jj]))
        vm = kitm.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)
        return float(np.asarray(vm).item())

    kit3 = DerivativeKit(g_func, float(theta0[kk]))
    v3 = kit3.differentiate(order=1, method=method, n_workers=n_workers, **dk_kwargs)
    return float(np.asarray(v3).item())


def _compute_component_hyper_hessian(
    idx: int,
    theta: NDArray[np.float64],
    method: str | None,
    inner_workers: int | None,
    dk_kwargs: dict[str, Any],
    function: Callable[[ArrayLike], float | np.ndarray],
) -> NDArray[np.float64]:
    """Computes the hyper-Hessian for one output component of a tensor-valued function.

    Args:
        idx: Index of the output component to differentiate.
        theta: Parameter vector (1D).
        method: Derivative method name or alias.
        inner_workers: Number of inner workers for DerivativeKit calls.
        dk_kwargs: Extra keyword args forwarded to :meth:`DerivativeKit.differentiate`.
        function: Original tensor-valued function.

    Returns:
        Hyper-Hessian array with shape ``(p, p, p)`` with ``p`` the number of parameters.
    """
    g = partial(component_scalar_eval, function=function, idx=int(idx))
    # Use scalar builder with outer_workers=1 (parallel computation is already over components outside)
    return _build_hyper_hessian_scalar(
        g,
        theta,
        method=method,
        inner_workers=inner_workers,
        outer_workers=1,
        **dk_kwargs,
    )
