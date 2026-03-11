"""Construct third-derivative tensors ("hyper-Hessians") for scalar- or vector-valued functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from itertools import permutations
from typing import Any

import dask
import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
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
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    fun_check_eval = np.atleast_1d(function(theta))
    ensure_finite(fun_check_eval, msg="Non-finite values in model output at theta0.")

    if fun_check_eval.ndim not in [0, 1]:
        raise TypeError(
            "Hyper-Hessian expects a scalar- or vector-valued function; "
            f"got output with shape {fun_check_eval.shape}."
        )

    hyper_hessian = _build_hyper_hessian(
        function=function,
        theta=theta,
        method=method,
        fun_out_shape=fun_check_eval.shape,
        **dk_kwargs,
    )

    return hyper_hessian


def _build_hyper_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    method: str | None,
    fun_out_shape: tuple[int, ...],
    **dk_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar- or vector-valued function.

    Args:
        function: Scalar- or vector-valued function to differentiate.
        theta: 1D parameter vector where the derivatives are evaluated.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        **dk_kwargs: Extra keyword args forwarded to :meth:`derivkit.DerivativeKit.differentiate`.
            Pass ``inner_workers=<int>`` to control parallelism inside each
            derivative evaluation.

    Returns:
        The full hyper-Hessian array for the scalar- or vector-valued function.

    Raises:
        TypeError: If ``function`` does not return a scalar or a vector.
    """
    p = theta.size

    # Compute only unique entries i<=j<=k, then symmetrize.
    triplets: list[tuple[int, int, int]] = [
        (i, j, k) for i in range(p) for j in range(i, p) for k in range(j, p)
    ]

    worker = partial(
        _third_derivative_entry,
        function=function,
        theta0=theta,
        method=method,
        **dk_kwargs,
    )

    vals_list = [worker(i=i, j=j, k=k) for (i, j, k) in triplets]
    hyper_hess = dask.delayed(_fill_hyper_hessian)(vals_list, triplets, (*fun_out_shape, p, p, p))

    return hyper_hess


def _fill_hyper_hessian(vals_list, triplets, shape):
    hyper_hess = np.zeros(shape, dtype=float)

    for (i, j, k), v in zip(triplets, vals_list, strict=True):
        for idx in set(permutations((i, j, k))):
            hyper_hess[..., *idx] = v

    ensure_finite(hyper_hess, msg="Non-finite values encountered in hyper-Hessian.")

    return hyper_hess


def _third_derivative_entry(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64],
    i: int,
    j: int,
    k: int,
    method: str | None,
    **dk_kwargs: dict[str, Any],
) -> float:
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
    ii, jj, kk = sorted((i, j, k))

    if ii == jj == kk:
        f1 = get_partial_function(function, ii, theta0)
        kit = DerivativeKit(f1, float(theta0[ii]))
        return kit.differentiate(order=3, method=method, **dk_kwargs)

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
            return kit2.differentiate(order=2, method=method, delayed_fun=True, **dk_kwargs)

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
            return kit1.differentiate(order=1, method=method, **dk_kwargs)

        kit2 = DerivativeKit(h_func, float(th[jj]))
        return kit2.differentiate(order=1, method=method, delayed_fun=True, **dk_kwargs)

    kit3 = DerivativeKit(g_func, float(theta0[kk]))
    return kit3.differentiate(order=1, method=method, delayed_fun=True, **dk_kwargs)
