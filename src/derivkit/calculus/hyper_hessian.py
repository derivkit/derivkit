"""Construct derivative tensors ("hyper-Hessians") for scalar- or vector-valued functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations_with_replacement, permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.caching import wrap_input_cache
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
    order: int = 3,
    method: str | None = None,
    n_workers: int = 1,
    dk_init_kwargs: dict[str, Any] | None = None,
    **dk_diff_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a derivative tensor ("hyper-Hessian") of a function.

    This function computes all partial derivatives of the requested order of a
    scalar- or vector-valued function with respect to its parameters, evaluated at
    a single point in parameter space. The resulting tensor is useful for
    higher-order Taylor expansions, non-Gaussian approximations, and sensitivity
    analyses beyond quadratic order.

    Args:
        function: Function to differentiate.
        theta0: 1D Parameter vector where the derivatives are evaluated.
        order: Derivative order. An order of zero returns the function value
            evaluated at ``theta0``.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        n_workers: Outer parallelism across output components (tensor outputs only).
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        **dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Function value or derivative tensor evaluated at ``theta0``. For
        ``order=0``, the function value itself is returned. For positive
        derivative orders, ``order`` parameter axes are appended to the
        function output shape.

    Raises:
        ValueError: If ``theta0`` is empty or ``order`` is negative.
        TypeError: If ``function`` does not return a scalar or a vector.
        FloatingPointError: If non-finite values are encountered.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    if order < 0:
        raise ValueError("Derivative order must be non-negative.")

    probe = np.asarray(function(theta), dtype=np.float64)
    ensure_finite(probe, msg="Non-finite values in model output at theta0.")

    if probe.ndim not in (0, 1):
        raise TypeError(
            "Hyper-Hessian expects a scalar- or vector-valued function; "
            f"got output with shape {probe.shape}."
        )

    if order == 0:
        return probe

    out_shape = probe.shape

    dk_init_kwargs = dict(dk_init_kwargs or {})

    use_input_cache = dk_init_kwargs.pop("use_input_cache", True)
    cache_number_decimal_places = dk_init_kwargs.pop(
        "cache_number_decimal_places",
        None,
    )
    cache_maxsize = dk_init_kwargs.pop("cache_maxsize", 4096)
    cache_copy = dk_init_kwargs.pop("cache_copy", True)

    inner_override = dk_diff_kwargs.pop("inner_workers", None)
    outer_workers = int(n_workers) if n_workers is not None else 1
    inner_workers = (
        int(inner_override)
        if inner_override is not None
        else resolve_inner_from_outer(outer_workers)
    )

    shared_function = (
        wrap_input_cache(
            function,
            number_decimal_places=cache_number_decimal_places,
            maxsize=cache_maxsize,
            copy=cache_copy,
        )
        if use_input_cache
        else function
    )

    out = _build_hyper_hessian(
        function=shared_function,
        theta=theta,
        out_shape=out_shape,
        order=order,
        method=method,
        inner_workers=inner_workers,
        outer_workers=outer_workers,
        dk_init_kwargs=dk_init_kwargs,
        **dk_diff_kwargs,
    )

    return out


def _build_hyper_hessian(
    function: Callable[[ArrayLike], float | np.ndarray],
    theta: NDArray[np.float64],
    out_shape: tuple[int, ...],
    order: int,
    method: str | None,
    inner_workers: int | None,
    outer_workers: int,
    dk_init_kwargs: dict[str, Any] | None = None,
    **dk_diff_kwargs: Any,
) -> NDArray[np.float64]:
    """Returns a hyper-Hessian for a scalar- or vector-valued function.

    Args:
        function: Scalar- or vector-valued function to differentiate.
        theta: 1D parameter vector where the derivatives are evaluated.
        out_shape: Shape of the output array.
        order: Derivative order.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        inner_workers: Number of inner workers for :class:`derivkit.DerivativeKit` calls.
        outer_workers: Number of outer workers for parallelism over entries.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        **dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        The full hyper-Hessian array for the scalar- or vector-valued function.

    Raises:
        TypeError: If ``function`` does not return a scalar or a vector.
    """
    p = int(theta.size)
    iw = int(inner_workers or 1)

    # Compute only unique entries, then symmetrize.
    index_combinations = list(
        combinations_with_replacement(range(p), order)
    )

    def entry_worker(*indices: int) -> float | NDArray[np.float64]:
        """Worker to compute one hyper-Hessian entry.

        Args:
            *indices: Parameter indices defining the derivative.

        Returns:
            Value of the requested derivative of the function at theta0.
        """
        return _higher_derivative_entry(
            function=function,
            theta0=theta,
            indices=indices,
            method=method,
            n_workers=iw,
            dk_init_kwargs=dk_init_kwargs,
            dk_diff_kwargs=dk_diff_kwargs,
        )

    vals = parallel_execute(
        entry_worker,
        arg_tuples=index_combinations,
        outer_workers=outer_workers,
        inner_workers=iw,
    )

    hess = np.empty((*out_shape, *([p] * order)), dtype=float)

    for indices, v in zip(index_combinations, vals, strict=True):
        v = np.asarray(v, dtype=float)
        for permutation in set(permutations(indices)):
            hess[(..., *permutation)] = v

    ensure_finite(hess, msg="Non-finite values encountered in hyper-Hessian.")
    return hess


def _higher_derivative_entry(
    *,
    function: Callable[[ArrayLike], float | np.ndarray],
    theta0: NDArray[np.float64],
    indices: Sequence[int],
    method: str | None,
    n_workers: int,
    dk_init_kwargs: dict[str, Any] | None,
    dk_diff_kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    """Computes one entry of a higher-order derivative tensor.

    Args:
        function: Scalar- or vector-valued function to differentiate.
        theta0: 1D parameter vector at which the derivative is evaluated.
        indices: Parameter indices defining the partial derivative.
        method: Derivative method name or alias. If ``None``,
            the :class:`derivkit.DerivativeKit` default is used.
        n_workers: Number of workers for :class:`derivkit.DerivativeKit` calls.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization.
        dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        Value of the requested derivative of the function at ``theta0``.
    """
    inner_init_kwargs = {"use_input_cache": False}
    inner_init_kwargs.update(dk_init_kwargs or {})

    indices = tuple(sorted(int(index) for index in indices))

    if len(set(indices)) == 1:
        index = indices[0]
        f1 = get_partial_function(function, index, theta0)
        kit = DerivativeKit(
            f1,
            float(theta0[index]),
            **inner_init_kwargs,
        )
        val = kit.differentiate(
            order=len(indices),
            method=method,
            n_workers=n_workers,
            **dk_diff_kwargs,
        )
        return np.asarray(val, dtype=float)

    outer_index = indices[-1]
    inner_indices = indices[:-1]

    def g_func(t: float) -> NDArray[np.float64]:
        """Function for the outer derivative.

        Args:
            t: Value of the outer differentiation parameter.

        Returns:
            Lower-order derivative evaluated with the outer parameter fixed.
        """
        th = theta0.copy()
        th[outer_index] = float(t)

        return _higher_derivative_entry(
            function=function,
            theta0=th,
            indices=inner_indices,
            method=method,
            n_workers=n_workers,
            dk_init_kwargs=dk_init_kwargs,
            dk_diff_kwargs=dk_diff_kwargs,
        )

    kit = DerivativeKit(
        g_func,
        float(theta0[outer_index]),
        **inner_init_kwargs,
    )
    val = kit.differentiate(
        order=1,
        method=method,
        n_workers=n_workers,
        **dk_diff_kwargs,
    )
    return np.asarray(val, dtype=float)
