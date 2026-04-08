"""Contains functions used to construct the Jacobian matrix."""

from collections.abc import Callable
from functools import partial
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


def build_jacobian(
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    method: str | None = None,
    n_workers: int | None = 1,
    dk_init_kwargs: dict[str, Any] | None = None,
    **dk_diff_kwargs: Any,
) -> NDArray[np.floating]:
    """Computes the Jacobian of a vector-valued function.

    Each column in the Jacobian is the derivative with respect to one parameter.

    Args:
        function: The vector-valued function to be differentiated.
            It should accept a list or array of parameter values as input and
            return an array of observable values.
        theta0: The parameter vector at which the jacobian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Number of workers used to parallelize across
            parameters. If None or 1, no parallelization is used.
            If greater than 1, this many threads will be used to compute
            derivatives with respect to different parameters in parallel.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        **dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A 2D array representing the jacobian. Each column corresponds to
            the derivative with respect to one parameter.

    Raises:
        FloatingPointError: If non-finite values are encountered.
        ValueError: If ``theta0`` is an empty array.
        TypeError: If ``function`` does not return a vector value.
    """
    # Validate inputs and evaluate baseline output
    theta = np.asarray(theta0, dtype=float).ravel()
    if theta.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    y0 = np.asarray(function(theta), dtype=float)
    if y0.ndim != 1:
        raise TypeError(
            f"build_jacobian expects f: R^n -> R^m with 1-D vector output; got shape {y0.shape}"
        )
    if not np.isfinite(y0).all():
        raise FloatingPointError("Non-finite values in model output at theta0.")

    m = int(y0.size)
    n = int(theta.size)

    try:
        outer_workers = max(1, int(n_workers or 1))
    except (TypeError, ValueError):
        outer_workers = 1
    inner_workers = resolve_inner_from_outer(outer_workers)

    dk_init_kwargs = dict(dk_init_kwargs or {})

    use_input_cache = dk_init_kwargs.pop("use_input_cache", True)
    cache_number_decimal_places = dk_init_kwargs.pop(
        "cache_number_decimal_places", None
    )
    cache_maxsize = dk_init_kwargs.pop("cache_maxsize", 4096)
    cache_copy = dk_init_kwargs.pop("cache_copy", True)

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

    worker = partial(
        _column_derivative,
        function=shared_function,
        theta0=theta,
        method=method,
        inner_workers=inner_workers,
        expected_m=m,
        dk_init_kwargs=dk_init_kwargs,
        **dk_diff_kwargs,
    )

    cols = parallel_execute(
        worker,
        arg_tuples=[(j,) for j in range(n)],
        outer_workers=outer_workers,
        inner_workers=inner_workers,
    )

    jac = np.column_stack([np.asarray(c, dtype=float).reshape(m) for c in cols])
    return jac


def _column_derivative(
    j: int,
    function: Callable[[ArrayLike], ArrayLike | float],
    theta0: ArrayLike,
    method: str | None,
    inner_workers: int | None,
    expected_m: int,
    dk_init_kwargs: dict[str, Any] | None = None,
    **dk_diff_kwargs: Any,
) -> NDArray[np.floating]:
    """Derivative of function with respect to parameter j.

    Args:
        j: Index of the parameter to differentiate with respect to.
        function: The vector-valued function to be differentiated.
        theta0: The parameter vector at which the jacobian is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        inner_workers: Number of workers used by
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
        expected_m: Expected length of the derivative vector.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        **dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A 1D array representing the derivative with respect to parameter j.

    Raises:
        TypeError: If the derivative does not have the expected length.
        FloatingPointError: If non-finite values are encountered.
    """
    theta_x = np.asarray(theta0, dtype=float).ravel().copy()
    f_j = get_partial_function(function, j, theta_x)

    method_norm = None
    if method is not None:
        m = method.lower()
        alias = {"auto": "adaptive", "fd": "finite"}
        method_norm = alias.get(m, m)

    inner_init_kwargs = {"use_input_cache": False}
    inner_init_kwargs.update(dk_init_kwargs or {})

    kit = DerivativeKit(
        f_j,
        theta_x[j],
        **inner_init_kwargs,
    )
    g = kit.differentiate(
        method=method_norm,
        order=1,
        n_workers=inner_workers,
        **dk_diff_kwargs,
    )

    g = np.atleast_1d(np.asarray(g, dtype=float)).reshape(-1)
    if g.size != expected_m:
        raise TypeError(
            f"Expected derivative of length {expected_m} but got {g.size} for parameter index {j}."
        )
    if not np.isfinite(g).all():
        raise FloatingPointError(f"Non-finite derivative for parameter index {j}.")

    return g
