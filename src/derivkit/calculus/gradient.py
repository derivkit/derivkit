"""Contains functions used to construct the gradient of scalar-valued functions."""

from collections.abc import Callable
from functools import partial

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.caching import wrap_input_cache
from derivkit.utils.concurrency import (
    parallel_execute,
    resolve_inner_from_outer,
)
from derivkit.utils.sandbox import get_partial_function
from derivkit.utils.validate import check_scalar_valued


def build_gradient(
    function: Callable,
    theta0: np.ndarray,
    method: str | None = None,
    n_workers: int = 1,
    dk_init_kwargs: dict | None = None,
    **dk_diff_kwargs: dict,
) -> np.ndarray:
    """Returns the gradient of a scalar-valued function.

    Args:
        function: The function to be differentiated.
        theta0: The parameter vector at which the gradient is evaluated.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Number of workers used by
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.
            This setting does not parallelize across parameters. Default is ``1``.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        **dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A 1D array representing the gradient.

    Raises:
        TypeError: If ``function`` does not return a scalar value.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    inner = resolve_inner_from_outer(n_workers)

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

    check_scalar_valued(shared_function, theta0, 0, n_workers)

    worker = partial(
        _grad_component,
        method=method,
        n_workers=inner,
        dk_init_kwargs=dk_init_kwargs,
        dk_diff_kwargs=dk_diff_kwargs,
    )
    tasks = [(shared_function, theta0, i) for i in range(theta0.size)]

    vals = parallel_execute(
        worker,
        tasks,
        outer_workers=n_workers,
        inner_workers=inner,
    )
    grad = np.asarray(vals, dtype=float)
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in build_gradient.")
    return grad


def _grad_component(
    function: Callable,
    theta0: np.ndarray,
    i: int,
    method: str | None = None,
    n_workers: int = 1,
    dk_init_kwargs: dict | None = None,
    dk_diff_kwargs: dict | None = None,
) -> float:
    """Returns one entry of the gradient for a scalar-valued function.

    Used inside :func:`derivkit.calculus.gradient.build_gradient` to find how
    the function changes with respect to a single parameter while keeping the
    others fixed.

    Args:
        function: A function that returns a single value.
        theta0: The parameter values where the derivative is evaluated.
        i: The index of the parameter being varied.
        method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
            default (``"adaptive"``) is used.
        n_workers: Number of workers used for the internal derivative step. Default is ``1``.
        dk_init_kwargs: Optional keyword arguments passed to
            :class:`derivkit.derivative_kit.DerivativeKit` during
            initialization. This can include cache-related settings.
        dk_diff_kwargs: Additional keyword arguments passed to
            :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

    Returns:
        A single number showing how the function changes with that parameter.
    """
    partial_vec = get_partial_function(function, i, theta0)

    inner_init_kwargs = {"use_input_cache": False}
    inner_init_kwargs.update(dk_init_kwargs or {})

    kit = DerivativeKit(partial_vec, float(theta0[i]), **inner_init_kwargs)

    deriv = kit.differentiate(
        order=1,
        method=method,
        n_workers=n_workers,
        **(dk_diff_kwargs or {}),
    )

    deriv = np.asarray(deriv, dtype=float)
    if deriv.size != 1:
        raise TypeError(
            "Gradient component derivative must be scalar-like, "
            f"got shape {deriv.shape}."
        )
    return float(deriv.reshape(()))
