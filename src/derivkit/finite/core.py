"""Finite difference derivative estimation with a single step size."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .batch_eval import eval_points
from .stencil import get_finite_difference_tables

__all__ = [
    "single_finite_step",
]


def single_finite_step(
    function,
    x0: float,
    order: int,
    stepsize: float,
    num_points: int,
    n_workers: int,
) -> NDArray | float:
    """Returns one central finite-difference estimate at a given step size h.

    Args:
        function:
            The function whose derivative is to be estimated. Must accept
            a float or NumPy array and return a float or NumPy array.
        x0:
            The point at which to evaluate the derivative.
        order:
            The order of the derivative to compute.
        stepsize:
            The step size (h) used to evaluate the function around x0.
        num_points:
            The number of points in the finite difference stencil. Must be
            one of [3, 5, 7, 9].
        n_workers:
            The number of workers to use in multiprocessing. Default is 1.

    Returns:
        The estimated derivative. Returns a float for scalar-valued functions,
        or a NumPy array for vector-valued functions.

    Raises:
        ValueError:
            If the combination of ``num_points`` and ``order`` is not supported.
    """
    offsets, coeffs_table = get_finite_difference_tables(stepsize)
    key = (num_points, order)
    if key not in coeffs_table:
        raise ValueError(
            f"[FiniteDifference] Internal table missing coefficients for "
            f"stencil={num_points}, order={order}."
        )

    stencil = np.array(
        [x0 + i * stepsize for i in offsets[num_points]],
        dtype=float,
    )

    values = eval_points(function, stencil, n_workers=n_workers)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    derivs = values.T @ coeffs_table[key]
    return derivs.ravel() if derivs.size > 1 else float(derivs.item())
