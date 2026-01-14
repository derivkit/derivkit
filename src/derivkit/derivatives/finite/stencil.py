"""Stencil definitions and utilities for finite-difference derivative calculations."""

import math

import numpy as np
from numpy.typing import NDArray

__all = [
    "get_finite_difference_tables",
    "validate_supported_combo",
    "SUPPORTED_BY_STENCIL",
    "TRUNCATION_ORDER",
]

SUPPORTED_BY_STENCIL: dict[int, set[int]] = {
    3: {1, 2},
    5: {1, 2, 3, 4},
    7: {1, 2, 3, 4},
    9: {1, 2, 3, 4},
}

TRUNCATION_ORDER: dict[tuple[int, int], int] = {
    (3, 1): 2,
    (3, 2): 0,
    (5, 1): 4,
    (5, 2): 4,
    (5, 3): 2,
    (5, 4): 2,
    (7, 1): 6,
    (7, 2): 6,
    (7, 3): 0,
    (7, 4): 0,
    (9, 1): 8,
    (9, 2): 8,
    (9, 3): 0,
    (9, 4): 0,
}


def _finite_difference_coeffs(
    offsets: list[int] | NDArray,
    deriv_order: int,
    stepsize: float,
) -> NDArray:
    """Compute finite difference coefficients for given offsets and derivative order.

    This method solves a linear system to find the coefficients that
    approximate the derivative of specified order using the provided offsets.

    Args:
        offsets:
            List or array of integer offsets for the finite difference stencil.
        deriv_order:
            The order of the derivative to approximate.
        stepsize:
            The stepsize used in the finite difference calculation.

    Returns:
        An array of finite difference coefficients.
    """
    offsets = np.asarray(offsets, dtype=float)
    n = offsets.size

    matrix = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # Match Taylor expansion up to degree n-1
    for k in range(n):
        matrix[k, :] = offsets**k / math.factorial(k)
    b[deriv_order] = 1.0  # enforce correct derivative of order m

    coeffs = np.linalg.solve(matrix, b) / (stepsize**deriv_order)
    return coeffs


def _get_finite_difference_tables_dynamically(
    stepsize: float,
    stencil_sizes: tuple[int, ...] = (3, 5, 7, 9),
    max_order: int = 4,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], np.ndarray]]:
    """Dynamically computes offset patterns and coefficient tables.

    I am not adding this to __all__ since the static version is preferred for now.
    Also Matthijs will add Fornberg's method later.
    """
    offsets: dict[int, list[int]] = {}
    coeffs_table: dict[tuple[int, int], np.ndarray] = {}

    for s in stencil_sizes:
        # symmetric odd stencil around 0: e.g. 5 -> [-2,-1,0,1,2]
        half = s // 2
        o = list(range(-half, half + 1))
        offsets[s] = o

        for m in range(1, max_order + 1):
            # only build combinations that make sense
            if m > s - 1:  # or whatever rules you want
                continue
            coeffs_table[(s, m)] = _finite_difference_coeffs(o, m, stepsize)

    return offsets, coeffs_table


def get_finite_difference_tables(
        stepsize: float
    ) -> tuple[dict[int, list[int]], dict[tuple[int, int], np.ndarray]]:
    """Returns the offset patterns and coefficient tables.

    Args:
        stepsize:
            Stepsize for finite difference calculation.

    Returns:
        A tuple of two dictionaries. The first maps from
        stencil size to symmetric offsets. The second maps from
        (stencil_size, order) to coefficient arrays.
    """
    offsets = {
        3: [-1, 0, 1],
        5: [-2, -1, 0, 1, 2],
        7: [-3, -2, -1, 0, 1, 2, 3],
        9: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    }

    coeffs_table = {
        (3, 1): np.array([-0.5, 0, 0.5]) / stepsize,
        (3, 2): np.array([1, -2, 1]) / (stepsize**2),
        (5, 1): np.array([1, -8, 0, 8, -1]) / (12 * stepsize),
        (5, 2): np.array([-1, 16, -30, 16, -1]) / (12 * stepsize**2),
        (5, 3): np.array([-1, 2, 0, -2, 1]) / (2 * stepsize**3),
        (5, 4): np.array([1, -4, 6, -4, 1]) / (stepsize**4),
        (7, 1): np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * stepsize),
        (7, 2): np.array([2, -27, 270, -490, 270, -27, 2])
        / (180 * stepsize**2),
        (7, 3): np.array([ 1,  -8,   13,   0,  -13,   8,  -1]) / (  8*stepsize**3),
        (7, 4): np.array([-1,  12,  -39,  56,  -39,  12,  -1]) / (  6*stepsize**4),
        (9, 1): np.array([3, -32, 168, -672, 0, 672, -168, 32, -3])
        / (840 * stepsize),
        (9, 2): np.array(
            [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]
        )
        / (5040 * stepsize**2),
        (9, 3): np.array([ -7,   72,  -338,  488,    0, -488,  338,  -72,   7]) / (240*stepsize**3),
        (9, 4): np.array([  7,  -96,   676, -1952, 2730, -1952, 676,  -96,   7]) / (240*stepsize**4),
    }

    return offsets, coeffs_table


def validate_supported_combo(num_points: int, order: int) -> None:
    """Validates that the (stencil size, order) combo is supported.

    Args:
        num_points:
            Number of points in the finite difference stencil.
        order:
            The order of the derivative to compute.

    Raises:
        ValueError: If the combination of num_points and order is not supported.
    """
    if num_points not in (3, 5, 7, 9):
        raise ValueError(
            f"[FiniteDifference] Unsupported stencil size: {num_points}. "
            "Must be one of [3, 5, 7, 9]."
        )
    if order not in (1, 2, 3, 4):
        raise ValueError(
            f"[FiniteDifference] Unsupported derivative order: {order}. "
            "Must be one of [1, 2, 3, 4]."
        )

    allowed = SUPPORTED_BY_STENCIL[num_points]
    if order not in allowed:
        raise ValueError(
            "[FiniteDifference] Not implemented yet: "
            f"{num_points}-point stencil for order {order}.\n"
        )
