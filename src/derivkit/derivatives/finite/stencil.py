"""Stencil definitions and utilities for finite-difference derivative calculations."""

import math

import numpy as np
from numpy.typing import NDArray

__all = [
    "get_finite_difference_tables",
    "validate_supported_combo",
    "SUPPORTED_BY_STENCIL",
    "TRUNCATION_ORDER",
    "STENCILS",
    "ORDERS",
]


#: A list of supported stencil sizes.
STENCILS = (3, 5, 7, 9)
#: A list of supported derivative orders.
ORDERS = (1, 2, 3, 4)


def supported_orders(
    num_points: int, 
    *, 
    max_order: int = 4,
) -> set[int]:
    """Creates a list of supported derivative orders for a given stencil size, and a maximum derivative order.

    Args:
        num_points: Number of points in the stencil. The supported stencil sizes are defined in :data:`STENCILS`.
        max_order: The maximum supported derivative order.

    Returns:
        The list of supported derivative orders for the given stencil size.

    Raises:
        ValueError: If ``num_points`` is not in :data:`STENCILS`.
        ValueError: If ``max_order < 1`` or ``max_order > 4``.
    """
    if num_points not in STENCILS:
        raise ValueError("num_points must be 3, 5, 7, or 9")
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    if max_order > 4:
        raise ValueError("max_order must be 4 or less")

    return set(range(1, min(max_order, num_points - 1) + 1))


#: Dictionary containing the derivative orders by the available stencil sizes.
SUPPORTED_BY_STENCIL = {n: supported_orders(n) for n in STENCILS}


def _central_offsets(
    num_points: int,
) -> NDArray[np.float64]:
    """Creates a grid of central offset values for a desired number of points.

    Args:
        num_points: Number of points desired in the grid.

    Returns:
        An array of offset values centered at zero.
    """
    half = num_points // 2
    return np.arange(-half, half + 1, dtype=np.float64)


def truncation_order_from_coeffs(
    offsets: NDArray[np.float64], 
    coeffs: NDArray[np.float64], 
    deriv_order: int,
    tol: float = 1e-12,
) -> int:
    """Computes the truncation order from the coefficients and offsets, for some numerical tolerance.
    
    Args:
        offsets: Array of integer offsets for the finite difference stencil.
        coeffs: Array of finite difference coefficients.
        deriv_order: The requested derivative order.
        tol: Numerical tolerance used to determine the truncation order.
            
    Returns:
        The truncation order for the given numerical tolerance.
    """

    m = deriv_order
    max_r = 40  # plenty for n<=9

    for r in range(m + 1, max_r + 1):
        moment = float(np.dot(coeffs, offsets**r))
        if abs(moment) > tol:
            return r - m
    raise RuntimeError("Could not detect truncation order.")


def _finite_difference_coeffs(
    offsets: list[int] | NDArray,
    deriv_order: int,
    stepsize: float,
) -> NDArray:
    """Compute finite difference coefficients for given offsets and derivative order.

    This method solves a linear system to find the coefficients that
    approximate the derivative of specified order using the provided offsets.

    Args:
        offsets: List or array of integer offsets for the finite difference stencil.
        deriv_order: The order of the derivative to approximate.
        stepsize: The stepsize used in the finite difference calculation.

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


def _build_truncation_orders(
) -> dict[tuple[int, int], int]:
    """Dynamically computes the truncation orders for the supported stencil combinations.

    Returns:
        A dictionary of truncation order for the supported stencil sizes and derivative orders.
    """
    out: dict[tuple[int, int], int] = {}
    h = 1.0  # any value works for p-detection
    for n in STENCILS:
        k = _central_offsets(n)  # integers as floats
        for m in supported_orders(n):
            c = _finite_difference_coeffs(k, m, h)
            out[(n, m)] = truncation_order_from_coeffs(k, c, m)
    return out


#: Dictionary of truncation order for the supported stencil sizes and derivative orders.
TRUNCATION_ORDER = _build_truncation_orders()


def get_finite_difference_tables(
    stepsize: float,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], np.ndarray]]:
    """Dynamically computes offset patterns and coefficient tables.

    Args:
        stepsize: The step size to use for the stencil spacing.
    
    Returns:
        A dictionary of offsets, and a dictionary of coefficient tables, 
        for a range of supported stencil sizes and derivative orders.
    """
    offsets = {n: _central_offsets(n).tolist() for n in STENCILS}
    coeffs_table = {}

    for n, ks in offsets.items():
        k = _central_offsets(n)
        for m in supported_orders(n):
            c = _finite_difference_coeffs(ks, m, 1.0)
            coeffs_table[(n, m)] = _finite_difference_coeffs(k, m, stepsize)

    return offsets, coeffs_table


def validate_supported_combo(
    num_points: int, 
    order: int,
) -> None:
    """Validates that the (stencil size, order) combo is supported.

    Args:
        num_points: Number of points in the finite difference stencil.
        order: The order of the derivative to compute.

    Raises:
        ValueError: If the combination of num_points and order is not supported.
    """
    list_STENCILS = list(STENCILS)
    if num_points not in STENCILS:
        raise ValueError(
            f"[FiniteDifference] Unsupported stencil size: {num_points}. "
            f"Must be one of {list_STENCILS}."
        )
    list_ORDERS = list(ORDERS)
    if order not in ORDERS:
        raise ValueError(
            f"[FiniteDifference] Unsupported derivative order: {order}. "
            f"Must be one of {list_ORDERS}."
        )

    allowed = SUPPORTED_BY_STENCIL[num_points]
    if order not in allowed:
        raise ValueError(
            "[FiniteDifference] Not implemented yet: "
            f"{num_points}-point stencil for order {order}.\n"
        )
