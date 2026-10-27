"""Numerical diagnostics for Fisher and DALI forecasts.

These utilities inspect supplied matrices and model derivatives. They do not
re-evaluate the model or compute additional derivatives.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["matrix_diagnostics", "fisher_diagnostics", "dali_diagnostics"]


def _relative_tolerance(rtol: float | None, dimension: int) -> float:
    """Return a validated relative spectral tolerance.

    Args:
        rtol: Requested relative tolerance. If ``None``, uses
            ``dimension * eps`` for float64 precision.
        dimension: Matrix dimension used to determine the default tolerance.

    Returns:
        Finite, nonnegative relative tolerance.

    Raises:
        ValueError: If ``rtol`` is not a finite, nonnegative real number.
    """
    if rtol is None:
        return dimension * np.finfo(np.float64).eps
    if isinstance(rtol, (bool, np.bool_)) or not np.isscalar(rtol):
        raise ValueError("rtol must be a finite, nonnegative real number.")
    try:
        value = float(rtol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rtol must be a finite, nonnegative real number.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("rtol must be a finite, nonnegative real number.")
    return value


def _symmetric_matrix(matrix: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Validate a finite, nonempty symmetric matrix.

    Args:
        matrix: Array-like input expected to be a square, symmetric matrix.
        name: Name of the input used in validation error messages.

    Returns:
        Validated two-dimensional float64 NumPy array. The matrix values
        are not modified.

    Raises:
        ValueError: If the matrix is empty, nonsquare, contains non-finite
            values, or is not symmetric within the specified internal
            numerical tolerance.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(arr, arr.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    return arr


def _covariance_whitener(
    cov: ArrayLike, *, rcond: float
) -> NDArray[np.float64]:
    r"""Compute the symmetric pseudoinverse square root of a covariance.

    Args:
        cov: Symmetric positive semidefinite covariance matrix of shape
            ``(n, n)``.
        rcond: Relative eigenvalue cutoff used to determine which
            covariance modes are retained. Must be finite and nonnegative.

    Returns:
        Symmetric matrix representing the pseudoinverse square root of
        the covariance. Eigenmodes with eigenvalues below the cutoff
        contribute zero weight.

    Raises:
        ValueError: If the covariance is empty, nonsquare, non-finite,
            nonsymmetric, or has eigenvalues below the allowed negative
            tolerance.

    Notes:
        For the eigendecomposition

        .. math::

            C = Q \Lambda Q^{\mathsf T},

        the pseudoinverse square root is

        .. math::

            C^{+/2} = Q \Lambda^{+/2} Q^{\mathsf T},

        where retained eigenvalues receive inverse square-root weights
        and discarded eigenvalues receive zero weights.

        The cutoff is defined as ``rcond * max(abs(eigenvalues))``.
        Small negative eigenvalues within the tolerance are discarded.
    """
    matrix = _symmetric_matrix(cov, name="cov")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    scale = float(np.max(np.abs(values)))
    threshold = rcond * scale
    if np.any(values < -threshold):
        raise ValueError("cov must be positive semidefinite within tolerance.")
    inverse_sqrt = np.zeros_like(values)
    retained = values > threshold
    inverse_sqrt[retained] = 1.0 / np.sqrt(values[retained])
    return (vectors * inverse_sqrt) @ vectors.T


def matrix_diagnostics(
    matrix: ArrayLike,
    *,
    rtol: float | None = None,
) -> dict[str, Any]:
    r"""Compute eigenvalue diagnostics for a symmetric matrix.

    Args:
        matrix: Nonempty symmetric matrix of shape ``(p, p)``.
        rtol: Relative eigenvalue cutoff for numerical rank and positive
            semidefiniteness. Defaults to ``p * eps`` for float64 precision.

    Returns:
        Dictionary with ``eigenvalues`` (ascending), ``eigenvectors``
        (columns corresponding to eigenvalues), ``condition_number``
        (ratio of largest to smallest absolute eigenvalue, or infinity
        if numerically rank deficient), ``rank``, and
        ``is_positive_semidefinite``.

    Raises:
        ValueError: If the matrix is empty, nonsquare, nonfinite, or
            nonsymmetric, or if ``rtol`` is invalid.

    Notes:
        Rank and conditioning depend on parameter scaling and tolerance.
        An eigenvalue is counted as nonzero only if its absolute value
        exceeds ``rtol * max(abs(eigenvalues))``. A zero matrix has rank
        zero and infinite condition number.
    """
    arr = _symmetric_matrix(matrix, name="matrix")
    tolerance_factor = _relative_tolerance(rtol, arr.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (arr + arr.T))
    absolute = np.abs(eigenvalues)
    maximum = float(np.max(absolute))
    threshold = tolerance_factor * maximum
    rank = int(np.count_nonzero(absolute > threshold))
    condition_number = (
        float(maximum / np.min(absolute))
        if rank == arr.shape[0] and maximum > 0.0
        else float("inf")
    )
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "condition_number": condition_number,
        "rank": rank,
        "is_positive_semidefinite": bool(np.all(eigenvalues >= -threshold)),
    }


def fisher_diagnostics(
    fisher: ArrayLike,
    *,
    jacobian: ArrayLike | None = None,
    cov: ArrayLike | None = None,
    rtol: float | None = None,
    covariance_rcond: float = 1e-12,
) -> dict[str, Any]:
    r"""Compute Fisher matrix and optional noise-weighted sensitivity diagnostics.

    Args:
        fisher: Symmetric Fisher information matrix of shape ``(p, p)``.
        jacobian: Optional model Jacobian of shape ``(n_observables, p)``.
        cov: Observational covariance of shape
            ``(n_observables, n_observables)``. Supply together with
            ``jacobian``. May be positive semidefinite.
        rtol: Relative cutoff for matrix and Jacobian numerical ranks.
            The default is dimension dependent for each calculation.
        covariance_rcond: Relative eigenvalue cutoff used when forming
            the covariance pseudoinverse square root. Defaults to ``1e-12``.

    Returns:
        The dictionary from :func:`matrix_diagnostics`. If ``jacobian``
        and ``cov`` are provided, also includes
        ``jacobian_singular_values``, ``jacobian_rank``, and
        ``jacobian_condition_number``.

    Raises:
        ValueError: If inputs have invalid shapes or values, only one of
            ``jacobian`` and ``cov`` is supplied, or the covariance has
            a significantly negative eigenvalue.

    Notes:
        For a parameter independent covariance, the Fisher matrix is

        .. math::

            F = J^{\mathsf T} C^{+} J,

        where ``+`` denotes a pseudoinverse when necessary. The whitened
        Jacobian is ``W = C^{+/2} J``. Its singular values satisfy
        ``W.T @ W = J.T @ C^+ @ J``. When the covariance is strictly
        positive definite, this agrees with the usual Fisher expression.

        Numerical ranks of ``F`` and ``W`` can differ because forming
        ``F`` squares singular values. The supplied ``fisher`` is not
        checked against ``J.T @ C^+ @ J``.
    """
    fisher_array = _symmetric_matrix(fisher, name="fisher")
    diagnostics = matrix_diagnostics(fisher_array, rtol=rtol)
    if (jacobian is None) != (cov is None):
        raise ValueError("jacobian and cov must be supplied together.")
    if jacobian is None:
        return diagnostics

    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    if jacobian_array.ndim != 2 or 0 in jacobian_array.shape:
        raise ValueError("jacobian must be a nonempty 2D array.")
    if not np.all(np.isfinite(jacobian_array)):
        raise ValueError("jacobian must contain only finite values.")
    n_observables, n_parameters = jacobian_array.shape
    if fisher_array.shape != (n_parameters, n_parameters):
        raise ValueError("fisher shape must match the Jacobian parameter dimension.")
    cov_array = _symmetric_matrix(cov, name="cov")
    if cov_array.shape != (n_observables, n_observables):
        raise ValueError("cov shape must match the Jacobian observable dimension.")
    cutoff = _relative_tolerance(covariance_rcond, n_observables)
    whitener = _covariance_whitener(cov_array, rcond=cutoff)
    singular_values = np.linalg.svd(whitener @ jacobian_array, compute_uv=False)
    threshold_factor = _relative_tolerance(rtol, max(n_observables, n_parameters))
    maximum = float(np.max(singular_values))
    rank = int(np.count_nonzero(singular_values > threshold_factor * maximum))
    condition_number = (
        float(maximum / np.min(singular_values))
        if rank == n_parameters and maximum > 0.0
        else float("inf")
    )
    diagnostics.update({
        "jacobian_singular_values": singular_values,
        "jacobian_rank": rank,
        "jacobian_condition_number": condition_number,
    })
    return diagnostics


def dali_diagnostics(
    derivatives: dict[int, NDArray[np.float64]],
    cov: ArrayLike,
    displacement: ArrayLike,
    *,
    covariance_rcond: float = 1e-12,
) -> dict[str, Any]:
    r"""Measure directional nonlinear model response from DALI derivatives.

    Args:
        derivatives: Model derivative tensors keyed by orders 1 up to 4.
            Order 1 is required. An order ``k`` tensor has shape
            ``(n_observables, p, ..., p)`` with ``k`` parameter axes.
        cov: Symmetric positive semidefinite observational covariance
            of shape ``(n_observables, n_observables)``.
        displacement: Parameter displacement from the expansion point,
            with shape ``(p,)``.
        covariance_rcond: Relative cutoff used for the covariance
            pseudoinverse square root. Defaults to ``1e-12``.

    Returns:
        Dictionary with ``response_norms`` (covariance weighted norm of
        each Taylor response contribution), ``relative_response_strength``
        (higher order norms divided by the linear norm),
        ``linear_response_norm``, and ``total_response_norm`` (norm of
        the sum of available Taylor contributions).

        If the linear norm is zero, a positive higher order norm produces
        an infinite ratio, while zero divided by zero produces ``NaN``.

    Raises:
        ValueError: If the derivative dictionary is missing order 1,
            includes unsupported orders, or contains invalid tensor shapes
            or values; if the displacement is invalid; or if the covariance
            is not positive semidefinite within tolerance.

    Notes:
        At displacement :math:`\Delta\theta`, the order ``k`` model
        response contribution is

        .. math::

            r_k = \frac{1}{k!}\nu_{,i_1\ldots i_k}
            \Delta\theta_{i_1}\cdots\Delta\theta_{i_k}.

        Its noise-weighted norm is

        .. math::

            \|r_k\|_{C^+} = \sqrt{r_k^{\mathsf T}C^+r_k},

        and the relative nonlinear response strength is

        .. math::

            R_k = \frac{\|r_k\|_{C^+}}{\|r_1\|_{C^+}}.

        These are directional diagnostics of the model response, not
        diagnostics of the assembled DALI likelihood tensors and not
        guarantees of convergence or posterior accuracy. They depend on
        the chosen displacement.
    """
    if not isinstance(derivatives, dict) or 1 not in derivatives:
        raise ValueError("derivatives must be a dictionary containing order 1.")
    if any(type(order) is not int or order not in (1, 2, 3, 4) for order in derivatives):
        raise ValueError("derivative orders must be integers between 1 and 4.")

    displacement_array = np.asarray(displacement, dtype=np.float64)
    if displacement_array.ndim != 1 or displacement_array.size == 0:
        raise ValueError("displacement must be a nonempty 1D array.")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError("displacement must contain only finite values.")
    jacobian = np.asarray(derivatives[1], dtype=np.float64)
    if jacobian.ndim != 2 or 0 in jacobian.shape:
        raise ValueError("first order derivatives must be a nonempty 2D array.")
    n_observables, n_parameters = jacobian.shape
    if displacement_array.shape != (n_parameters,):
        raise ValueError("displacement length must match the number of parameters.")
    cov_array = _symmetric_matrix(cov, name="cov")
    if cov_array.shape != (n_observables, n_observables):
        raise ValueError("cov shape must match the number of observables.")
    cutoff = _relative_tolerance(covariance_rcond, n_observables)
    whitener = _covariance_whitener(cov_array, rcond=cutoff)

    responses: dict[int, NDArray[np.float64]] = {}
    response_norms: dict[int, float] = {}
    for order in sorted(derivatives):
        derivative = np.asarray(derivatives[order], dtype=np.float64)
        expected_shape = (n_observables,) + (n_parameters,) * order
        if derivative.shape != expected_shape:
            raise ValueError(
                f"derivative order {order} has shape {derivative.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(derivative)):
            raise ValueError(f"derivative order {order} contains non-finite values.")
        response = derivative
        for _ in range(order):
            response = np.tensordot(response, displacement_array, axes=([-1], [0]))
        response = np.asarray(response / factorial(order), dtype=np.float64)
        responses[order] = response
        response_norms[order] = float(np.linalg.norm(whitener @ response))

    linear_norm = response_norms[1]
    relative_strength = {
        order: (
            norm / linear_norm if linear_norm > 0.0
            else float("inf") if norm > 0.0 else float("nan")
        )
        for order, norm in response_norms.items()
        if order != 1
    }
    total_response = sum(responses.values(), np.zeros(n_observables, dtype=np.float64))
    return {
        "response_norms": response_norms,
        "relative_response_strength": relative_strength,
        "linear_response_norm": linear_norm,
        "total_response_norm": float(np.linalg.norm(whitener @ total_response)),
    }
