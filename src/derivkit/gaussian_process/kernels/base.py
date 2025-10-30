"""Radial Basis Function (squared-exponential) kernel.

This kernel defines smooth correlations between inputs and provides
closed-form expressions for covariances involving function values,
first-order derivatives, and the diagonal of second-order derivatives.

Expected keys in ``params``:
- ``length_scale``: Positive float or 1D array with one entry per input dimension.
- ``output_scale``: Positive float amplitude; used internally as a variance.

Attributes:
    name: Short identifier for this kernel ("rbf").
    param_names: Tuple of supported parameter names.
    ard_ok: Whether a per-dimension ``length_scale`` vector is supported.
"""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["Kernel"]


class Kernel(Protocol):
    """Protocol every kernel must implement (supports up to 2nd derivatives).

    Attributes:
        name: Short, lowercase identifier (e.g., "rbf").
        param_names: Allowed parameter keys (e.g., ("length_scale", "output_scale")).
        ard_ok: Whether ``length_scale`` may be a vector (ARD).
    """
    name: str
    param_names: tuple[str, ...]
    ard_ok: bool

    def cov_value_value(
            self,
            x_train: NDArray[np.floating],
            x_test: NDArray[np.floating],
            params: Mapping[str, float | np.ndarray],
    ) -> NDArray[np.floating]:
        """Covariance between function values at two sets of inputs.

        Args:
            x_train: Array of shape ``(n_points_train, n_dims)``.
            x_test: Array of shape ``(n_points_test, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.

        Returns:
            Array of shape ``(n_points_train, n_points_test)`` with pairwise covariances.

        Raises:
            ValueError: If input shapes are incompatible or contain invalid values.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """


    def cov_value_grad(
            self,
            x_value: NDArray[np.floating],
            x_grad: NDArray[np.floating],
            params: Mapping[str, float | np.ndarray],
            *,
            axis: int,
    ) -> NDArray[np.floating]:
        """Cross-covariance between values and first derivatives along one axis.

        Args:
            x_value: Locations for function values, shape ``(n_points_value, n_dims)``.
            x_grad: Locations for first derivatives, shape ``(n_points_grad, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based derivative axis. Must be in ``[0, n_dims)``.

        Returns:
            Array of shape ``(n_points_value, n_points_grad)`` with value-to-gradient covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """


    def cov_grad_grad(
            self,
            x_grad_left: NDArray[np.floating],
            x_grad_right: NDArray[np.floating],
            params: Mapping[str, float | np.ndarray],
            *,
            axis: int,
    ) -> NDArray[np.floating]:
        """Covariance between first derivatives taken along the same axis.

        Args:
            x_grad_left: First set of derivative locations, shape ``(n_points_left, n_dims)``.
            x_grad_right: Second set of derivative locations, shape ``(n_points_right, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based derivative axis. Must be in ``[0, n_dims)``.

        Returns:
            Array of shape ``(n_points_left, n_points_right)`` with gradient-to-gradient covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """


    def cov_value_hessdiag(
            self,
            x_value: NDArray[np.floating],
            x_hess: NDArray[np.floating],
            params: Mapping[str, float | np.ndarray],
            *,
            axis: int,
    ) -> NDArray[np.floating]:
        """Cross-covariance between values and the diagonal of the second derivative.

        The second derivative is taken with respect to a single input dimension
        (the diagonal term of the Hessian) at the query points.

        Args:
            x_value: Locations for function values, shape ``(n_points_value, n_dims)``.
            x_hess: Locations for second-derivative diagonals, shape ``(n_points_hess, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based derivative axis. Must be in ``[0, n_dims)``.

        Returns:
            Array of shape ``(n_points_value, n_points_hess)`` with value-to-curvature covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """


    def var_hessdiag_samepoint(
            self,
            x_hess: NDArray[np.floating],
            params: Mapping[str, float | np.ndarray],
            *,
            axis: int,
    ) -> NDArray[np.floating]:
        """Variance of the second-derivative diagonal at the same input locations.

        Returns a diagonal matrix where each diagonal entry is the variance of the
        second derivative (along one input dimension) at the corresponding query point.

        Args:
            x_hess: Locations for the second-derivative diagonal, shape ``(n_points_hess, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based derivative axis. Must be in ``[0, n_dims)``.

        Returns:
            Array of shape ``(n_points_hess, n_points_hess)`` with variances on the diagonal.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
