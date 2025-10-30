"""Radial Basis Function (squared-exponential) kernel.

This kernel defines smooth correlations between inputs and provides
closed-form expressions for covariances that involve function values,
first-order derivatives, and second-order derivative diagonals.

Expected keys in ``params``:
- ``length_scale``: Positive float or 1D array (per-dimension).
- ``output_scale``: Positive float amplitude; internally used as a variance.

Attributes:
    name: Short identifier for this kernel ("rbf").
    param_names: Tuple of supported parameter names.
    ard_ok: Whether a per-dimension ``length_scale`` vector is supported.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from derivkit.gaussian_process.kernels.kernel_utils import (
    axis_length_scale_squared,
    output_variance,
    rbf_similarity,
    to_2d,
)
from derivkit.gaussian_process.kernels.registry import register_kernel

__all__ = ["RBFKernel"]


@register_kernel("rbf")
class RBFKernel:
    """Radial Basis Function (squared-exponential) kernel.

    Expected keys in ``params``:
        length_scale: float or 1D array (per-dimension). Must be > 0.
        output_scale: float amplitude. Squared internally to get variance.

    Notes:
        Provides closed-form covariances for values, value–gradient,
        gradient–gradient, value–Hessian-diagonal, and the same-point
        variance of the Hessian diagonal along a chosen axis.
    """

    name = "rbf"
    param_names = ("length_scale", "output_scale")
    ard_ok = True

    def cov_value_value(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        params: Mapping[str, float | np.ndarray],
    ) -> np.ndarray:
        """Covariance between function values at two sets of inputs.

        Args:
            x_train: Array of shape ``(n_points:left, n_dims)``.
            x_test: Array of shape ``(n_points_right, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.

        Returns:
            Array of shape ``(n_points_left, n_points_right)`` with pairwise covariances.

        Raises:
            ValueError: If input shapes are incompatible or contain invalid values.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
        return output_variance(params) * rbf_similarity(x_train, x_test, params)

    def cov_value_grad(
        self,
        x_value: np.ndarray,
        x_grad: np.ndarray,
        params: Mapping[str, float | np.ndarray],
        *,
        axis: int,
    ) -> np.ndarray:
        """Cross-covariance between values and first derivatives along one axis.

        Args:
            x_value: Locations where function values are evaluated, shape ``(n_val, n_dims)``.
            x_grad: Locations where first derivatives are evaluated, shape ``(n_grad, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based index of the input dimension used for the derivative.

        Returns:
            Array of shape ``(n_val, n_grad)`` with value-to-gradient covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
        base = self.cov_value_value(x_value, x_grad, params)
        ell2 = axis_length_scale_squared(params, axis)
        x_value = to_2d(x_value)
        x_grad = to_2d(x_grad)
        delta = x_value[:, None, axis] - x_grad[None, :, axis]
        return (delta / ell2) * base

    def cov_grad_grad(
        self,
        x_grad_left: np.ndarray,
        x_grad_right: np.ndarray,
        params: Mapping[str, float | np.ndarray],
        *,
        axis: int,
    ) -> np.ndarray:
        """Covariance between first derivatives taken along the same axis.

        Args:
            x_grad_left: First set of derivative locations, shape ``(n_left, n_dims)``.
            x_grad_right: Second set of derivative locations, shape ``(n_right, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based index of the input dimension used for the derivatives.

        Returns:
            Array of shape ``(n_left, n_right)`` with gradient-to-gradient covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
        base = self.cov_value_value(x_grad_left, x_grad_right, params)
        ell2 = axis_length_scale_squared(params, axis)
        x_grad_left = to_2d(x_grad_left)
        x_grad_right = to_2d(x_grad_right)
        delta = x_grad_left[:, None, axis] - x_grad_right[None, :, axis]
        return (1.0 / ell2 - (delta**2) / (ell2**2)) * base

    def value_hessdiag(
        self,
        x_value: np.ndarray,
        x_hess: np.ndarray,
        params: Mapping[str, float | np.ndarray],
        *,
        axis: int,
    ) -> np.ndarray:
        """Cross-covariance between values and the diagonal of the second derivative.

        This uses the second derivative with respect to a single input dimension
        (the diagonal term of the Hessian) at the query points.

        Args:
            x_value: Locations where function values are evaluated, shape ``(n_val, n_dims)``.
            x_hess:  Locations where second-derivative diagonals are evaluated, shape ``(n_hess, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based index of the input dimension used for the second derivative.

        Returns:
            Array of shape ``(n_val, n_hess)`` with value-to-curvature covariances.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
        base = self.cov_value_value(x_value, x_hess, params)
        ell2 = axis_length_scale_squared(params, axis)
        x_value = to_2d(x_value)
        x_hess = to_2d(x_hess)
        delta = x_value[:, None, axis] - x_hess[None, :, axis]
        return ((delta**2) / (ell2**2) - 1.0 / ell2) * base

    def hessdiag_samepoint(
        self,
        x_hess: np.ndarray,
        params: Mapping[str, float | np.ndarray],
        *,
        axis: int,
    ) -> np.ndarray:
        """Variance of the second-derivative diagonal at the same input locations.

        Returns a diagonal matrix where each diagonal entry is the variance of the
        second derivative (along one input dimension) at the corresponding query point.

        Args:
            x_hess: Locations where the second-derivative diagonal is evaluated,
                shape ``(n_hess, n_dims)``.
            params: Mapping with at least ``"length_scale"`` and ``"output_scale"``.
            axis: Zero-based index of the input dimension used for the second derivative.

        Returns:
            Array of shape ``(n_hess, n_hess)`` with variances on the diagonal.

        Raises:
            ValueError: If ``axis`` is out of range or inputs are incompatible.
            KeyError, TypeError: If required parameters are missing or have wrong types.
        """
        x_hess = to_2d(x_hess)
        ell2 = axis_length_scale_squared(params, axis)
        var = output_variance(params)
        val = var * 3.0 / (ell2**2)
        return np.eye(x_hess.shape[0]) * val
