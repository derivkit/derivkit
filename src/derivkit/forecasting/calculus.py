"""Differential calculus helpers."""

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils import get_partial_function


def gradient(function, theta0, n_workers=1):
    """Returns the gradient of a scalar-valued function.

    Args:
        function: f(theta) -> scalar.
        theta0: 1D parameter point (array-like).
        n_workers (int): Number of workers used by DerivativeKit.adaptive.differentiate.
                        This setting does not parallelize across parameters.

    Returns:
        1D array of gradient values.
    """
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    # n_workers controls inner 1D differentiation (not across parameters).
    grad = np.array(
        [_grad_component(function, theta0, i, n_workers) for i in range(theta0.size)],
        dtype=float,
    )
    if not np.isfinite(grad).all():
        raise FloatingPointError("Non-finite values encountered in gradient.")
    return grad

def _grad_component(function, theta0: np.ndarray, i: int, n_workers: int) -> float:
    """Compute the partial derivative ∂f/∂θ_i at ``theta0``.

    Helper used by ``gradient``. Wraps ``function`` into a single-variable
    callable via ``derivkit.utils.get_partial_function`` and differentiates it
    with ``DerivativeKit.adaptive.differentiate``.

    Args:
        function: Callable ``f(theta) -> scalar``. Must be scalar-valued; this
            is enforced by ``get_partial_function``.
        theta0: 1D array of parameter values at which to evaluate the derivative.
            The input is not mutated.
        i: Zero-based index of the parameter with respect to which to differentiate.
        n_workers: Number of workers used inside
            ``DerivativeKit.adaptive.differentiate``. This does not parallelize
            across parameters.

    Returns:
        float: The partial derivative ∂f/∂θ_i evaluated at ``theta0``.

    Raises:
        ValueError: If ``i`` is out of bounds for ``theta0``.
        TypeError: If ``function`` does not return a scalar value.
    """
    i = int(i)  # tolerate NumPy integer types
    partial_vec = get_partial_function(function, i, theta0)

    # One-time scalar validation (no wrapper function needed)
    probe = np.asarray(partial_vec(theta0[i]), dtype=float)
    if probe.size != 1:
        raise TypeError(
            "gradient() expects a scalar-valued function; "
            f"got shape {probe.shape} from full_function(params)."
        )

    # If DerivativeKit accepts size-1/0-d arrays as scalars, we can pass partial_vec directly.
    # Otherwise (if it requires a true float), we can switch to a tiny lambda cast later.
    kit = DerivativeKit(partial_vec, theta0[i])
    return kit.adaptive.differentiate(order=1, n_workers=n_workers)


def jacobian(*args, **kwargs):
    """This is a placeholder for a Jacobian computation function."""
    raise NotImplementedError
def hessian(*args, **kwargs):
    """This is a placeholder for a Hessian computation function."""
    raise NotImplementedError
def hessian_diag(*args, **kwargs):
    """This is a placeholder for a Hessian diagonal computation function."""
    raise NotImplementedError
def jacobian_diag(*args, **kwargs):
    """This is a placeholder for a Jacobian diagonal computation function."""
    raise NotImplementedError
def gauss_newton_hessian(*args, **kwargs):
    """This is a placeholder for a Gauss-Newton Hessian computation function."""
    raise NotImplementedError
