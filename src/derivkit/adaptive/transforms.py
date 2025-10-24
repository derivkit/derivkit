"""Parameter reparameterization helpers and derivative pullbacks.

This module provides small, self-contained transforms that make adaptive
polynomial fitting robust near parameter boundaries.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "signed_log_forward",
    "signed_log_to_physical",
    "signed_log_derivatives_to_x",
    "sqrt_domain_forward",
    "sqrt_to_physical",
    "pullback_sqrt_at_zero",
]


def signed_log_forward(x0: float) -> Tuple[float, float]:
    """Computes the signed-log coordinates for an expansion point.

    The *signed-log* map represents a physical coordinate ``x`` as
    ``x = sgn * exp(q)``, where ``q = log(|x|)`` and ``sgn = sign(x)``.
    Here, **physical** means the model’s native parameter (``x``), while
    **internal** means the reparameterized coordinate used for numerics (``q``).
    This reparameterization keeps multiplicative variation (orders of magnitude)
    well-behaved and avoids crossing through zero during local polynomial fits.

    Args:
        x0: Expansion point in physical coordinates. Must be finite and non-zero.

    Returns:
        Tuple[float, float]: ``(q0, sgn)``, where ``q0 = log(|x0|)`` and
        ``sgn = +1.0`` if ``x0 > 0`` else ``-1.0``.

    Raises:
        ValueError: If ``x0`` is not finite or equals zero.
    """
    if not np.isfinite(x0):
        raise ValueError("signed_log_forward requires a finite value of x0.")
    if x0 == 0.0:
        raise ValueError("signed_log_forward requires that x0 is non-zero.")
    sgn = 1.0 if x0 > 0.0 else -1.0
    q0 = np.log(abs(x0))
    return q0, sgn


def signed_log_to_physical(q: np.ndarray, sgn: float) -> np.ndarray:
    """Map internal signed-log coordinate(s) to physical coordinate(s).

    Args:
        q: Internal coordinate(s) q = log(abs(x)).
        sgn: Fixed sign (+1 or -1) taken from sign(x0).

    Returns:
        Physical coordinate(s) x = sgn * exp(q).

    Raises:
        ValueError: If `sgn` is not +1 or -1, or if `q` contains non-finite values.
    """
    try:
        sgn = _normalize_sign(sgn)
    except ValueError as e:
        raise ValueError(f"signed_log_to_physical: invalid `sgn`: {e}") from None

    q = np.asarray(q, dtype=float)
    try:
        _require_finite("q", q)
    except ValueError as e:
        raise ValueError(f"signed_log_to_physical: {e}") from None

    return sgn * np.exp(q)


def signed_log_derivatives_to_x(
    order: int,
    x0: float,
    dfdq: np.ndarray,
    d2fdq2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Converts derivatives from the signed-log coordinate ``q`` to the original parameter ``x`` at ``x0 ≠ 0``.

    This method uses the chain rule to convert derivatives computed in the
    internal signed-log coordinate q back to physical coordinates x at a
    non-zero expansion point x0.

    Args:
        order: Derivative order to return (1 or 2).
        x0: Expansion point in physical coordinates; must be non-zero.
        dfdq: First derivative in q (shape: (n_comp,) or broadcastable).
        d2fdq2: Second derivative in q (same shape as dfdq); required for order=2.

    Returns:
        The derivative(s) in physical coordinates at x0.

    Raises:
        ValueError: If `x0 == 0`, if required inputs (d2fdq2) are missing for order=2,
            or if `x0` is not finite.
        NotImplementedError: If `order` not in {1, 2}.
    """
    if not np.isfinite(x0) or x0 == 0.0:
        raise ValueError("signed_log_derivatives_to_x requires finite x0 != 0.")
    dfdq = np.asarray(dfdq, dtype=float)
    if order == 1:
        return dfdq / x0
    elif order == 2:
        if d2fdq2 is None:
            raise ValueError("order=2 conversion requires d2fdq2.")
        d2fdq2 = np.asarray(d2fdq2, dtype=float)
        return (d2fdq2 - dfdq) / (x0 ** 2)
    raise NotImplementedError("signed_log_derivatives_to_x supports orders 1 and 2.")


def sqrt_domain_forward(x0: float, sign: Optional[float] = None) -> float:
    """Compute the internal domain coordinate u0 for the sqrt-domain map.

    This method computes the internal domain coordinate u0 satisfying x0 = s * u0^2,
    where s = sign(x0) unless `sign` is provided explicitly. At x0 == 0, u0 = 0.

    Args:
        x0: Expansion point in physical coordinates (can be zero).
        sign: Domain sign (optional). If None, infer s = sign(x0).
            - For x ≥ 0 domain, pass sign=+1 explicitly when x0 == 0.
            - For x ≤ 0 domain, pass sign=-1 explicitly when x0 == 0.

    Returns:
        u0 satisfying x0 = s * u0^2. For x0 == 0, returns 0.0.

    Raises:
        ValueError: If ``x0 == 0`` and ``sign`` is ``None`` (ambiguous domain)
           or if ``x0`` is not finite.
    """
    if not np.isfinite(x0):
        raise ValueError("sqrt_domain_forward requires finite x0.")
    if x0 == 0.0 and sign is None:
        raise ValueError("At x0=0 you must pass sign=+1 (x≥0) or -1 (x≤0).")
    s = _normalize_sign(sign) if sign is not None else (1.0 if x0 > 0.0 else -1.0)
    # Here we guarantee that x0 and s are consistent.
    if x0 != 0.0 and np.sign(x0) != s:
        raise ValueError(f"Inconsistent sign {s:+.0f} for x0={x0}.")
    u0 = 0.0 if x0 == 0.0 else float(np.sqrt(abs(x0)))
    return u0, s


def sqrt_to_physical(u: np.ndarray, sign: float) -> np.ndarray:
    """Map internal domain coordinate(s) to physical coordinate(s).

    This method maps internal coordinate(s) u to physical coordinate(s) x
    using the relation x = sign * u^2.

    Args:
        u: Internal coordinate(s).
        sign: Domain sign (+1 for x ≥ 0, -1 for x ≤ 0).

    Returns:
        Physical coordinate(s) x = sign * u^2.

    Raises:
        ValueError: If `sign` is not +1 or -1, or if `u` contains non-finite values.
    """
    u = np.asarray(u, dtype=float)
    _require_finite("u", u)
    s = _normalize_sign(sign)
    return s * (u**2)


def pullback_sqrt_at_zero(
    order: int,
    sign: float,
    g2: Optional[np.ndarray] = None,
    g4: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pull back derivatives at value x0=0 from u-space (sqrt-domain) to physical x.

    This method maps derivatives computed in the internal sqrt-domain coordinate u
    back to physical coordinates x at the expansion point x0=0 using the chain rule.

    Args:
        order: Derivative order to return (1 or 2).
        sign: Domain sign (+1 for x ≥ 0, -1 for x ≤ 0).
        g2: Second derivative of g with respect to u at u=0; required for order=1.
        g4: Fourth derivative of g with respect to u at u=0; required for order=2.

    Returns:
        The derivative(s) in physical coordinates at x0=0.

    Raises:
        ValueError: If required inputs (g2/g4) are missing for the requested order.
        NotImplementedError: If `order` not in {1, 2}.
    """
    s = _normalize_sign(sign)
    if order == 1:
        if g2 is None:
            raise ValueError("order=1 pullback requires g2 (g'' at u=0).")
        return np.asarray(g2, dtype=float) / (2.0 * s)
    if order == 2:
        if g4 is None:
            raise ValueError("order=2 pullback requires g4 (g'''' at u=0).")
        return np.asarray(g4, dtype=float) / (12.0 * s * s)
    raise NotImplementedError("pullback_sqrt_at_zero supports orders 1 and 2.")


def _normalize_sign(s: float) -> float:
    """Normalize a sign value to +1 or -1.

    Args:
        s: Input sign value.

    Returns:
        +1.0 if s >= 0.0, else -1.0.
    """
    return 1.0 if s >= 0.0 else -1.0


def _require_finite(name: str, arr: np.ndarray) -> None:
    """Raises a ``ValueError`` if an array contains any non-finite values.

    Args:
        name: Name of the array (for error message).
        arr: Array to check.

    Raises:
        ValueError: If arr contains any non-finite values.
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
