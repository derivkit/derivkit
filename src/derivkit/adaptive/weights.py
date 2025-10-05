"""Weighting utilities for adaptive polynomial fits."""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["inverse_distance_weights"]


def inverse_distance_weights(
    x_vals: Iterable[float] | np.ndarray,
    x0: float,
    eps_frac: float = 1e-3,
    eps_floor: float = 1e-9,
) -> np.ndarray:
    """Scale-aware inverse-distance weights centered at ``x0``.

    Each sample gets a weight based on how far it is from the center point ``x0``.
    To avoid division by zero or overly large weights, we add a small distance
    floor. This floor is chosen as either a fraction of the total spread of points
    or a fixed minimum, whichever is larger. The result is that points closer to
    ``x0`` have larger weights, while those farther away have smaller weights.

    Args:
        x_vals: Sample locations.
        x0: Center location.
        eps_frac: Fraction of the span used to set the distance floor.
        eps_floor: Absolute lower bound for the distance floor.

    Returns:
        np.ndarray: Unnormalized weights with shape ``(n_points,)``.
    """
    distances = np.abs(np.asarray(x_vals, dtype=float) - float(x0))  # distances
    span = float(np.max(distances)) if distances.size else 0.0  # span
    eps = max(eps_frac * span, eps_floor)
    return 1.0 / (distances + eps)
