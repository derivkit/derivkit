"""Tensor algebra utilities."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np

from derivkit.utils.types import FloatArray

__all__ = [
    "contract_tensor",
    "gaussian_fourth_moment",
    "symmetrize_tensor",
]


def contract_tensor(
    tensor: FloatArray,
    vector: FloatArray,
    n: int = 1,
) -> FloatArray:
    """Contracts trailing tensor axes with copies of a vector.

    Args:
        tensor: Tensor whose trailing axes are contracted.
        vector: Vector or array of vectors to contract with the tensor. The
            final axis contains the parameter coordinates.
        n: Number of trailing tensor axes to contract.

    Returns:
        Tensor after contracting the last ``n`` axes with copies of ``vector``.

    Raises:
        ValueError: If ``vector`` is scalar, ``n`` is invalid, or the contracted
            dimensions are incompatible with the vector dimension.
    """
    tensor = np.asarray(tensor, dtype=float)
    vector = np.asarray(vector, dtype=float)

    if vector.ndim == 0:
        raise ValueError("vector must have at least one dimension.")

    if n < 0 or n > tensor.ndim:
        raise ValueError(
            f"n must satisfy 0 <= n <= tensor.ndim; got n={n} "
            f"for tensor.ndim={tensor.ndim}."
        )

    if n > 0 and any(size != vector.shape[-1] for size in tensor.shape[-n:]):
        raise ValueError(
            "Contracted tensor dimensions must match the final vector dimension."
        )

    batch_ndim = vector.ndim - 1
    batch_labels = list(range(batch_ndim))
    free_ndim = tensor.ndim - n
    free_labels = list(range(batch_ndim, batch_ndim + free_ndim))
    contracted_labels = list(
        range(batch_ndim + free_ndim, batch_ndim + free_ndim + n)
    )

    operands = [tensor, free_labels + contracted_labels]

    for label in contracted_labels:
        operands.extend([vector, batch_labels + [label]])

    return np.einsum(*operands, batch_labels + free_labels)


def symmetrize_tensor(tensor: FloatArray) -> FloatArray:
    """Symmetrizes a tensor over all axes.

    Args:
        tensor: Tensor to symmetrize.

    Returns:
        Tensor averaged over all permutations of its axes.

    Raises:
        ValueError: If the tensor axes do not all have equal size.
    """
    tensor = np.asarray(tensor, dtype=float)

    if tensor.ndim > 1 and len(set(tensor.shape)) != 1:
        raise ValueError(
            "All tensor dimensions must have equal size for symmetrization."
        )

    axis_permutations = permutations(range(tensor.ndim))

    return sum(
        (tensor.transpose(permutation) for permutation in axis_permutations),
        start=np.zeros_like(tensor),
    ) / factorial(tensor.ndim)


def gaussian_fourth_moment(cov: FloatArray) -> FloatArray:
    """Computes the centered fourth moment of a multivariate Gaussian.

    Args:
        cov: Gaussian covariance matrix.

    Returns:
        Rank-four tensor containing the centered Gaussian fourth moment.

    Raises:
        ValueError: If ``cov`` is not a square matrix.
    """
    cov = np.asarray(cov, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix.")

    return (
        np.einsum("ab,cd->abcd", cov, cov)
        + np.einsum("ac,bd->abcd", cov, cov)
        + np.einsum("ad,bc->abcd", cov, cov)
    )
