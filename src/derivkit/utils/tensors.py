
"""Tensor algebra utilities."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np

from derivkit.utils.types import FloatArray

__all__ = [
    "contract_tensor_with_vector",
    "contract_tensor_with_vector_batch",
    "gaussian_fourth_moment",
    "symmetrize_tensor",
]


def contract_tensor_with_vector(
    tensor: FloatArray,
    vector: FloatArray,
    n_axes: int = 1,
) -> FloatArray:
    """Contracts trailing tensor axes with repeated copies of a vector.

    Args:
        tensor: Tensor whose trailing axes are contracted.
        vector: One-dimensional vector of shape ``(d,)``.
        n_axes: Number of trailing tensor axes to contract.

    Returns:
        Tensor with the contracted axes removed and leading tensor axes
        preserved. If ``n_axes=0``, the tensor is unchanged.

    Raises:
        TypeError: If ``n_axes`` is not an integer.
        ValueError: If ``vector`` is not one-dimensional, ``n_axes`` is invalid,
            or the contracted dimensions do not match the vector dimension.
    """
    tensor = np.asarray(tensor, dtype=float)
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional.")

    if isinstance(n_axes, (bool, np.bool_)) or not isinstance(n_axes, (int, np.integer)):
        raise TypeError("n_axes must be an integer.")

    if n_axes < 0 or n_axes > tensor.ndim:
        raise ValueError(
            f"n_axes must satisfy 0 <= n_axes <= tensor.ndim; got n_axes={n_axes} "
            f"for tensor.ndim={tensor.ndim}."
        )

    if n_axes > 0 and any(size != vector.shape[-1] for size in tensor.shape[-n_axes:]):
        raise ValueError(
            "Contracted tensor dimensions must match the final vector dimension."
        )

    free_ndim = tensor.ndim - n_axes
    free_labels = list(range(free_ndim))
    contracted_labels = list(range(free_ndim, tensor.ndim))

    operands = [tensor, free_labels + contracted_labels]

    for label in contracted_labels:
        operands.extend([vector, [label]])

    return np.einsum(*operands, free_labels)


def contract_tensor_with_vector_batch(
    tensor: FloatArray,
    vectors: FloatArray,
    n_axes: int = 1,
) -> FloatArray:
    """Contracts trailing tensor axes independently with a batch of vectors.

    Each vector in the batch is independently contracted with the same tensor
    along ``n_axes`` trailing axes. Leading batch dimensions are preserved.

    Args:
        tensor: Tensor whose trailing axes are contracted.
        vectors: Batch of vectors of shape ``(..., d)``.
        n_axes: Number of trailing tensor axes to contract.

    Returns:
        Tensor with the contracted axes removed and leading vector batch
        dimensions preserved. If ``n_axes=0``, the tensor is unchanged apart
        from broadcasting over the batch dimensions.

    Raises:
        TypeError: If ``n_axes`` is not an integer.
        ValueError: If ``vectors`` has fewer than two dimensions, ``n_axes`` is
            invalid, or the contracted dimensions do not match the vector
            dimension.
    """
    tensor = np.asarray(tensor, dtype=float)
    vectors = np.asarray(vectors, dtype=float)

    if vectors.ndim < 2:
        raise ValueError("vectors must have at least two dimensions.")

    if isinstance(n_axes, (bool, np.bool_)) or not isinstance(n_axes, (int, np.integer)):
        raise TypeError("n_axes must be an integer.")

    if n_axes < 0 or n_axes > tensor.ndim:
        raise ValueError(
            f"n_axes must satisfy 0 <= n_axes <= tensor.ndim; got n_axes={n_axes} "
            f"for tensor.ndim={tensor.ndim}."
        )

    if n_axes > 0 and any(size != vectors.shape[-1] for size in tensor.shape[-n_axes:]):
        raise ValueError(
            "Contracted tensor dimensions must match the final vector dimension."
        )

    batch_ndim = vectors.ndim - 1
    batch_labels = list(range(batch_ndim))
    free_ndim = tensor.ndim - n_axes
    free_labels = list(range(batch_ndim, batch_ndim + free_ndim))
    contracted_labels = list(
        range(batch_ndim + free_ndim, batch_ndim + free_ndim + n_axes)
    )

    operands = [tensor, free_labels + contracted_labels]

    for label in contracted_labels:
        operands.extend([vectors, batch_labels + [label]])

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

