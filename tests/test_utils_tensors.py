"""Unit tests for ``derivkit.utils.tensors``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.tensors import (
    contract_tensor_with_vector,
    contract_tensor_with_vector_batch,
    gaussian_fourth_moment,
    symmetrize_tensor,
)


def test_contract_tensor_with_vector_once():
    """Tests contraction of one trailing tensor axis with a vector."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    vector = np.array([2.0, -1.0])

    out = contract_tensor_with_vector(tensor, vector)

    expected = np.array([0.0, 2.0])

    assert out.shape == (2,)
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_multiple_axes():
    """Tests contraction of multiple trailing tensor axes with a vector."""
    tensor = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    vector = np.array([1.0, 2.0])

    out = contract_tensor_with_vector(tensor, vector, n_axes=2)

    expected = np.array([27.0, 63.0])

    assert out.shape == (2,)
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_all_axes():
    """Tests contraction of all tensor axes returns a scalar array."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    vector = np.array([1.0, 2.0])

    out = contract_tensor_with_vector(tensor, vector, n_axes=2)

    expected = 27.0

    assert np.ndim(out) == 0
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_batch():
    """Tests that batched contractions preserve the batch and free tensor axes."""
    tensor = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    vectors = np.array(
        [
            [1.0, 2.0],
            [2.0, -1.0],
        ]
    )

    out = contract_tensor_with_vector_batch(tensor, vectors, n_axes=2)

    expected = np.array(
        [
            [27.0, 63.0],
            [-2.0, 2.0],
        ]
    )

    assert out.shape == (2, 2)
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_batch_multidimensional():
    """Tests that contraction preserves multiple leading vector batch dimensions."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    vectors = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [2.0, -1.0]],
    ])

    out = contract_tensor_with_vector_batch(tensor, vectors)

    expected = np.array([
        [[1.0, 3.0], [2.0, 4.0]],
        [[3.0, 7.0], [0.0, 2.0]],
    ])

    assert out.shape == (2, 2, 2)
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_preserves_leading_tensor_axes():
    """Tests that contraction preserves multiple leading tensor axes."""
    tensor = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    vector = np.array([1.0, 2.0])

    out = contract_tensor_with_vector(tensor, vector)

    expected = np.array([
        [5.0, 11.0],
        [17.0, 23.0],
    ])

    assert out.shape == (2, 2)
    assert_allclose(out, expected)


def test_contract_tensor_with_vector_n_zero_returns_tensor():
    """Tests that zero contractions return the original tensor values."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    vector = np.array([1.0, 2.0])

    out = contract_tensor_with_vector(tensor, vector, n_axes=0)

    assert out.shape == tensor.shape
    assert_allclose(out, tensor)


def test_contract_tensor_with_vector_invalid_n_raises():
    """Tests that invalid numbers of contracted axes raise ValueError."""
    tensor = np.ones((2, 2, 2))
    vector = np.ones(2)

    with pytest.raises(ValueError, match=r"0 <= n_axes <= tensor.ndim"):
        contract_tensor_with_vector(tensor, vector, n_axes=-1)

    with pytest.raises(ValueError, match=r"0 <= n_axes <= tensor.ndim"):
        contract_tensor_with_vector(tensor, vector, n_axes=4)


def test_contract_tensor_with_vector_incompatible_shapes_raises():
    """Tests that incompatible tensor and vector dimensions raise ValueError."""
    tensor = np.ones((2, 2, 2))
    vector = np.ones(3)

    with pytest.raises(
        ValueError,
        match="Contracted tensor dimensions must match the final vector dimension",
    ):
        contract_tensor_with_vector(tensor, vector)


def test_symmetrize_tensor_matrix():
    """Tests that rank-two tensor symmetrization averages transposed entries."""
    tensor = np.array([[1.0, 2.0], [4.0, 3.0]])

    out = symmetrize_tensor(tensor)

    expected = np.array([[1.0, 3.0], [3.0, 3.0]])

    assert out.shape == tensor.shape
    assert_allclose(out, expected)
    assert_allclose(out, out.T)


def test_symmetrize_tensor_rank_three():
    """Tests rank-three tensor symmetrization against known averaged entries."""
    tensor = np.zeros((2, 2, 2))
    tensor[0, 0, 1] = 3.0
    tensor[0, 1, 0] = 6.0
    tensor[1, 0, 0] = 9.0
    tensor[0, 1, 1] = 12.0
    tensor[1, 0, 1] = 15.0
    tensor[1, 1, 0] = 18.0

    out = symmetrize_tensor(tensor)

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 1] = 6.0
    expected[0, 1, 0] = 6.0
    expected[1, 0, 0] = 6.0
    expected[0, 1, 1] = 15.0
    expected[1, 0, 1] = 15.0
    expected[1, 1, 0] = 15.0

    assert out.shape == tensor.shape
    assert_allclose(out, expected)


def test_symmetrize_tensor_rank_four():
    """Tests rank-four tensor symmetrization over a known permutation orbit."""
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 1] = 4.0
    tensor[0, 0, 1, 0] = 8.0
    tensor[0, 1, 0, 0] = 12.0
    tensor[1, 0, 0, 0] = 16.0

    out = symmetrize_tensor(tensor)

    assert_allclose(out[0, 0, 0, 1], 10.0)
    assert_allclose(out[0, 0, 1, 0], 10.0)
    assert_allclose(out[0, 1, 0, 0], 10.0)
    assert_allclose(out[1, 0, 0, 0], 10.0)


def test_symmetrize_tensor_preserves_symmetric_tensor():
    """Tests that a symmetric tensor is unchanged when symmetrized."""
    tensor = np.ones((2, 2, 2))
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = 3.0

    out = symmetrize_tensor(tensor)

    assert_allclose(out, tensor)


def test_gaussian_fourth_moment_one_dimensional():
    """Tests the Gaussian fourth moment in one dimension."""
    cov = np.array([[2.0]])

    out = gaussian_fourth_moment(cov)

    expected = np.array([[[[12.0]]]])

    assert out.shape == (1, 1, 1, 1)
    assert_allclose(out, expected)


def test_gaussian_fourth_moment_diagonal_covariance():
    """Tests known fourth moments for independent Gaussian variables."""
    cov = np.diag([2.0, 3.0])

    out = gaussian_fourth_moment(cov)

    assert out.shape == (2, 2, 2, 2)
    assert_allclose(out[0, 0, 0, 0], 12.0)
    assert_allclose(out[1, 1, 1, 1], 27.0)
    assert_allclose(out[0, 0, 1, 1], 6.0)
    assert_allclose(out[0, 1, 0, 1], 6.0)
    assert_allclose(out[0, 0, 0, 1], 0.0)


def test_gaussian_fourth_moment_correlated_covariance():
    """Tests known fourth moments for correlated Gaussian variables."""
    cov = np.array([[2.0, 0.5], [0.5, 3.0]])

    out = gaussian_fourth_moment(cov)

    assert out.shape == (2, 2, 2, 2)
    assert_allclose(out[0, 0, 0, 0], 12.0)
    assert_allclose(out[1, 1, 1, 1], 27.0)
    assert_allclose(out[0, 0, 1, 1], 6.5)
    assert_allclose(out[0, 0, 0, 1], 3.0)
    assert_allclose(out[0, 1, 1, 1], 4.5)


def test_gaussian_fourth_moment_is_fully_symmetric():
    """Tests that the Gaussian fourth-moment tensor is fully symmetric."""
    cov = np.array([[2.0, 0.5], [0.5, 3.0]])

    out = gaussian_fourth_moment(cov)

    assert_allclose(out[0, 0, 1, 1], out[0, 1, 0, 1])
    assert_allclose(out[0, 0, 1, 1], out[1, 0, 1, 0])
    assert_allclose(out[0, 0, 0, 1], out[1, 0, 0, 0])
    assert_allclose(out[0, 1, 1, 1], out[1, 0, 1, 1])


def test_contract_tensor_with_vector_scalar_vector_raises():
    """Tests that a scalar contraction vector raises ValueError."""
    tensor = np.ones((2, 2))
    vector = np.array(2.0)

    with pytest.raises(ValueError, match="vector must be one-dimensional"):
        contract_tensor_with_vector(tensor, vector)


def test_symmetrize_tensor_unequal_dimensions_raises():
    """Tests that tensors with unequal axis dimensions raise ValueError."""
    tensor = np.ones((2, 3, 2))

    with pytest.raises(
        ValueError,
        match="All tensor dimensions must have equal size for symmetrization",
    ):
        symmetrize_tensor(tensor)


@pytest.mark.parametrize(
    "cov",
    [
        np.ones(3),
        np.ones((2, 3)),
    ],
)
def test_gaussian_fourth_moment_non_square_covariance_raises(cov):
    """Tests that non-matrix and non-square covariances raise ValueError."""
    with pytest.raises(ValueError, match="cov must be a square matrix"):
        gaussian_fourth_moment(cov)


@pytest.mark.parametrize("vector", [
    np.ones((2, 2)),
    np.ones((2, 2, 2)),
])
def test_contract_tensor_with_vector_multidimensional_vector_raises(vector):
    """Tests that the single-vector function rejects batched vectors."""
    tensor = np.ones((2, 2))

    with pytest.raises(ValueError, match="vector must be one-dimensional"):
        contract_tensor_with_vector(tensor, vector)


@pytest.mark.parametrize("vectors", [
    np.array(2.0),
    np.array([1.0, 2.0]),
])
def test_contract_tensor_with_vector_batch_invalid_shape_raises(vectors):
    """Tests that the batch function requires at least two dimensions."""
    tensor = np.ones((2, 2))

    with pytest.raises(ValueError, match="vectors must have at least two dimensions"):
        contract_tensor_with_vector_batch(tensor, vectors)


@pytest.mark.parametrize("n_axes", [1.5, "2", None, True, np.bool_(False), [1], np.array([1])])
def test_contract_tensor_with_vector_invalid_n_type_raises(n_axes):
    """Tests that non-integer contraction orders raise TypeError."""
    tensor = np.ones((2, 2))
    vector = np.ones(2)

    with pytest.raises(TypeError, match="n_axes must be an integer"):
        contract_tensor_with_vector(tensor, vector, n_axes=n_axes)


@pytest.mark.parametrize("n_axes", [1.5, "2", None, True, np.bool_(False), [1], np.array([1])])
def test_contract_tensor_with_vector_batch_invalid_n_type_raises(n_axes):
    """Tests that non-integer batch contraction orders raise TypeError."""
    tensor = np.ones((2, 2))
    vectors = np.ones((3, 2))

    with pytest.raises(TypeError, match="n_axes must be an integer"):
        contract_tensor_with_vector_batch(tensor, vectors, n_axes=n_axes)


def test_contract_tensor_with_vector_batch_incompatible_shapes_raises():
    """Tests that incompatible batched vectors raise ValueError."""
    tensor = np.ones((2, 2))
    vectors = np.ones((3, 4))

    with pytest.raises(
        ValueError,
        match="Contracted tensor dimensions must match the final vector dimension",
    ):
        contract_tensor_with_vector_batch(tensor, vectors)
