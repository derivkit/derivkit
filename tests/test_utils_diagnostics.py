"""Tests for ``derivkit.utils.diagnostics``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.utils.diagnostics import (
    dali_diagnostics,
    fisher_diagnostics,
    matrix_diagnostics,
)


def test_matrix_diagnostics_diagonal_positive_definite():
    """Tests eigenvalues, rank, and conditioning for a positive definite matrix."""
    matrix = np.diag([2.0, 8.0])

    result = matrix_diagnostics(matrix)

    assert_allclose(result["eigenvalues"], [2.0, 8.0])
    assert_allclose(result["eigenvectors"], np.eye(2))
    assert result["condition_number"] == pytest.approx(4.0)
    assert result["rank"] == 2
    assert result["is_positive_semidefinite"]


def test_matrix_diagnostics_rotated_matrix():
    """Tests that eigenvectors reconstruct a symmetric non-diagonal matrix."""
    matrix = np.array([[3.0, 1.0], [1.0, 3.0]])

    result = matrix_diagnostics(matrix)

    assert_allclose(result["eigenvalues"], [2.0, 4.0])
    vectors = result["eigenvectors"]
    reconstructed = vectors @ np.diag(result["eigenvalues"]) @ vectors.T
    assert_allclose(reconstructed, matrix)
    assert_allclose(vectors.T @ vectors, np.eye(2), atol=1e-14)
    assert result["condition_number"] == pytest.approx(2.0)


def test_matrix_diagnostics_singular_matrix():
    """Tests that a singular matrix has reduced rank and infinite conditioning."""
    matrix = np.diag([4.0, 0.0])

    result = matrix_diagnostics(matrix)

    assert_allclose(result["eigenvalues"], [0.0, 4.0])
    assert result["rank"] == 1
    assert np.isinf(result["condition_number"])
    assert result["is_positive_semidefinite"]


def test_matrix_diagnostics_zero_matrix():
    """Tests that a zero matrix has rank zero and infinite conditioning."""
    result = matrix_diagnostics(np.zeros((3, 3)))

    assert_allclose(result["eigenvalues"], np.zeros(3))
    assert result["rank"] == 0
    assert np.isinf(result["condition_number"])
    assert result["is_positive_semidefinite"]


def test_matrix_diagnostics_indefinite_matrix():
    """Tests that a negative eigenvalue identifies an indefinite matrix."""
    matrix = np.diag([-2.0, 4.0])

    result = matrix_diagnostics(matrix)

    assert_allclose(result["eigenvalues"], [-2.0, 4.0])
    assert result["rank"] == 2
    assert result["condition_number"] == pytest.approx(2.0)
    assert not result["is_positive_semidefinite"]


def test_matrix_diagnostics_negative_semidefinite_matrix():
    """Tests that a negative semidefinite matrix is not positive semidefinite."""
    matrix = np.diag([-3.0, 0.0])

    result = matrix_diagnostics(matrix)

    assert result["rank"] == 1
    assert np.isinf(result["condition_number"])
    assert not result["is_positive_semidefinite"]


def test_matrix_diagnostics_rank_tolerance():
    """Tests that the relative eigenvalue tolerance controls numerical rank."""
    matrix = np.diag([1.0, 1e-8])

    strict = matrix_diagnostics(matrix, rtol=1e-10)
    loose = matrix_diagnostics(matrix, rtol=1e-6)

    assert strict["rank"] == 2
    assert strict["condition_number"] == pytest.approx(1e8)
    assert loose["rank"] == 1
    assert np.isinf(loose["condition_number"])


def test_matrix_diagnostics_psd_tolerance():
    """Tests that small negative eigenvalues are classified using the tolerance."""
    matrix = np.diag([-1e-8, 1.0])

    strict = matrix_diagnostics(matrix, rtol=1e-10)
    loose = matrix_diagnostics(matrix, rtol=1e-6)

    assert not strict["is_positive_semidefinite"]
    assert loose["is_positive_semidefinite"]


def test_matrix_diagnostics_single_parameter():
    """Tests that diagnostics support a one-dimensional parameter space."""
    result = matrix_diagnostics([[5.0]])

    assert_allclose(result["eigenvalues"], [5.0])
    assert result["rank"] == 1
    assert result["condition_number"] == pytest.approx(1.0)
    assert result["is_positive_semidefinite"]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.array([]), "must be 2D"),
        (np.array([1.0, 2.0]), "must be 2D"),
        (np.zeros((0, 0)), "must be nonempty"),
        (np.ones((2, 3)), "must be square"),
        (np.ones((2, 2, 2)), "must be 2D"),
    ],
)
def test_matrix_diagnostics_invalid_shape_raises(matrix, message):
    """Tests that empty, nonsquare, or non-2D matrices raise ValueError."""
    with pytest.raises(ValueError, match=message):
        matrix_diagnostics(matrix)


def test_matrix_diagnostics_asymmetric_matrix_raises():
    """Tests that a nonsymmetric matrix raises ValueError."""
    matrix = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        matrix_diagnostics(matrix)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matrix_diagnostics_nonfinite_matrix_raises(value):
    """Tests that nonfinite matrix entries raise ValueError."""
    matrix = np.diag([1.0, value])

    with pytest.raises(ValueError, match="finite"):
        matrix_diagnostics(matrix)


@pytest.mark.parametrize(
    "rtol",
    [-1.0, np.nan, np.inf, True, "invalid", [1e-6]],
)
def test_matrix_diagnostics_invalid_tolerance_raises(rtol):
    """Tests that invalid relative tolerances raise ValueError."""
    with pytest.raises(ValueError, match="rtol"):
        matrix_diagnostics(np.eye(2), rtol=rtol)


def test_fisher_diagnostics_without_jacobian():
    """Tests that Fisher diagnostics reduce to symmetric matrix diagnostics."""
    fisher = np.diag([2.0, 8.0])

    result = fisher_diagnostics(fisher)

    assert_allclose(result["eigenvalues"], [2.0, 8.0])
    assert result["rank"] == 2
    assert result["condition_number"] == pytest.approx(4.0)
    assert "jacobian_singular_values" not in result
    assert "jacobian_rank" not in result
    assert "jacobian_condition_number" not in result


def test_fisher_diagnostics_diagonal_jacobian():
    """Tests noise-weighted Jacobian singular values against an analytic result."""
    jacobian = np.diag([2.0, 6.0])
    cov = np.diag([4.0, 9.0])
    fisher = np.diag([1.0, 4.0])

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    assert_allclose(result["eigenvalues"], [1.0, 4.0])
    assert_allclose(result["jacobian_singular_values"], [2.0, 1.0])
    assert result["jacobian_rank"] == 2
    assert result["jacobian_condition_number"] == pytest.approx(2.0)
    assert result["condition_number"] == pytest.approx(4.0)


def test_fisher_diagnostics_correlated_covariance():
    """Tests Jacobian whitening with a correlated observational covariance."""
    jacobian = np.eye(2)
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    fisher = np.array([[2.0, -1.0], [-1.0, 2.0]]) / 3.0

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    expected_singular_values = [1.0, 1.0 / np.sqrt(3.0)]
    assert_allclose(
        result["jacobian_singular_values"],
        expected_singular_values,
    )
    assert result["jacobian_rank"] == 2
    assert result["jacobian_condition_number"] == pytest.approx(np.sqrt(3.0))
    assert result["condition_number"] == pytest.approx(3.0)


def test_fisher_diagnostics_rank_deficient_jacobian():
    """Tests that an unconstrained parameter produces a singular Fisher matrix."""
    jacobian = np.array([[1.0, 0.0], [2.0, 0.0]])
    cov = np.eye(2)
    fisher = np.diag([5.0, 0.0])

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    assert_allclose(result["jacobian_singular_values"], [np.sqrt(5.0), 0.0])
    assert result["jacobian_rank"] == 1
    assert np.isinf(result["jacobian_condition_number"])
    assert result["rank"] == 1
    assert np.isinf(result["condition_number"])


def test_fisher_diagnostics_singular_covariance():
    """Tests that zero covariance modes are discarded by the pseudoinverse."""
    jacobian = np.eye(2)
    cov = np.diag([4.0, 0.0])
    fisher = np.diag([0.25, 0.0])

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    assert_allclose(result["jacobian_singular_values"], [0.5, 0.0])
    assert result["jacobian_rank"] == 1
    assert np.isinf(result["jacobian_condition_number"])


def test_fisher_diagnostics_covariance_cutoff():
    """Tests that the covariance cutoff discards sufficiently small eigenmodes."""
    jacobian = np.eye(2)
    cov = np.diag([1.0, 1e-8])
    fisher = np.diag([1.0, 0.0])

    result = fisher_diagnostics(
        fisher,
        jacobian=jacobian,
        cov=cov,
        covariance_rcond=1e-6,
    )

    assert_allclose(result["jacobian_singular_values"], [1.0, 0.0])
    assert result["jacobian_rank"] == 1


def test_fisher_diagnostics_rectangular_jacobian():
    """Tests sensitivity diagnostics with more observables than parameters."""
    jacobian = np.array([[3.0], [4.0]])
    cov = np.eye(2)
    fisher = np.array([[25.0]])

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    assert_allclose(result["jacobian_singular_values"], [5.0])
    assert result["jacobian_rank"] == 1
    assert result["jacobian_condition_number"] == pytest.approx(1.0)


def test_fisher_diagnostics_zero_jacobian():
    """Tests that a zero Jacobian has rank zero and infinite conditioning."""
    jacobian = np.zeros((2, 2))
    cov = np.eye(2)
    fisher = np.zeros((2, 2))

    result = fisher_diagnostics(fisher, jacobian=jacobian, cov=cov)

    assert_allclose(result["jacobian_singular_values"], [0.0, 0.0])
    assert result["jacobian_rank"] == 0
    assert np.isinf(result["jacobian_condition_number"])


def test_fisher_diagnostics_jacobian_rank_tolerance():
    """Tests that Jacobian rank responds to the requested relative tolerance."""
    jacobian = np.diag([1.0, 1e-8])
    cov = np.eye(2)
    fisher = np.diag([1.0, 1e-16])

    strict = fisher_diagnostics(
        fisher, jacobian=jacobian, cov=cov, rtol=1e-10
    )
    loose = fisher_diagnostics(
        fisher, jacobian=jacobian, cov=cov, rtol=1e-6
    )

    assert strict["jacobian_rank"] == 2
    assert loose["jacobian_rank"] == 1
    assert np.isinf(loose["jacobian_condition_number"])


def test_fisher_diagnostics_missing_covariance_raises():
    """Tests that supplying a Jacobian without covariance raises ValueError."""
    with pytest.raises(ValueError, match="together"):
        fisher_diagnostics(np.eye(2), jacobian=np.eye(2))


def test_fisher_diagnostics_missing_jacobian_raises():
    """Tests that supplying covariance without a Jacobian raises ValueError."""
    with pytest.raises(ValueError, match="together"):
        fisher_diagnostics(np.eye(2), cov=np.eye(2))


def test_fisher_diagnostics_invalid_jacobian_shape_raises():
    """Tests that a non-2D Jacobian raises ValueError."""
    with pytest.raises(ValueError, match="nonempty 2D"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.ones(2),
            cov=np.eye(2),
        )


def test_fisher_diagnostics_parameter_dimension_mismatch_raises():
    """Tests that Fisher and Jacobian parameter dimensions must agree."""
    with pytest.raises(ValueError, match="parameter dimension"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.ones((3, 1)),
            cov=np.eye(3),
        )


def test_fisher_diagnostics_observable_dimension_mismatch_raises():
    """Tests that covariance and Jacobian observable dimensions must agree."""
    with pytest.raises(ValueError, match="observable dimension"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.ones((3, 2)),
            cov=np.eye(2),
        )


def test_fisher_diagnostics_nonfinite_jacobian_raises():
    """Tests that a nonfinite Jacobian raises ValueError."""
    jacobian = np.array([[1.0, np.nan], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        fisher_diagnostics(np.eye(2), jacobian=jacobian, cov=np.eye(2))


def test_fisher_diagnostics_negative_covariance_raises():
    """Tests that a covariance with a negative eigenvalue raises ValueError."""
    cov = np.diag([1.0, -1.0])

    with pytest.raises(ValueError, match="positive semidefinite"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.eye(2),
            cov=cov,
        )


def test_fisher_diagnostics_asymmetric_covariance_raises():
    """Tests that an asymmetric covariance raises ValueError."""
    cov = np.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.eye(2),
            cov=cov,
        )


@pytest.mark.parametrize("rcond", [-1.0, np.nan, np.inf, True])
def test_fisher_diagnostics_invalid_covariance_cutoff_raises(rcond):
    """Tests that invalid covariance cutoffs raise ValueError."""
    with pytest.raises(ValueError, match="rtol"):
        fisher_diagnostics(
            np.eye(2),
            jacobian=np.eye(2),
            cov=np.eye(2),
            covariance_rcond=rcond,
        )


def test_dali_diagnostics_linear_model():
    """Tests that a linear model has only a first-order response contribution."""
    derivatives = {1: np.array([[2.0], [4.0]])}
    cov = np.diag([4.0, 16.0])
    displacement = np.array([0.5])

    result = dali_diagnostics(derivatives, cov, displacement)

    expected = 1.0 / np.sqrt(2.0)

    assert result["response_norms"][1] == pytest.approx(expected)
    assert result["linear_response_norm"] == pytest.approx(expected)
    assert result["total_response_norm"] == pytest.approx(expected)
    assert result["relative_response_strength"] == {}


def test_dali_diagnostics_quadratic_model():
    """Tests that the quadratic Taylor contribution includes the factor 1/2."""
    derivatives = {
        1: np.array([[2.0]]),
        2: np.array([[[6.0]]]),
    }
    cov = np.array([[4.0]])
    displacement = np.array([0.5])

    result = dali_diagnostics(derivatives, cov, displacement)

    # r1 = 2 * 0.5 = 1, r2 = 6 * 0.5**2 / 2 = 0.75
    # The covariance-weighted norms are |r1| / 2 and |r2| / 2.
    assert result["response_norms"][1] == pytest.approx(0.5)
    assert result["response_norms"][2] == pytest.approx(0.375)
    assert result["relative_response_strength"][2] == pytest.approx(0.75)
    assert result["total_response_norm"] == pytest.approx(0.875)


def test_dali_diagnostics_cubic_model():
    """Tests that the cubic Taylor contribution includes the factor 1/6."""
    derivatives = {
        1: np.array([[1.0]]),
        3: np.array([[[[6.0]]]]),
    }
    cov = np.eye(1)
    displacement = np.array([2.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    # r1 = 2, r3 = 6 * 2**3 / 6 = 8
    assert result["response_norms"][1] == pytest.approx(2.0)
    assert result["response_norms"][3] == pytest.approx(8.0)
    assert result["relative_response_strength"][3] == pytest.approx(4.0)
    assert result["total_response_norm"] == pytest.approx(10.0)


def test_dali_diagnostics_fourth_order_model():
    """Tests that the fourth-order Taylor contribution includes the factor 1/24."""
    derivatives = {
        1: np.array([[1.0]]),
        4: np.array([[[[[24.0]]]]]),
    }
    cov = np.eye(1)
    displacement = np.array([2.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    # r1 = 2, r4 = 24 * 2**4 / 24 = 16
    assert result["response_norms"][1] == pytest.approx(2.0)
    assert result["response_norms"][4] == pytest.approx(16.0)
    assert result["relative_response_strength"][4] == pytest.approx(8.0)
    assert result["total_response_norm"] == pytest.approx(18.0)


def test_dali_diagnostics_mixed_hessian():
    """Tests that mixed second derivatives are contracted with both parameters."""
    derivatives = {
        1: np.array([[1.0, 0.0]]),
        2: np.array([[[0.0, 2.0], [2.0, 0.0]]]),
    }
    cov = np.eye(1)
    displacement = np.array([1.0, 3.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    # r1 = 1, r2 = (2 * 1 * 3 + 2 * 3 * 1) / 2 = 6
    assert result["response_norms"][1] == pytest.approx(1.0)
    assert result["response_norms"][2] == pytest.approx(6.0)
    assert result["relative_response_strength"][2] == pytest.approx(6.0)
    assert result["total_response_norm"] == pytest.approx(7.0)


def test_dali_diagnostics_multiple_observables():
    """Tests the covariance-weighted response norm for multiple observables."""
    derivatives = {
        1: np.array([[2.0], [4.0]]),
        2: np.array([[[4.0]], [[8.0]]]),
    }
    cov = np.diag([4.0, 16.0])
    displacement = np.array([1.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    # r1 = [2, 4], r2 = [2, 4].
    # Each has squared weighted norm 2; their sum has squared norm 8.
    assert result["response_norms"][1] == pytest.approx(np.sqrt(2.0))
    assert result["response_norms"][2] == pytest.approx(np.sqrt(2.0))
    assert result["relative_response_strength"][2] == pytest.approx(1.0)
    assert result["total_response_norm"] == pytest.approx(np.sqrt(8.0))


def test_dali_diagnostics_response_cancellation():
    """Tests that the total response norm accounts for cancellation between orders."""
    derivatives = {
        1: np.array([[1.0]]),
        2: np.array([[[-2.0]]]),
    }
    cov = np.eye(1)
    displacement = np.array([1.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    assert result["response_norms"][1] == pytest.approx(1.0)
    assert result["response_norms"][2] == pytest.approx(1.0)
    assert result["relative_response_strength"][2] == pytest.approx(1.0)
    assert result["total_response_norm"] == pytest.approx(0.0)


def test_dali_diagnostics_singular_covariance():
    """Tests that response norms ignore discarded covariance eigenmodes."""
    derivatives = {
        1: np.array([[2.0], [4.0]]),
        2: np.array([[[4.0]], [[8.0]]]),
    }
    cov = np.diag([4.0, 0.0])
    displacement = np.array([1.0])

    result = dali_diagnostics(derivatives, cov, displacement)

    assert result["response_norms"][1] == pytest.approx(1.0)
    assert result["response_norms"][2] == pytest.approx(1.0)
    assert result["relative_response_strength"][2] == pytest.approx(1.0)
    assert result["total_response_norm"] == pytest.approx(2.0)


def test_dali_diagnostics_zero_displacement():
    """Tests that zero displacement gives zero response and undefined ratios."""
    derivatives = {
        1: np.array([[2.0]]),
        2: np.array([[[4.0]]]),
    }

    result = dali_diagnostics(derivatives, np.eye(1), [0.0])

    assert result["linear_response_norm"] == pytest.approx(0.0)
    assert result["total_response_norm"] == pytest.approx(0.0)
    assert result["response_norms"][2] == pytest.approx(0.0)
    assert np.isnan(result["relative_response_strength"][2])


def test_dali_diagnostics_zero_linear_response():
    """Tests that nonlinear response with zero linear response has infinite ratio."""
    derivatives = {
        1: np.array([[0.0]]),
        2: np.array([[[2.0]]]),
    }

    result = dali_diagnostics(derivatives, np.eye(1), [1.0])

    assert result["linear_response_norm"] == pytest.approx(0.0)
    assert result["response_norms"][2] == pytest.approx(1.0)
    assert np.isinf(result["relative_response_strength"][2])
    assert result["total_response_norm"] == pytest.approx(1.0)


def test_dali_diagnostics_zero_higher_order_response():
    """Tests that a vanishing higher-order response produces a zero ratio."""
    derivatives = {
        1: np.array([[1.0]]),
        2: np.array([[[0.0]]]),
    }

    result = dali_diagnostics(derivatives, np.eye(1), [1.0])

    assert result["relative_response_strength"][2] == pytest.approx(0.0)


def test_dali_diagnostics_displacement_dependence():
    """Tests that quadratic relative strength scales with displacement amplitude."""
    derivatives = {
        1: np.array([[1.0]]),
        2: np.array([[[2.0]]]),
    }
    cov = np.eye(1)

    small = dali_diagnostics(derivatives, cov, [0.5])
    large = dali_diagnostics(derivatives, cov, [1.0])

    assert small["relative_response_strength"][2] == pytest.approx(0.5)
    assert large["relative_response_strength"][2] == pytest.approx(1.0)


def test_dali_diagnostics_missing_first_derivative_raises():
    """Tests that derivative dictionaries must contain first-order derivatives."""
    with pytest.raises(ValueError, match="order 1"):
        dali_diagnostics({2: np.ones((1, 1, 1))}, np.eye(1), [1.0])


@pytest.mark.parametrize("order", [0, 5, 1.5, "2"])
def test_dali_diagnostics_invalid_derivative_order_raises(order):
    """Tests that unsupported derivative dictionary keys raise ValueError."""
    derivatives = {
        1: np.array([[1.0]]),
        order: np.array([[[1.0]]]),
    }

    with pytest.raises(ValueError, match="orders"):
        dali_diagnostics(derivatives, np.eye(1), [1.0])


def test_dali_diagnostics_boolean_derivative_order_raises():
    """Tests that boolean derivative orders are rejected."""
    derivatives = {True: np.array([[1.0]])}

    with pytest.raises(ValueError, match="orders"):
        dali_diagnostics(derivatives, np.eye(1), [1.0])


def test_dali_diagnostics_invalid_derivative_shape_raises():
    """Tests that derivative tensors must have the expected number of axes."""
    derivatives = {
        1: np.array([[1.0, 2.0]]),
        2: np.ones((1, 2, 3)),
    }

    with pytest.raises(ValueError, match="shape"):
        dali_diagnostics(derivatives, np.eye(1), [1.0, 1.0])


def test_dali_diagnostics_nonfinite_derivative_raises():
    """Tests that nonfinite derivative tensors raise ValueError."""
    derivatives = {
        1: np.array([[1.0]]),
        2: np.array([[[np.nan]]]),
    }

    with pytest.raises(ValueError, match="finite"):
        dali_diagnostics(derivatives, np.eye(1), [1.0])


def test_dali_diagnostics_invalid_displacement_shape_raises():
    """Tests that the displacement must be a nonempty one-dimensional array."""
    derivatives = {1: np.array([[1.0, 2.0]])}

    with pytest.raises(ValueError, match="nonempty 1D"):
        dali_diagnostics(derivatives, np.eye(1), [[1.0, 2.0]])


def test_dali_diagnostics_displacement_dimension_mismatch_raises():
    """Tests that displacement length must match the parameter dimension."""
    derivatives = {1: np.array([[1.0, 2.0]])}

    with pytest.raises(ValueError, match="displacement length"):
        dali_diagnostics(derivatives, np.eye(1), [1.0])


def test_dali_diagnostics_nonfinite_displacement_raises():
    """Tests that nonfinite parameter displacements raise ValueError."""
    derivatives = {1: np.array([[1.0]])}

    with pytest.raises(ValueError, match="finite"):
        dali_diagnostics(derivatives, np.eye(1), [np.inf])


def test_dali_diagnostics_covariance_dimension_mismatch_raises():
    """Tests that covariance shape must match the observable dimension."""
    derivatives = {1: np.array([[1.0], [2.0]])}

    with pytest.raises(ValueError, match="number of observables"):
        dali_diagnostics(derivatives, np.eye(1), [1.0])


def test_dali_diagnostics_negative_covariance_raises():
    """Tests that a covariance with negative eigenvalues raises ValueError."""
    derivatives = {1: np.array([[1.0], [2.0]])}
    cov = np.diag([1.0, -1.0])

    with pytest.raises(ValueError, match="positive semidefinite"):
        dali_diagnostics(derivatives, cov, [1.0])


def test_dali_diagnostics_invalid_covariance_cutoff_raises():
    """Tests that invalid covariance cutoffs raise ValueError."""
    derivatives = {1: np.array([[1.0]])}

    with pytest.raises(ValueError, match="rtol"):
        dali_diagnostics(
            derivatives,
            np.eye(1),
            [1.0],
            covariance_rcond=-1.0,
        )
