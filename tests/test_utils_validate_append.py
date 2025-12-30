"""Unit tests for derivkit.utils.validate module."""

import numpy as np
import pytest

from derivkit.utils.validate import (
    check_scalar_valued,
    flatten_matrix_c_order,
    require_callable,
    resolve_covariance_input,
    validate_covariance_matrix_shape,
    validate_dali_shapes,
    validate_fisher_shapes,
    validate_square_matrix,
    validate_tabulated_xy,
)


def vector_function(params: np.ndarray) -> np.ndarray:
    """A vector-valued function for testing."""
    return np.array([params[0], params[1]], dtype=float)


def cov_fn(_: np.ndarray) -> np.ndarray:
    """A covariance function returning a 2x2 matrix."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)


def cov_fn_3by3(_: np.ndarray) -> np.ndarray:
    """A covariance function returning a 3x3 matrix."""
    return np.eye(3, dtype=float)

def lin_func(x: float) -> float:
    """A linear function for testing require_callable."""
    return x


def test_check_scalar_valued_accepts_scalar_return() -> None:
    """Tests that check_scalar_valued accepts scalar-valued functions."""
    def scalar_function(params: np.ndarray) -> float:
        return float(params[0] + 2.0 * params[1])

    theta0 = np.array([0.1, -0.2], dtype=float)

    check_scalar_valued(scalar_function, theta0, i=0, n_workers=1)


def test_check_scalar_valued_raises_for_vector_return() -> None:
    """Tests that check_scalar_valued raises for vector-valued functions."""
    theta0 = np.array([0.1, -0.2], dtype=float)
    with pytest.raises(TypeError, match="expects a scalar-valued function"):
        check_scalar_valued(vector_function, theta0, i=0, n_workers=1)


def test_validate_tabulated_xy_accepts_strictly_increasing_x_and_matching_y() -> None:
    """Tests that validate_tabulated_xy accepts valid x and y arrays."""
    x = np.array([0.0, 0.5, 1.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    x_out, y_out = validate_tabulated_xy(x, y)

    assert x_out.shape == (3,)
    assert y_out.shape == (3,)
    assert np.all(np.diff(x_out) > 0)


def test_validate_tabulated_xy_accepts_vector_valued_y() -> None:
    """Tests that validate_tabulated_xy accepts vector-valued y arrays."""
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)

    x_out, y_out = validate_tabulated_xy(x, y)

    assert x_out.shape == (3,)
    assert y_out.shape == (3, 2)


def test_validate_tabulated_xy_rejects_non_1d_x() -> None:
    """Tests that validate_tabulated_xy rejects non-1D x arrays."""
    x = np.zeros((2, 2), dtype=float)
    y = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="x must be 1D"):
        validate_tabulated_xy(x, y)


def test_validate_tabulated_xy_rejects_length_mismatch() -> None:
    """Tests that validate_tabulated_xy rejects x and y arrays of different lengths."""
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="same length"):
        validate_tabulated_xy(x, y)


def test_validate_tabulated_xy_rejects_non_increasing_x() -> None:
    """Tests that validate_tabulated_xy rejects non-increasing x arrays."""
    x = np.array([0.0, 1.0, 1.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_tabulated_xy(x, y)


def test_validate_fisher_shapes_accepts_valid_shapes() -> None:
    """Tests that validate_fisher_shapes accepts valid theta0 and fisher shapes."""
    theta0 = np.array([0.1, 0.2, 0.3], dtype=float)
    fisher = np.eye(3, dtype=float)

    validate_fisher_shapes(theta0, fisher)


def test_validate_fisher_shapes_rejects_non_1d_theta0() -> None:
    """Tests that validate_fisher_shapes rejects non-1D theta0 arrays."""
    theta0 = np.array([[0.1, 0.2]], dtype=float)  # 2D
    fisher = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="theta0 must be 1D"):
        validate_fisher_shapes(theta0, fisher)


def test_validate_fisher_shapes_rejects_wrong_fisher_shape() -> None:
    """Tests that validate_fisher_shapes rejects fisher matrices with wrong shape."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(3, dtype=float)

    with pytest.raises(ValueError, match="fisher must have shape"):
        validate_fisher_shapes(theta0, fisher)


def test_validate_dali_shapes_accepts_valid_shapes_with_h() -> None:
    """Tests that validate_dali_shapes accepts valid theta0, fisher, g_tensor, and h_tensor shapes."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    g_tensor = np.zeros((2, 2, 2), dtype=float)
    h_tensor = np.zeros((2, 2, 2, 2), dtype=float)

    validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)


def test_validate_dali_shapes_accepts_valid_shapes_without_h() -> None:
    """Tests that validate_dali_shapes accepts valid theta0, fisher, and g_tensor shapes when h_tensor is None."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    g_tensor = np.zeros((2, 2, 2), dtype=float)

    validate_dali_shapes(theta0, fisher, g_tensor, None)


def test_validate_dali_shapes_rejects_wrong_g_shape() -> None:
    """Tests that validate_dali_shapes rejects g_tensor with wrong shape."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    g_tensor = np.zeros((2, 2), dtype=float)  # wrong ndim
    h_tensor = np.zeros((2, 2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="g_tensor must have shape"):
        validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)


def test_validate_dali_shapes_rejects_wrong_h_shape() -> None:
    """Tests that validate_dali_shapes rejects h_tensor with wrong shape."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    g_tensor = np.zeros((2, 2, 2), dtype=float)
    h_tensor = np.zeros((2, 2, 2), dtype=float)  # wrong ndim

    with pytest.raises(ValueError, match="h_tensor must have shape"):
        validate_dali_shapes(theta0, fisher, g_tensor, h_tensor)


def test_validate_square_matrix_accepts_square_2d() -> None:
    """Tests that validate_square_matrix accepts square 2D arrays."""
    mat = np.eye(3, dtype=float)
    out = validate_square_matrix(mat, name="thing")
    assert out.shape == (3, 3)


def test_validate_square_matrix_rejects_non_2d() -> None:
    """Tests that validate_square_matrix rejects non-2D arrays."""
    mat = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="must be 2D"):
        validate_square_matrix(mat, name="thing")


def test_validate_square_matrix_rejects_non_square() -> None:
    """Tests that validate_square_matrix rejects non-square 2D arrays."""
    mat = np.zeros((2, 3), dtype=float)

    with pytest.raises(ValueError, match="must be square"):
        validate_square_matrix(mat, name="thing")


def test_resolve_covariance_input_fixed_cov_returns_none_callable() -> None:
    """Tests that resolve_covariance_input returns None callable for fixed covariance input."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    cov0 = np.eye(2, dtype=float)

    out_cov0, out_cov_fn = resolve_covariance_input(
        cov0, theta0=theta0, validate=validate_covariance_matrix_shape
    )

    assert out_cov_fn is None
    assert out_cov0.shape == (2, 2)


def test_resolve_covariance_input_callable_cov_evaluates_at_theta0() -> None:
    """Tests that resolve_covariance_input evaluates callable covariance at theta0."""
    theta0 = np.array([0.1, 0.2], dtype=float)

    def cov_fn(theta: np.ndarray) -> np.ndarray:
        return np.eye(2, dtype=float) * (1.0 + float(theta[0]) * 0.0)

    out_cov0, out_cov_fn = resolve_covariance_input(
        cov_fn, theta0=theta0, validate=validate_covariance_matrix_shape
    )

    assert callable(out_cov_fn)
    assert out_cov0.shape == (2, 2)


def test_resolve_covariance_input_tuple_requires_len_2() -> None:
    """Tests that resolve_covariance_input raises for tuple covariance input of wrong length."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    cov0 = np.eye(2, dtype=float)

    with pytest.raises(TypeError, match="tuple form must be \\(cov0, cov_fn\\)"):
        resolve_covariance_input((
            cov0,),
            theta0=theta0,
            validate=validate_covariance_matrix_shape)


def test_resolve_covariance_input_tuple_requires_callable() -> None:
    """Tests that resolve_covariance_input raises for tuple covariance input with non-callable second element."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    cov0 = np.eye(2, dtype=float)

    with pytest.raises(TypeError, match="cov_fn callable"):
        resolve_covariance_input(
            (cov0, 123),
            theta0=theta0,
            validate=validate_covariance_matrix_shape)


def test_flatten_matrix_c_order_returns_row_major_flattening() -> None:
    """Tests that flatten_matrix_c_order returns C-order flattened matrix."""
    theta = np.array([0.0, 0.0], dtype=float)
    out = flatten_matrix_c_order(cov_fn, theta, n_observables=2)

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=float))
    assert out.shape == (4,)


def test_flatten_matrix_c_order_rejects_wrong_shape() -> None:
    """Tests that flatten_matrix_c_order raises for covariance function returning wrong shape."""
    theta = np.array([0.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match="must return shape"):
        flatten_matrix_c_order(cov_fn_3by3, theta, n_observables=2)


def test_require_callable_returns_callable_when_present() -> None:
    """Tests that require_callable returns the callable when provided."""
    out = require_callable(lin_func, name="function", context="SomeContext")
    assert out is lin_func


def test_require_callable_raises_with_context_and_hint() -> None:
    """Tests that require_callable raises ValueError with context and hint when None is provided."""
    with pytest.raises(ValueError, match="SomeContext: function must be provided\\. Add one\\."):
        require_callable(None, name="function", context="SomeContext", hint="Add one.")
