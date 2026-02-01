"""Unit tests for derivkit.utils.validate module."""

import numpy as np
import pytest

from derivkit.utils.validate import (
    check_scalar_valued,
    flatten_matrix_c_order,
    require_callable,
    resolve_covariance_input,
    resolve_dali_assembled_multiplet,
    resolve_dali_introduced_multiplet,
    validate_covariance_matrix_shape,
    validate_dali_shape,
    validate_fisher_shape,
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


def test_validate_fisher_shape_accepts_valid_shape() -> None:
    """Tests that validate_fisher_shape accepts theta0 (p,) and fisher (p,p)."""
    theta0 = np.array([0.1, 0.2, 0.3], dtype=float)
    fisher = np.eye(3, dtype=float)

    validate_fisher_shape(theta0, fisher)


def test_validate_fisher_shape_rejects_empty_theta0() -> None:
    """Tests that validate_fisher_shape rejects an empty theta0."""
    theta0 = np.array([], dtype=float)
    fisher = np.eye(1, dtype=float)

    with pytest.raises(ValueError, match="theta0 must be non-empty"):
        validate_fisher_shape(theta0, fisher)


def test_validate_fisher_shape_rejects_wrong_fisher_shape() -> None:
    """Tests that validate_fisher_shape rejects fisher matrices with wrong shape."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(3, dtype=float)

    with pytest.raises(ValueError, match="fisher must have shape"):
        validate_fisher_shape(theta0, fisher)


def test_validate_fisher_shape_rejects_nonfinite_when_check_finite() -> None:
    """Tests that validate_fisher_shape raises when check_finite=True and fisher is non-finite."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    fisher[0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="contains non-finite"):
        validate_fisher_shape(theta0, fisher, check_finite=True)


def test_validate_dali_shape_accepts_order1_multiplet() -> None:
    """Tests that validate_dali_shape accepts order-1 multiplets."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    validate_dali_shape(theta0, (fisher,))


def test_validate_dali_shape_accepts_order2_multiplet() -> None:
    """Tests that validate_dali_shape accepts order-2 multiplets."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)
    validate_dali_shape(theta0, (d21, d22))


def test_validate_dali_shape_accepts_order3_multiplet() -> None:
    """Tests that validate_dali_shape accepts order-3 multiplets."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    t31 = np.zeros((2, 2, 2, 2), dtype=float)
    t32 = np.zeros((2, 2, 2, 2, 2), dtype=float)
    t33 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)
    validate_dali_shape(theta0, (t31, t32, t33))



def test_validate_dali_shape_accepts_dict_of_multiplets() -> None:
    """Tests that validate_dali_shape accepts dict outputs with valid multiplets."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)
    t11 = np.zeros((2, 2, 2, 2), dtype=float)
    t12 = np.zeros((2, 2, 2, 2, 2), dtype=float)
    t13 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)

    dali = {
        1: (fisher,),
        2: (d21, d22),
        3: (t11, t12, t13),
    }

    validate_dali_shape(theta0, dali)


def test_validate_dali_shape_rejects_fisher_only_array_like() -> None:
    """Tests that validate_dali_shape rejects fisher-only array-like inputs."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    with pytest.raises(TypeError, match="Invalid DALI type"):
        validate_dali_shape(theta0, fisher)


def test_validate_dali_shape_rejects_dict_with_non_int_key() -> None:
    """Tests that validate_dali_shape rejects dict outputs with non-int keys."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    dali = {"2": (fisher,)}  # invalid key type

    with pytest.raises(TypeError, match="dict keys must be int"):
        validate_dali_shape(theta0, dali)


def test_validate_dali_shape_rejects_bad_multiplet_length() -> None:
    """Tests that validate_dali_shape rejects multiplets with invalid length."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    extra = np.zeros((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="Unrecognized DALI tuple form"):
        validate_dali_shape(theta0, (fisher, extra))


def test_validate_dali_shape_rejects_wrong_tensor_ndim_for_order2() -> None:
    """Tests that validate_dali_shape rejects order-2 tensors with wrong dimensionality."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21_wrong = np.zeros((2, 2), dtype=float)  # should be ndim=3
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match=r"Unrecognized DALI tuple form"):
        validate_dali_shape(theta0, (d21_wrong, d22))


def test_validate_dali_shape_rejects_wrong_tensor_shape_for_order3() -> None:
    """Tests that validate_dali_shape rejects order-3 tensors with wrong shape."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    t31 = np.zeros((2, 2, 2, 2), dtype=float)
    t32_wrong = np.zeros((2, 2, 2, 2), dtype=float)  # should be ndim=5
    t33 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match=r"must have ndim=5"):
        validate_dali_shape(theta0, (t31, t32_wrong, t33))

def test_validate_dali_shape_rejects_nonfinite_when_check_finite() -> None:
    """Tests that validate_dali_shape raises when check_finite=True and any tensor is non-finite."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)
    d22[0, 0, 0, 0] = np.inf

    with pytest.raises(FloatingPointError, match="contains non-finite values"):
        validate_dali_shape(theta0, (d21, d22), check_finite=True)


def test_resolve_dali_multiplet_from_tuple_infers_order2() -> None:
    """Tests that resolve_dali_multiplet infers the correct order from a tuple of tensors."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    order, multiplet = resolve_dali_introduced_multiplet(theta0, (d21, d22))

    assert order == 2
    assert isinstance(multiplet, tuple)
    assert len(multiplet) == 2
    assert multiplet[0].shape == (2, 2, 2)
    assert multiplet[1].shape == (2, 2, 2, 2)


def test_resolve_dali_multiplet_from_tuple_infers_order3() -> None:
    """Tests that resolve_dali_multiplet infers the correct order from a tuple of tensors."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    t31 = np.zeros((2, 2, 2, 2), dtype=float)
    t32 = np.zeros((2, 2, 2, 2, 2), dtype=float)
    t33 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)

    order, multiplet = resolve_dali_introduced_multiplet(theta0, (t31, t32, t33))

    assert order == 3
    assert len(multiplet) == 3


def test_resolve_dali_multiplet_rejects_fisher_only_array_like() -> None:
    """Tests that resolve_dali_multiplet rejects fisher-only array-like inputs."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    with pytest.raises(TypeError, match="Invalid DALI type"):
        resolve_dali_introduced_multiplet(theta0, fisher)


def test_resolve_dali_multiplet_from_dict_picks_highest_order_by_default() -> None:
    """Tests that resolve_dali_multiplet selects the highest key when forecast_order is None."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    dali = {
        1: (fisher,),
        2: (d21, d22),
    }

    order, multiplet = resolve_dali_introduced_multiplet(theta0, dali)

    assert order == 2
    assert len(multiplet) == 2


def test_resolve_dali_multiplet_from_dict_respects_forecast_order() -> None:
    """Tests that resolve_dali_multiplet selects the requested forecast_order key."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    dali = {
        1: (fisher,),
    }

    order, multiplet = resolve_dali_introduced_multiplet(theta0, dali, forecast_order=1)

    assert order == 1
    assert len(multiplet) == 1


def test_resolve_dali_multiplet_from_dict_rejects_missing_forecast_order() -> None:
    """Tests that resolve_dali_multiplet raises if forecast_order is not in dict keys."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    dali = {1: (fisher,)}

    with pytest.raises(ValueError, match="not in DALI dict keys"):
        resolve_dali_introduced_multiplet(theta0, dali, forecast_order=2)


def test_resolve_dali_multiplet_rejects_empty_dict() -> None:
    """Tests that resolve_dali_multiplet rejects an empty dict."""
    theta0 = np.array([0.1, 0.2], dtype=float)

    with pytest.raises(ValueError, match="DALI dict is empty"):
        resolve_dali_introduced_multiplet(theta0, {})


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


def test_flatten_matrix_c_order_returns_row_major_flattening() -> None:
    """Tests that flatten_matrix_c_order returns C-order flattened matrix."""
    theta = np.array([0.0, 0.0], dtype=float)
    out = flatten_matrix_c_order(cov_fn, theta, n_observables=2)

    np.testing.assert_allclose(out,
                               np.array([1.0, 2.0, 3.0, 4.0],
                                        dtype=float))
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
    with pytest.raises(ValueError, match="SomeContext:"
                                         " function must be provided. "
                                         "Add one."):
        require_callable(None,
                         name="function",
                         context="SomeContext",
                         hint="Add one.")


def test_resolve_covariance_input_rejects_tuple_input() -> None:
    """Tests that resolve_covariance_input raises Exception for tuple covariance input."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    cov0 = np.eye(2, dtype=float)

    with pytest.raises(Exception):
        resolve_covariance_input(
            (cov0, cov_fn),
            theta0=theta0,
            validate=validate_covariance_matrix_shape,
        )


def test_resolve_dali_assembled_multiplet_from_dict_order1_returns_fisher_only() -> None:
    """Tests that resolve_dali_assembled_multiplet returns (F,) for order=1 dict input."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    dali = {1: (fisher,)}

    order, multiplet = resolve_dali_assembled_multiplet(theta0, dali, forecast_order=1)

    assert order == 1
    assert isinstance(multiplet, tuple)
    assert len(multiplet) == 1
    assert multiplet[0].shape == (2, 2)
    np.testing.assert_allclose(multiplet[0], fisher)


def test_resolve_dali_assembled_multiplet_from_dict_order2_assembles_fisher_and_doublet() -> None:
    """Tests that resolve_dali_assembled_multiplet assembles (F, D1, D2) for order=2 dict input."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    dali = {
        1: (fisher,),
        2: (d21, d22),
    }

    order, multiplet = resolve_dali_assembled_multiplet(theta0, dali, forecast_order=2)

    assert order == 2
    assert isinstance(multiplet, tuple)
    assert len(multiplet) == 3
    assert multiplet[0].shape == (2, 2)
    assert multiplet[1].shape == (2, 2, 2)
    assert multiplet[2].shape == (2, 2, 2, 2)
    np.testing.assert_allclose(multiplet[0], fisher)


def test_resolve_dali_assembled_multiplet_from_dict_order3_assembles_fisher_doublet_and_triplet() -> None:
    """Tests that resolve_dali_assembled_multiplet assembles (F, D1, D2, T1, T2, T3) for order=3 dict input."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)
    t31 = np.zeros((2, 2, 2, 2), dtype=float)
    t32 = np.zeros((2, 2, 2, 2, 2), dtype=float)
    t33 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)

    dali = {
        1: (fisher,),
        2: (d21, d22),
        3: (t31, t32, t33),
    }

    order, multiplet = resolve_dali_assembled_multiplet(theta0, dali, forecast_order=3)

    assert order == 3
    assert isinstance(multiplet, tuple)
    assert len(multiplet) == 6
    assert multiplet[0].shape == (2, 2)
    assert multiplet[1].shape == (2, 2, 2)
    assert multiplet[2].shape == (2, 2, 2, 2)
    assert multiplet[3].shape == (2, 2, 2, 2)
    assert multiplet[4].shape == (2, 2, 2, 2, 2)
    assert multiplet[5].shape == (2, 2, 2, 2, 2, 2)
    np.testing.assert_allclose(multiplet[0], fisher)


def test_resolve_dali_assembled_multiplet_from_dict_picks_highest_order_by_default() -> None:
    """Tests that resolve_dali_assembled_multiplet selects and assembles the highest available order when forecast_order is None."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    dali = {
        1: (fisher,),
        2: (d21, d22),
    }

    order, multiplet = resolve_dali_assembled_multiplet(theta0, dali)

    assert order == 2
    assert len(multiplet) == 3
    assert multiplet[0].shape == (2, 2)


def test_resolve_dali_assembled_multiplet_from_dict_rejects_missing_fisher_for_order2() -> None:
    """Tests that resolve_dali_assembled_multiplet rejects dict inputs missing order=1 when order>1 is requested."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    dali = {2: (d21, d22)}

    with pytest.raises(ValueError, match="must start at key=1"):
        resolve_dali_assembled_multiplet(theta0, dali, forecast_order=2)


def test_resolve_dali_assembled_multiplet_rejects_tuple_input_for_order2() -> None:
    """Tests that resolve_dali_assembled_multiplet rejects tuple inputs for order>1 because Fisher is unavailable."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    d21 = np.zeros((2, 2, 2), dtype=float)
    d22 = np.zeros((2, 2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="Order>1 evaluation requires the dict form"):
        resolve_dali_assembled_multiplet(theta0, (d21, d22), forecast_order=2)


def test_resolve_dali_assembled_multiplet_from_tuple_accepts_fisher_only() -> None:
    """Tests that resolve_dali_assembled_multiplet accepts Fisher-only tuple inputs and returns (F,) for order=1."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    order, multiplet = resolve_dali_assembled_multiplet(theta0, (fisher,))

    assert order == 1
    assert isinstance(multiplet, tuple)
    assert len(multiplet) == 1
    assert multiplet[0].shape == (2, 2)


def test_resolve_dali_assembled_multiplet_from_tuple_rejects_forecast_order_mismatch() -> None:
    """Tests that resolve_dali_assembled_multiplet rejects forecast_order>1 for Fisher-only tuple inputs."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="forecast_order>1 requires the dict form"):
        resolve_dali_assembled_multiplet(theta0, (fisher,), forecast_order=2)


def test_resolve_dali_assembled_multiplet_from_dict_rejects_forecast_order_not_in_keys() -> None:
    """Tests that resolve_dali_assembled_multiplet rejects forecast_order not present in dict keys."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    fisher = np.eye(2, dtype=float)

    dali = {1: (fisher,)}

    with pytest.raises(ValueError, match=r"forecast_order=4 not in DALI dict keys"):
        resolve_dali_assembled_multiplet(theta0, dali, forecast_order=4)
