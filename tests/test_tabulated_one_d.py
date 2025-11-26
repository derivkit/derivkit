"""Unit tests for derivkit.tabulated_model.one_d.Tabulated1DModel."""

import numpy as np
import pytest

from derivkit.tabulated_model.one_d import (
    Tabulated1DModel,
    parse_xy_table,
    tabulated1d_from_table,
)


def test_scalar_interpolation_basic():
    """Tests that basic scalar interpolation works as expected."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])

    model = Tabulated1DModel(x, y)

    x_new = np.array([0.5, 1.5])
    y_new = model(x_new)

    assert y_new.shape == x_new.shape
    np.testing.assert_allclose(y_new, [5.0, 15.0])


def test_vector_output_shape_and_values():
    """Tests that vector-valued outputs are handled correctly."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [4.0, 8.0],
            [9.0, 27.0],
        ]
    )

    model = Tabulated1DModel(x, y)

    x_new = np.array([0.5, 1.5, 2.5])
    y_new = model(x_new)

    expected = np.array(
        [
            [0.5, 0.5],
            [2.5, 4.5],
            [6.5, 17.5],
        ]
    )

    assert y_new.shape == (3, 2)
    np.testing.assert_allclose(y_new, expected)


def test_tensor_output_shape_preserved():
    """Tests that tensor-valued outputs are handled correctly."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.stack(
        [
            np.zeros((2, 3)),
            np.ones((2, 3)),
            2.0 * np.ones((2, 3)),
        ],
        axis=0,
    )

    model = Tabulated1DModel(x, y)

    x_new = np.array([0.5, 1.5])
    y_new = model(x_new)

    assert y_new.shape == (2, 2, 3)

    np.testing.assert_allclose(y_new[0], 0.5 * np.ones((2, 3)))
    np.testing.assert_allclose(y_new[1], 1.5 * np.ones((2, 3)))


def test_scalar_input_scalar_output():
    """Tests that scalar input yields scalar output."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])

    model = Tabulated1DModel(x, y)

    y_new = model(0.5)

    assert np.ndim(y_new) == 0
    assert float(y_new) == pytest.approx(5.0)


def test_scalar_input_tensor_output():
    """Tests that scalar input yields tensor output when y is tensor-valued."""
    x = np.array([0.0, 1.0])
    y = np.stack(
        [np.zeros((2, 2)), np.ones((2, 2))],
        axis=0,
    )

    model = Tabulated1DModel(x, y)

    out = model(0.5)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, 0.5 * np.ones((2, 2)))


def test_extrapolation_allowed_uses_numpy_interp():
    """Tests that extrapolation=True uses numpy.interp behaviour."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])

    model = Tabulated1DModel(x, y, extrapolate=True)

    x_new = np.array([-1.0, 3.0])
    y_new = model(x_new)

    expected = np.interp(x_new, x, y)
    np.testing.assert_allclose(y_new, expected)


def test_extrapolation_disabled_with_fill_value():
    """Tests that extrapolation=False with fill_value works as expected."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.column_stack([x, 2 * x])

    model = Tabulated1DModel(x, y, extrapolate=False, fill_value=-1.0)

    x_new = np.array([-0.5, 0.5, 1.5, 2.5])
    y_new = model(x_new)

    expected_inside = np.array(
        [
            [0.5, 1.0],
            [1.5, 3.0],
        ]
    )

    assert y_new.shape == (4, 2)
    np.testing.assert_allclose(y_new[1:3], expected_inside)
    np.testing.assert_allclose(y_new[0], [-1.0, -1.0])
    np.testing.assert_allclose(y_new[3], [-1.0, -1.0])


def test_extrapolation_disabled_raises_without_fill_value():
    """Tests that extrapolation=False without fill_value raises ValueError on out-of-range."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    model = Tabulated1DModel(x, y, extrapolate=False, fill_value=None)

    assert model(0.5) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        model(-0.1)


@pytest.mark.parametrize("dtype", [float, int])
def test_parse_xy_table_layout_n_2(dtype):
    """Tests that parse_xy_table works with (N, 2) layout for different dtypes."""
    x = np.array([0.0, 1.0, 2.0], dtype=dtype)
    y = np.array([10.0, 20.0, 30.0], dtype=dtype)
    table = np.column_stack([x, y])

    x_parsed, y_parsed = parse_xy_table(table)

    np.testing.assert_allclose(x_parsed, x.astype(float))
    np.testing.assert_allclose(y_parsed, y.astype(float))


def test_parse_xy_table_layout_n_m_plus_1():
    """Tests that parse_xy_table works with (N, M+1) layout."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    table = np.column_stack([x, y])

    x_parsed, y_parsed = parse_xy_table(table)

    assert x_parsed.shape == (3,)
    assert y_parsed.shape == (3, 2)
    np.testing.assert_allclose(x_parsed, x)
    np.testing.assert_allclose(y_parsed, y)


def test_parse_xy_table_layout_2_n():
    """Tests that parse_xy_table works with (2, N) layout."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 20.0, 30.0])
    table = np.vstack([x, y])

    x_parsed, y_parsed = parse_xy_table(table)

    np.testing.assert_allclose(x_parsed, x)
    np.testing.assert_allclose(y_parsed, y)


def test_parse_xy_table_invalid_shape_raises():
    """Tests that parse_xy_table raises ValueError on invalid shapes."""
    with pytest.raises(ValueError):
        parse_xy_table(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        parse_xy_table(np.zeros((3, 1)))


def test_tabulated1d_from_table_n_2_basic():
    """Tests that tabulated1d_from_table works with (N, 2) table."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])
    table = np.column_stack([x, y])

    model = tabulated1d_from_table(table)

    x_new = np.array([0.5, 1.5])
    y_new = model(x_new)

    np.testing.assert_allclose(y_new, [5.0, 15.0])


def test_tabulated1d_from_table_multi_component():
    """Tests that tabulated1d_from_table works with multi-component outputs."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 4.0],
            [2.0, 8.0],
        ]
    )
    table = np.column_stack([x, y])

    model = tabulated1d_from_table(table, extrapolate=False, fill_value=-1.0)

    x_new = np.array([-0.5, 0.5, 1.5, 2.5])
    y_new = model(x_new)

    assert y_new.shape == (4, 2)
    np.testing.assert_allclose(y_new[0], [-1.0, -1.0])
    np.testing.assert_allclose(y_new[3], [-1.0, -1.0])

    expected_inside = np.array(
        [
            [0.5, 2.0],
            [1.5, 6.0],
        ]
    )
    np.testing.assert_allclose(y_new[1:3], expected_inside)


def test_higher_dim_x_new_shape_preserved():
    """Tests that higher-dimensional x_new preserves shape in output."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])

    model = Tabulated1DModel(x, y)

    x_new = np.array([[0.0, 0.5], [1.0, 1.5]])
    y_new = model(x_new)

    assert y_new.shape == x_new.shape
    expected = np.array([[0.0, 5.0], [10.0, 15.0]])
    np.testing.assert_allclose(y_new, expected)


def test_non_monotonic_x_raises_value_error():
    """Tests that non-monotonic x raises ValueError."""
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 10.0, 5.0])

    with pytest.raises(ValueError):
        Tabulated1DModel(x, y)


def test_mismatched_x_y_length_raises_value_error():
    """Tests that mismatched lengths of x and y raise ValueError."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0])

    with pytest.raises(ValueError):
        Tabulated1DModel(x, y)
