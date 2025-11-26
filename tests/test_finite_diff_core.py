"""Unit tests for single_finite_step in derivkit.finite.single_step."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import derivkit.finite.core as ss
from derivkit.finite.core import single_finite_step

OFFSETS: dict[int, np.ndarray] = {}
COEFFS_TABLE: dict[tuple[int, int], np.ndarray] = {}

LAST_H: float | None = None
LAST_STENCIL: np.ndarray | None = None
EVAL_VALUES: np.ndarray | None = None


def fake_get_finite_difference_tables(h: float):
    """Stub for get_finite_difference_tables that returns global tables."""
    global LAST_H
    LAST_H = float(h)
    return OFFSETS, COEFFS_TABLE


def fake_eval_points(function: Any, stencil: np.ndarray, n_workers: int):
    """Stub for eval_points that returns pre-defined values."""
    # Record stencil for inspection
    global LAST_STENCIL
    LAST_STENCIL = np.asarray(stencil, dtype=float)

    if EVAL_VALUES is None:
        raise RuntimeError("EVAL_VALUES not set in test.")
    return np.asarray(EVAL_VALUES, dtype=float)


def dummy_function(x: float) -> float:
    """Dummy function, not actually used by stubs (eval_points ignores it)."""
    return x  # placeholder; fake_eval_points overrides evaluation


def test_single_finite_step_scalar_uses_offsets_and_coeffs(monkeypatch):
    """Tests that single_finite_step should build the stencil and apply coeffs correctly."""
    global OFFSETS, COEFFS_TABLE, EVAL_VALUES, LAST_H, LAST_STENCIL

    # Setup fake tables: 3-point "stencil" with indices [0, 1]
    OFFSETS = {3: np.array([0.0, 1.0], dtype=float)}
    COEFFS_TABLE = {(3, 1): np.array([0.0, 1.0], dtype=float)}

    # Values at these two stencil points
    # derivative = values^T @ coeffs = 0 * 2.0 + 1 * 5.0 = 5.0
    EVAL_VALUES = np.array([2.0, 5.0], dtype=float)

    LAST_H = None
    LAST_STENCIL = None

    monkeypatch.setattr(ss, "get_finite_difference_tables", fake_get_finite_difference_tables)
    monkeypatch.setattr(ss, "eval_points", fake_eval_points)

    x0 = 1.5
    h = 0.1

    result = single_finite_step(
        dummy_function,
        x0=x0,
        order=1,
        stepsize=h,
        num_points=3,
        n_workers=1,
    )

    # Check scalar result
    assert isinstance(result, float)
    assert np.isclose(result, 5.0)

    # Check that get_finite_difference_tables saw the correct stepsize
    assert LAST_H == pytest.approx(h)

    # Check that stencil was built as x0 + offsets * h
    expected_stencil = np.array([x0 + 0.0 * h, x0 + 1.0 * h], dtype=float)
    assert LAST_STENCIL is not None
    assert np.allclose(LAST_STENCIL, expected_stencil)


def test_single_finite_step_vector_output_flattens_correctly(monkeypatch):
    """Tests that vector-valued outputs should produce a 1D array derivative."""
    global OFFSETS, COEFFS_TABLE, EVAL_VALUES, LAST_STENCIL

    OFFSETS = {3: np.array([0.0, 1.0], dtype=float)}
    # Two stencil points, weights [1, 1] => derivative = row-wise sums
    COEFFS_TABLE = {(3, 1): np.array([1.0, 1.0], dtype=float)}

    # Shape (n_points, n_comp) = (2, 3)
    # derivative[k] = sum over points of values[i, k]
    EVAL_VALUES = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
        ],
        dtype=float,
    )

    LAST_STENCIL = None

    monkeypatch.setattr(ss, "get_finite_difference_tables", fake_get_finite_difference_tables)
    monkeypatch.setattr(ss, "eval_points", fake_eval_points)

    x0 = 0.0
    h = 0.5

    result = single_finite_step(
        dummy_function,
        x0=x0,
        order=1,
        stepsize=h,
        num_points=3,
        n_workers=2,
    )

    # Should be a 1D array with componentwise sums
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.allclose(result, np.array([3.0, 30.0, 300.0], dtype=float))

    # Ensure stencil matched our offsets
    expected_stencil = np.array([x0 + 0.0 * h, x0 + 1.0 * h], dtype=float)
    assert LAST_STENCIL is not None
    assert np.allclose(LAST_STENCIL, expected_stencil)


def test_single_finite_step_raises_on_missing_coefficients(monkeypatch):
    """Tests that if (num_points, order) is missing in the coefficient table, raise ValueError."""
    global OFFSETS, COEFFS_TABLE, EVAL_VALUES

    OFFSETS = {3: np.array([0.0, 1.0], dtype=float)}
    COEFFS_TABLE = {}  # no entries, so (3, 2) is missing
    EVAL_VALUES = np.array([0.0, 0.0], dtype=float)

    monkeypatch.setattr(ss, "get_finite_difference_tables", fake_get_finite_difference_tables)
    monkeypatch.setattr(ss, "eval_points", fake_eval_points)

    with pytest.raises(ValueError) as excinfo:
        single_finite_step(
            dummy_function,
            x0=0.0,
            order=2,
            stepsize=0.1,
            num_points=3,
            n_workers=1,
        )

    msg = str(excinfo.value)
    assert "stencil=3, order=2" in msg


def test_single_finite_step_tensor_layout():
    """Tests that tensor-valued outputs are flattened correctly in C order."""

    def tensor_func(x: float) -> np.ndarray:
        return np.array([[x, 2.0 * x], [3.0 * x, 4.0 * x]])

    x0 = 0.3
    order = 1
    stepsize = 1e-2
    num_points = 5
    n_workers = 1

    deriv = single_finite_step(
        tensor_func,
        x0=x0,
        order=order,
        stepsize=stepsize,
        num_points=num_points,
        n_workers=n_workers,
    )

    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_flat = expected.ravel(order="C")

    assert isinstance(deriv, np.ndarray)
    assert deriv.shape == expected_flat.shape
    np.testing.assert_allclose(deriv, expected_flat, rtol=1e-6, atol=1e-8)
