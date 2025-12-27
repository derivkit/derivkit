"""Unit tests for derivkit.calculus.calculus_core module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from derivkit.calculus.calculus_core import (
    component_scalar_eval,
    dispatch_tensor_output,
)


def scalar_model(_theta: np.ndarray) -> float:
    """Scalar model for testing."""
    return 3.0


def vector_model(theta: np.ndarray) -> np.ndarray:
    """Vector model for testing."""
    return np.array([theta[0] + 1.0, theta[1] + 2.0], dtype=float)


def length2_model(_theta: np.ndarray) -> np.ndarray:
    """Length-2 vector model for testing."""
    return np.array([1.0, 2.0], dtype=float)


def matrix2x2_model(_theta: np.ndarray) -> np.ndarray:
    """2x2 matrix model for testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)


def scalar_one_model(_theta: np.ndarray) -> float:
    """Scalar model returning 1.0 for testing."""
    return 1.0


def nan_model(_theta: np.ndarray) -> np.ndarray:
    """Model returning NaN for testing."""
    return np.array([np.nan, 1.0], dtype=float)


def finite_vec_model(_theta: np.ndarray) -> np.ndarray:
    """Model returning finite values for testing."""
    return np.array([0.0, 1.0], dtype=float)


def build_component_full_pp(
        idx: int,
        theta: np.ndarray,
        method: str | None,
        inner_workers: int | None,
        dk_kwargs: dict,
        function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Builds a full (p, p) component for testing."""
    _ = (theta, method, inner_workers, dk_kwargs, function)
    p = int(theta.size)
    return np.full((p, p), float(idx))


def build_component_scalar0(
        idx: int,
        theta: np.ndarray,
        method: str | None,
        inner_workers: int | None,
        dk_kwargs: dict,
        function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Builds a scalar component returning 0.0 for testing."""
    _ = (idx, theta, method, inner_workers, dk_kwargs, function)
    return np.array([0.0], dtype=float)


def build_component_inf_for_idx1(
        idx: int,
        theta: np.ndarray,
        method: str | None,
        inner_workers: int | None,
        dk_kwargs: dict,
        function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Builds a scalar component returning inf for idx=1 for testing."""
    _ = (theta, method, inner_workers, dk_kwargs, function)
    if int(idx) == 1:
        return np.array([np.inf], dtype=float)
    return np.array([0.0], dtype=float)


def test_component_scalar_eval_scalar_output_idx0():
    """Tests that scalar outputs return the value for idx=0."""
    theta = np.array([1.0, 2.0], dtype=float)
    assert component_scalar_eval(theta, function=scalar_model, idx=0) == 3.0


def test_component_scalar_eval_vector_output_picks_component():
    """Tests that vector outputs return the correct component."""
    theta = np.array([10.0, 20.0], dtype=float)
    assert component_scalar_eval(theta, function=vector_model, idx=0) == 11.0
    assert component_scalar_eval(theta, function=vector_model, idx=1) == 22.0


def test_component_scalar_eval_out_of_range_raises():
    """Tests that out-of-range indices raise IndexError."""
    theta = np.array([0.0], dtype=float)
    with pytest.raises(IndexError):
        component_scalar_eval(theta, function=length2_model, idx=5)


def test_dispatch_tensor_output_shapes_and_values_serial():
    """Tests that tensor output dispatch works in serial."""
    theta = np.array([0.1, 0.2, 0.3], dtype=float)
    p = theta.size

    out = dispatch_tensor_output(
        function=matrix2x2_model,
        theta=theta,
        method="finite",
        outer_workers=1,
        inner_workers=1,
        dk_kwargs={},
        build_component=build_component_full_pp,
    )

    assert out.shape == (2, 2, p, p)
    assert np.all(out[0, 0] == 0.0)
    assert np.all(out[0, 1] == 1.0)
    assert np.all(out[1, 0] == 2.0)
    assert np.all(out[1, 1] == 3.0)


def test_dispatch_tensor_output_raises_on_scalar_output():
    """Tests that scalar outputs raise ValueError."""
    theta = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="only for tensor outputs"):
        dispatch_tensor_output(
            function=scalar_one_model,
            theta=theta,
            method=None,
            outer_workers=1,
            inner_workers=1,
            dk_kwargs={},
            build_component=build_component_scalar0,
        )


def test_dispatch_tensor_output_raises_on_nonfinite_model_output():
    """Tests that non-finite model outputs raise FloatingPointError."""
    theta = np.array([1.0], dtype=float)

    with pytest.raises(FloatingPointError, match="Non-finite values in model output"):
        dispatch_tensor_output(
            function=nan_model,
            theta=theta,
            method=None,
            outer_workers=1,
            inner_workers=1,
            dk_kwargs={},
            build_component=build_component_scalar0,
        )


def test_dispatch_tensor_output_raises_on_nonfinite_component_result():
    """Tests that non-finite component results raise FloatingPointError."""
    theta = np.array([1.0], dtype=float)

    with pytest.raises(
        FloatingPointError,
        match="Non-finite values encountered in tensor-output derivative object",
    ):
        dispatch_tensor_output(
            function=finite_vec_model,
            theta=theta,
            method=None,
            outer_workers=1,
            inner_workers=1,
            dk_kwargs={},
            build_component=build_component_inf_for_idx1,
        )
