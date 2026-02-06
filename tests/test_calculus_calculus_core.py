"""Unit tests for derivkit.calculus.calculus_core module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from derivkit.calculus.calculus_core import (
    cache_theta_function,
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

    with pytest.raises(ValueError, match=r"dispatch_tensor_output requires an array-valued model output"):
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


def test_cache_theta_function_reuses_results_for_identical_theta(monkeypatch):
    """Tests that cache_theta_function avoids repeated evaluations at identical theta."""
    calls = {"n": 0}

    def model(theta: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        # Return something that depends on theta to ensure correctness.
        t = np.asarray(theta, dtype=float)
        return np.array([t.sum(), t[0] * 2.0], dtype=float)

    cached = cache_theta_function(model)

    theta = np.array([1.0, 2.0, 3.0], dtype=float)

    y1 = cached(theta)
    y2 = cached(theta.copy())  # same values, different array object
    y3 = cached(theta.reshape(1, -1).ravel())  # different shape input, same content

    # Only the first call should hit the underlying model.
    assert calls["n"] == 1

    np.testing.assert_allclose(y1, np.array([6.0, 2.0]))
    np.testing.assert_allclose(y2, y1)
    np.testing.assert_allclose(y3, y1)


def test_cache_theta_function_returns_copies_not_views():
    """Tests that cache_theta_function returns copies so callers cannot mutate the cache."""
    def model(theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=float)
        return np.array([t[0], t[1]], dtype=float)

    cached = cache_theta_function(model)

    theta = np.array([5.0, 7.0], dtype=float)

    y1 = cached(theta)
    y1[0] = -999.0  # attempt to corrupt cached value

    y2 = cached(theta)

    # If we got a view into the cache, y2[0] would be -999.
    np.testing.assert_allclose(y2, np.array([5.0, 7.0]))


def test_cache_theta_function_distinguishes_different_theta_values():
    """Tests that cache_theta_function caches separately for different theta vectors."""
    calls = {"n": 0}

    def model(theta: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        t = np.asarray(theta, dtype=float)
        return np.array([t.sum()], dtype=float)  # shape (1,)

    cached = cache_theta_function(model)

    t1 = np.array([1.0, 2.0], dtype=float)
    t2 = np.array([1.0, 2.0 + 1e-3], dtype=float)

    y1 = cached(t1)
    y2 = cached(t2)
    y3 = cached(t1)

    # t1 and t2 differ in bytes, so should be two model calls.
    assert calls["n"] == 2

    assert float(np.asarray(y1).item()) == pytest.approx(3.0)
    assert float(np.asarray(y2).item()) == pytest.approx(3.001)
    assert float(np.asarray(y3).item()) == pytest.approx(3.0)


def test_dispatch_tensor_output_with_cached_function_reduces_model_calls():
    """Tests that using cache_theta_function upstream reduces repeated model calls in dispatcher."""
    calls = {"n": 0}

    def model(theta: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        # Any finite tensor output; content doesn't matter for this test.
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    cached = cache_theta_function(model)

    theta = np.array([0.1, 0.2, 0.3], dtype=float)

    _ = dispatch_tensor_output(
        function=cached,
        theta=theta,
        method="finite",
        outer_workers=1,
        inner_workers=1,
        dk_kwargs={},
        build_component=build_component_full_pp,
    )

    # dispatch_tensor_output calls function(theta) once to discover shape.
    # build_component_full_pp ignores the function, so no additional calls occur.
    assert calls["n"] == 1


def test_cache_theta_function_accepts_nonscalar_tensor_outputs():
    """Tests that cache_theta_function works for scalar and array outputs."""
    calls = {"n": 0}

    def scalarish(theta: np.ndarray) -> float:
        calls["n"] += 1
        t = np.asarray(theta, dtype=float)
        return float(t[0] + 10.0)

    cached = cache_theta_function(scalarish)

    theta = np.array([2.0], dtype=float)

    y1 = cached(theta)
    y2 = cached(theta.copy())

    assert calls["n"] == 1
    assert np.asarray(y1).shape == ()
    assert float(np.asarray(y2)) == pytest.approx(12.0)
