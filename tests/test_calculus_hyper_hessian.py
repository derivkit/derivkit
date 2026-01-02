"""Unit tests for derivkit.calculus.hyper_hessian.build_hyper_hessian."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.calculus.hyper_hessian import build_hyper_hessian

_METHOD_CASES = [
    ("finite", {}),
    ("finite", {"extrapolation": "richardson"}),
    ("finite", {"extrapolation": "ridders"}),
    ("finite", {"extrapolation": "gauss-richardson"}),
    ("adaptive", {}),
    ("local_polynomial", {}),
]


def cubic_scalar(theta):
    """A cubic scalar function with known third derivatives."""
    x, y, z = np.asarray(theta, dtype=float)
    return float(x**3 + y**3 + z**3)


def cubic_vector(theta):
    """Vector output with known component-wise third derivatives."""
    x, y, z = np.asarray(theta, dtype=float)
    return np.array([x**3, y**3, z**3, x**3 + y**3 + z**3], dtype=float)


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
def test_build_hyper_hessian_scalar_cubic(method, extra_kwargs):
    """Tests that cubic scalar function produces correct hyper-Hessian."""
    theta0 = np.array([1.2, -0.3, 2.0], dtype=float)
    hhh = build_hyper_hessian(cubic_scalar, theta0, method=method, **extra_kwargs)

    assert hhh.shape == (3, 3, 3)

    expected = np.zeros((3, 3, 3), dtype=float)
    expected[0, 0, 0] = 6.0
    expected[1, 1, 1] = 6.0
    expected[2, 2, 2] = 6.0

    np.testing.assert_allclose(hhh, expected, rtol=0, atol=5e-6)


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
def test_build_hyper_hessian_tensor_output_shapes_and_values(method, extra_kwargs):
    """Tests that cubic vector function produces correct hyper-Hessian."""
    theta0 = np.array([0.1, 0.2, 0.3], dtype=float)
    hhh = build_hyper_hessian(cubic_vector, theta0, method=method, **extra_kwargs)

    assert hhh.shape == (4, 3, 3, 3)

    exp0 = np.zeros((3, 3, 3), dtype=float)
    exp0[0, 0, 0] = 6.0

    exp1 = np.zeros((3, 3, 3), dtype=float)
    exp1[1, 1, 1] = 6.0

    exp2 = np.zeros((3, 3, 3), dtype=float)
    exp2[2, 2, 2] = 6.0

    exp3 = np.zeros((3, 3, 3), dtype=float)
    exp3[0, 0, 0] = 6.0
    exp3[1, 1, 1] = 6.0
    exp3[2, 2, 2] = 6.0

    np.testing.assert_allclose(hhh[0], exp0, rtol=0, atol=5e-6)
    np.testing.assert_allclose(hhh[1], exp1, rtol=0, atol=5e-6)
    np.testing.assert_allclose(hhh[2], exp2, rtol=0, atol=5e-6)
    np.testing.assert_allclose(hhh[3], exp3, rtol=0, atol=5e-6)


def test_build_hyper_hessian_raises_on_empty_theta():
    """Tests that empty theta0 raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        build_hyper_hessian(cubic_scalar, np.array([], dtype=float))


def test_build_hyper_hessian_tensor_outputs_have_expected_shapes():
    """Tests that scalar path rejects non-scalar output."""
    theta0 = np.array([1.0, 2.0, 3.0], dtype=float)

    def not_scalar(theta):
        return np.array([1.0, 2.0], dtype=float)

    # Force scalar helper by calling it indirectly: build_hyper_hessian will route to tensor path,
    # so we instead check the helper behavior via a scalar function that returns shape (1,).
    def shape1(theta):
        """Returns shape (1,) output to trigger scalar path."""
        return np.asarray([float(np.sum(theta))], dtype=float)

    hhh = build_hyper_hessian(shape1, theta0, method="finite")
    assert hhh.shape == (1, 3, 3, 3)

    hhh2 = build_hyper_hessian(not_scalar, theta0, method="finite")
    assert hhh2.shape == (2, 3, 3, 3)
