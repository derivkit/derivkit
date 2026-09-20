"""Unit tests for ``derivkit.calculus.hyper_hessian.build_hyper_hessian``."""

from __future__ import annotations

from itertools import permutations

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

_SMOOTH_METHOD_CASES = [
    ("finite", {}),
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


def quartic_scalar(theta):
    """A quartic scalar function with known derivatives through fourth order."""
    x, y = np.asarray(theta, dtype=float)
    return float(x**4 + x**3 * y + x**2 * y**2 + x * y**3 + y**4)


def smooth_vector(theta):
    """An infinitely differentiable vector-valued function."""
    x, y = np.asarray(theta, dtype=float)
    return np.array([x * np.cos(y), np.exp(2*x) + x*y*y])


@pytest.mark.parametrize("method, extra_kwargs", _SMOOTH_METHOD_CASES)
def test_build_hyper_hessian_smooth_function_quartic_derivative(
    method,
    extra_kwargs,
):
    """Tests fourth-order partials for smooth nonpolynomial functions."""
    theta0 = np.array([3.27, -1.4], dtype=float)

    calculated = build_hyper_hessian(
        smooth_vector,
        theta0,
        order=4,
        method=method,
        **extra_kwargs,
    )

    assert calculated.shape == (2, 2, 2, 2, 2)

    expected = np.zeros((2, 2, 2, 2, 2), dtype=float)
    expected[0, 0, 1, 1, 1] = np.sin(theta0[1])
    expected[0, 1, 1, 1, 1] = theta0[0] * np.cos(theta0[1])
    expected[1, 0, 0, 0, 0] = 16 * np.exp(2 * theta0[0])

    atol = 5e-3
    rtol = 2e-3

    def assert_values(variable, shape):
        """Checks that the equality of mixed partials holds."""
        for indices in set(permutations(shape)):
            argument = (variable,) + indices
            sorted_argument = (variable,) + tuple(sorted(indices))
            assert np.isclose(
                calculated[argument],
                expected[sorted_argument],
                atol=atol,
                rtol=rtol,
            )

    for i in (0, 1):
        np.testing.assert_allclose(
            calculated[i, 0, 0, 0, 0],
            expected[i, 0, 0, 0, 0],
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            calculated[i, 1, 1, 1, 1],
            expected[i, 1, 1, 1, 1],
            atol=atol,
            rtol=rtol,
        )
        assert_values(i, (0, 0, 0, 1))
        assert_values(i, (0, 0, 1, 1))
        assert_values(i, (0, 1, 1, 1))


def f_nonfinite(theta):
    """Produces a non-finite output."""
    x = np.asarray(theta, float)
    return np.nan + x.sum()  # force NaN


def f_nonfinite_hyper_hessian(theta: np.ndarray) -> np.ndarray:
    """Model returning nonfinite hyper-Hessian."""
    x = np.asarray(theta, float)
    return np.power(x,4/3)


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

    def not_scalar(_theta):
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


def test_hyper_hessian_raises_on_nonfinite_model_output():
    """Tests that non-finite model outputs raise FloatingPointError."""
    theta = np.array([1.0], dtype=float)

    with pytest.raises(FloatingPointError, match="Non-finite values in model output"):
        build_hyper_hessian(
            function=f_nonfinite,
            theta0=theta,
            method=None,
            n_workers=1,
        )


@pytest.mark.filterwarnings("ignore:invalid value encountered in power:RuntimeWarning")
def test_hyper_hessian_raises_on_nonfinite_component_result():
    """Tests that non-finite component results raise FloatingPointError."""
    theta = np.array([0.0], dtype=float)

    with pytest.raises(
        FloatingPointError,
        match="Non-finite values encountered in hyper-Hessian",
    ):
        build_hyper_hessian(
            function=f_nonfinite_hyper_hessian,
            theta0=theta,
            method=None,
            n_workers=1,
        )


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
@pytest.mark.parametrize("order", [1, 2])
def test_build_hyper_hessian_lower_orders(method, extra_kwargs, order):
    """Tests lower derivative orders across differentiation methods."""
    theta0 = np.array([1.0, 2.0], dtype=float)

    derivative = build_hyper_hessian(
        quartic_scalar,
        theta0,
        order=order,
        method=method,
        **extra_kwargs,
    )

    if order == 1:
        expected = np.array([26.0, 49.0])
    else:
        expected = np.array([
            [32.0, 23.0],
            [23.0, 62.0],
        ])

    np.testing.assert_allclose(
        derivative,
        expected,
        rtol=0,
        atol=5e-5,
    )


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
def test_build_hyper_hessian_scalar_quartic_order_four(method, extra_kwargs):
    """Tests that a quartic scalar function has the correct fourth derivative."""
    theta0 = np.array([0.7, -0.4], dtype=float)

    hhhh = build_hyper_hessian(
        quartic_scalar,
        theta0,
        order=4,
        method=method,
        **extra_kwargs,
    )

    assert hhhh.shape == (2, 2, 2, 2)

    expected = np.zeros((2, 2, 2, 2), dtype=float)

    expected[0, 0, 0, 0] = 24.0
    expected[1, 1, 1, 1] = 24.0

    for indices in set(permutations((0, 0, 0, 1))):
        expected[indices] = 6.0

    for indices in set(permutations((0, 0, 1, 1))):
        expected[indices] = 4.0

    for indices in set(permutations((0, 1, 1, 1))):
        expected[indices] = 6.0

    np.testing.assert_allclose(
        hhhh,
        expected,
        rtol=0,
        atol=5e-5,
    )


def test_build_hyper_hessian_default_order_is_three():
    """Tests that the default derivative order remains third order."""
    theta0 = np.array([1.2, -0.3, 2.0], dtype=float)

    default = build_hyper_hessian(
        cubic_scalar,
        theta0,
        method="finite",
    )
    explicit = build_hyper_hessian(
        cubic_scalar,
        theta0,
        order=3,
        method="finite",
    )

    np.testing.assert_allclose(default, explicit, rtol=0, atol=0)


def test_build_hyper_hessian_raises_on_negative_order():
    """Tests that negative derivative orders raise ValueError."""
    theta0 = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        build_hyper_hessian(
            quartic_scalar,
            theta0,
            order=-1,
        )

@pytest.mark.parametrize(
    "function",
    [
        cubic_scalar,
        cubic_vector,
    ],
)
def test_build_hyper_hessian_order_zero(function):
    """Tests that zeroth order returns the original function value."""
    theta0 = np.array([1.0, 2.0, 3.0], dtype=float)

    result = build_hyper_hessian(
        function,
        theta0,
        order=0,
    )

    expected = np.asarray(function(theta0), dtype=float)

    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    assert result.shape == expected.shape
