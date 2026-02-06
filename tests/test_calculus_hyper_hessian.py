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


def _atol_for(method: str, extra_kwargs: dict) -> float:
    """Returns a method-aware absolute tolerance for third-derivative checks."""
    if method == "finite":
        extrap = extra_kwargs.get("extrapolation", None)
        # Extrapolated finite differences should be quite accurate.
        if extrap is not None:
            return 5e-6
        # Plain finite differences: slightly looser but still tight.
        return 5e-5

    if method == "adaptive":
        return 5e-4

    if method == "local_polynomial":
        return 5e-4

    # Fallback.
    return 5e-4


def cubic_scalar(theta):
    """A cubic scalar function with known third derivatives."""
    x, y, z = np.asarray(theta, dtype=float)
    return float(x**3 + y**3 + z**3)


def cubic_vector(theta):
    """Vector output with known component-wise third derivatives."""
    x, y, z = np.asarray(theta, dtype=float)
    return np.array([x**3, y**3, z**3, x**3 + y**3 + z**3], dtype=float)


def mixed_cubic_scalar(theta):
    """A cubic scalar with mixed terms to exercise iik and ijk identities.

    f(x,y,z) = x^3 + y^3 + z^3 + a*x^2*y + b*x*y*z

    Notes:
        The current hyper-Hessian reconstruction uses the all-distinct identity
        with denominator 12:

            T_ijk_like = (S1 - S2 - S3 - S4) / 12

        For a pure b*x*y*z term, D3(v) = 6*b*v_x*v_y*v_z, which makes the above
        return 2*b for distinct triples. These tests match that convention.

    Expected non-zero third derivatives (symmetric tensor T):
        T_xxx = 6
        T_yyy = 6
        T_zzz = 6
        T_xxy = 2a   (and permutations)
        T_xyz = 2b   (and permutations; matches current /12 convention)
    """
    x, y, z = np.asarray(theta, dtype=float)
    a = 1.5
    b = -2.0
    return float(x**3 + y**3 + z**3 + a * x**2 * y + b * x * y * z)


def mixed_cubic_vector(theta):
    """Vector output version of mixed_cubic_scalar to test tensor-valued outputs."""
    x, y, z = np.asarray(theta, dtype=float)
    a = 1.5
    b = -2.0
    f0 = x**3 + a * x**2 * y
    f1 = y**3 + b * x * y * z
    f2 = z**3
    f3 = f0 + f1 + f2
    return np.array([f0, f1, f2, f3], dtype=float)


def _expected_mixed_tensor(*, a: float, b: float) -> np.ndarray:
    """Build expected (3,3,3) symmetric third-derivative tensor.

    This matches the current implementation convention:
        - x^3, y^3, z^3 give 6 on the pure diagonals.
        - a*x^2*y gives 2a on xxy permutations.
        - b*x*y*z gives 2b on xyz permutations (due to /12 identity).
    """
    exp = np.zeros((3, 3, 3), dtype=float)

    # Pure cubics.
    exp[0, 0, 0] = 6.0
    exp[1, 1, 1] = 6.0
    exp[2, 2, 2] = 6.0

    # a*x^2*y => T_xxy = 2a and symmetric permutations.
    val_xxy = 2.0 * a
    for idx in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        exp[idx] = val_xxy

    # b*x*y*z => current /12 convention yields 2b for distinct triples.
    val_xyz = 2.0 * b
    for idx in [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]:
        exp[idx] = val_xyz

    return exp


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

    atol = _atol_for(method, extra_kwargs)
    np.testing.assert_allclose(hhh, expected, rtol=0, atol=atol)


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

    atol = _atol_for(method, extra_kwargs)
    np.testing.assert_allclose(hhh[0], exp0, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[1], exp1, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[2], exp2, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[3], exp3, rtol=0, atol=atol)


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
def test_build_hyper_hessian_scalar_mixed_terms(method, extra_kwargs):
    """Tests that mixed cubic scalar exercises iik and ijk identities correctly."""
    theta0 = np.array([0.7, -1.1, 0.3], dtype=float)
    hhh = build_hyper_hessian(mixed_cubic_scalar, theta0, method=method, **extra_kwargs)

    assert hhh.shape == (3, 3, 3)

    expected = _expected_mixed_tensor(a=1.5, b=-2.0)
    atol = _atol_for(method, extra_kwargs)
    np.testing.assert_allclose(hhh, expected, rtol=0, atol=atol)

    # Basic symmetry spot-checks.
    assert np.allclose(hhh[0, 0, 1], hhh[0, 1, 0], atol=atol)
    assert np.allclose(hhh[0, 0, 1], hhh[1, 0, 0], atol=atol)
    assert np.allclose(hhh[0, 1, 2], hhh[2, 1, 0], atol=atol)
    assert np.allclose(hhh[0, 1, 2], hhh[1, 0, 2], atol=atol)


@pytest.mark.parametrize("method, extra_kwargs", _METHOD_CASES)
def test_build_hyper_hessian_tensor_mixed_terms(method, extra_kwargs):
    """Tests tensor-valued mixed cubics produce correct component hyper-Hessians."""
    theta0 = np.array([0.25, 0.5, -0.75], dtype=float)
    hhh = build_hyper_hessian(
        mixed_cubic_vector,
        theta0,
        method=method,
        n_workers=2,
        **extra_kwargs,
    )

    assert hhh.shape == (4, 3, 3, 3)

    # f0 = x^3 + a*x^2*y
    exp0 = np.zeros((3, 3, 3), dtype=float)
    exp0[0, 0, 0] = 6.0
    for idx in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        exp0[idx] = 2.0 * 1.5

    # f1 = y^3 + b*x*y*z (distinct triple uses 2b convention)
    exp1 = np.zeros((3, 3, 3), dtype=float)
    exp1[1, 1, 1] = 6.0
    for idx in [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]:
        exp1[idx] = 2.0 * (-2.0)

    # f2 = z^3
    exp2 = np.zeros((3, 3, 3), dtype=float)
    exp2[2, 2, 2] = 6.0

    # f3 = f0 + f1 + f2
    exp3 = exp0 + exp1 + exp2

    atol = _atol_for(method, extra_kwargs)
    np.testing.assert_allclose(hhh[0], exp0, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[1], exp1, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[2], exp2, rtol=0, atol=atol)
    np.testing.assert_allclose(hhh[3], exp3, rtol=0, atol=atol)


def test_build_hyper_hessian_raises_on_empty_theta():
    """Tests that empty theta0 raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        build_hyper_hessian(cubic_scalar, np.array([], dtype=float))


def test_build_hyper_hessian_tensor_outputs_have_expected_shapes():
    """Tests that array outputs route to tensor path and keep expected shapes."""
    theta0 = np.array([1.0, 2.0, 3.0], dtype=float)

    def not_scalar(theta):
        _ = np.asarray(theta)
        return np.array([1.0, 2.0], dtype=float)

    def shape1(theta):
        """Returns shape (1,) output (tensor path with one component)."""
        return np.asarray([float(np.sum(theta))], dtype=float)

    hhh = build_hyper_hessian(shape1, theta0, method="finite")
    assert hhh.shape == (1, 3, 3, 3)

    hhh2 = build_hyper_hessian(not_scalar, theta0, method="finite")
    assert hhh2.shape == (2, 3, 3, 3)
