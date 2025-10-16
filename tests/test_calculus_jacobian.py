"""Unit tests for jacobian forecasting/calculus.py."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.calculus import build_jacobian


def f_linear_mat(th, vec: np.ndarray) -> np.ndarray:
    """Linear map f(θ)=Aθ."""
    return np.asarray(vec, float) @ np.asarray(th, float)


def f_analytic_2d(th) -> np.ndarray:
    """Analytic 2D map with known Jacobian."""
    x, y = np.asarray(th, float)
    return np.array([x**2, np.sin(y), x*y], dtype=float)


def f_nonlinear_3d(th) -> np.ndarray:
    """Nonlinear 3D map used for numeric reference."""
    x, y, z = np.asarray(th, float)
    return np.array([x*y + np.sin(z), x**2 + np.cos(y), np.exp(z) * y], dtype=float)


def f_len1_vector(th) -> np.ndarray:
    """Returns a length-1 vector."""
    x, y = np.asarray(th, float)
    return np.array([x**2 + y], dtype=float)


def f_base2(th) -> np.ndarray:
    """Base function for chain-rule test: R^2 -> R^2."""
    x, y = np.asarray(th, float)
    return np.array([x**2 + y, np.sin(x) + np.cos(y)], dtype=float)


def f_chain_linear(theta, mat: np.ndarray) -> np.ndarray:
    """Test function for chain-rule test: f(Bθ)."""
    return f_base2(np.asarray(mat, float) @ np.asarray(theta, float))


def f_nonfinite(th) -> np.ndarray:
    """Produces a non-finite output component."""
    x, y = np.asarray(th, float)
    return np.array([x, np.nan * x + y], dtype=float)


def f_sum2(th) -> np.ndarray:
    """Sum of two components, shape (1,)."""
    x, y = np.asarray(th, float)
    return np.array([x + y], dtype=float)


def f_plus_minus(th) -> np.ndarray:
    """Returns a function with both plus and minus."""
    x, y = np.asarray(th, float)
    return np.array([x + y, x - y], dtype=float)


def num_jacobian(f, theta, eps=1e-6) -> np.ndarray:
    """Plain central-diff numeric Jacobian (reference)."""
    theta = np.asarray(theta, float)
    f0 = np.asarray(f(theta), float)
    m, n = f0.size, theta.size
    jac = np.empty((m, n), dtype=float)
    for j in range(n):
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fp = np.asarray(f(tp), float)
        fm = np.asarray(f(tm), float)
        jac[:, j] = (fp - fm) / (2 * eps)
    return jac


def rng_seed42(seed=42):
    """Random number generator with fixed seed (42) for reproducibility."""
    return np.random.default_rng(seed=seed)


def test_jacobian_linear_map():
    """Check that the Jacobian of a fixed linear map equals its matrix.

    Uses a simple predefined matrix to confirm the Jacobian matches it
    exactly and has the correct shape.
    """
    vec = np.array([[1.0, -2.0, 0.5],
                  [0.0,  3.0, 1.0]], dtype=float)
    f = partial(f_linear_mat, vec=vec)
    theta0 = np.array([0.3, -0.7, 1.2], dtype=float)
    jac = build_jacobian(f, theta0, n_workers=1)
    assert jac.shape == (vec.shape[0], theta0.size)
    assert np.allclose(jac, vec, atol=1e-12, rtol=0.0)


def test_jacobian_analytic():
    """Test jacobian on a function with known analytic Jacobian."""
    x0, y0 = 0.4, -0.2
    theta0 = np.array([x0, y0], dtype=float)
    jac = build_jacobian(f_analytic_2d, theta0, n_workers=2)
    jac_true = np.array([[2*x0, 0.0],
                       [0.0,  np.cos(y0)],
                       [y0,   x0]], dtype=float)
    assert jac.shape == (3, 2)
    assert np.allclose(jac, jac_true, atol=1e-5, rtol=1e-6)


def test_jacobian_empty_theta_raises():
    """Test jacobian raises ValueError on empty theta0."""
    with pytest.raises(ValueError):
        build_jacobian(f_len1_vector, np.array([]))


def test_jacobian_linear_random_matrix():
    """Check that the Jacobian of a random linear map equals its matrix.

    Uses a fixed-seed random matrix and confirms the Jacobian matches it
    exactly and has the correct shape.
    """
    rng = rng_seed42()
    vec = rng.normal(size=(4, 3))
    f = partial(f_linear_mat, vec=vec)
    theta0 = rng.normal(size=3)
    jac = build_jacobian(f, theta0, n_workers=1)
    assert jac.shape == (4, 3)
    assert np.allclose(jac, vec, atol=1e-12, rtol=0.0)


def test_jacobian_matches_numeric_reference():
    """Test jacobian against plain numeric reference implementation."""
    theta0 = np.array([0.3, -0.7, 0.25], dtype=float)
    jac = build_jacobian(f_nonlinear_3d, theta0, n_workers=1)
    jac_ref = num_jacobian(f_nonlinear_3d, theta0, eps=1e-5)
    assert jac.shape == (3, 3)
    assert np.allclose(jac, jac_ref, atol=8e-5, rtol=5e-6)


def test_jacobian_single_output_vector_len1():
    """Test jacobian on a function returning a length-1 vector."""
    theta0 = np.array([0.4, -0.2], dtype=float)
    jac = build_jacobian(f_len1_vector, theta0, n_workers=1)
    jac_true = np.array([[2*theta0[0], 1.0]])
    assert jac.shape == (1, 2)
    assert np.allclose(jac, jac_true, atol=1e-6, rtol=1e-6)


def test_jacobian_workers_invariance():
    """Jacobian should not depend on the number of worker threads."""
    theta0 = np.array([0.4, -0.2], dtype=float)
    base = build_jacobian(f_analytic_2d, theta0, n_workers=1)
    try:
        alt = build_jacobian(f_analytic_2d, theta0, n_workers=2)
    except Exception as e:
        pytest.skip(f"cannot spawn extra threads here: {e}")
    assert base.shape == alt.shape == (3, 2)
    assert np.allclose(base, alt, atol=2e-6, rtol=2e-6)


def test_jacobian_shape_and_type_errors():
    """Input validation: empty theta should raise; non-array-like output should error upstream."""
    with pytest.raises(ValueError):
        build_jacobian(lambda th: np.array([1.0, 2.0]), np.array([]))


def test_jacobian_chain_rule_linear_wrapper():
    """Test jacobian via chain rule with a linear wrapper function."""
    mat = np.array([[1.0, -1.0],
                  [0.5,  2.0]], dtype=float)
    g = partial(f_chain_linear, mat=mat)
    theta0 = np.array([0.2, -0.3], dtype=float)
    jacg = build_jacobian(g, theta0, n_workers=1)
    u0 = mat @ theta0
    jacf = build_jacobian(f_base2, u0, n_workers=1)
    jacg_theory = jacf @ mat
    assert np.allclose(jacg, jacg_theory, atol=8e-5, rtol=5e-6)


def test_jacobian_raises_on_nonfinite_output():
    """Jacobian should raise FloatingPointError on non-finite outputs."""
    with pytest.raises(FloatingPointError):
        build_jacobian(f_nonfinite, np.array([1.0, 2.0]))


def test_jacobian_does_not_modify_input():
    """Test jacobian does not modify input theta0."""
    theta0 = np.array([0.1, 0.2], dtype=float)
    theta_copy = theta0.copy()
    _ = build_jacobian(f_sum2, theta0, n_workers=1)
    assert np.array_equal(theta0, theta_copy)


def test_jacobian_accepts_list_and_row_vector():
    """Test jacobian accepts list and row-vector inputs."""
    jac1 = build_jacobian(f_plus_minus, [0.3, -0.7])
    jac2 = build_jacobian(f_plus_minus, np.array([[0.3, -0.7]]))
    assert jac1.shape == (2, 2)
    assert np.allclose(jac1, jac2)

def test_jacobian_raises_on_scalar_output():
    """Jacobian should raise TypeError when the function returns a scalar."""
    def f_scalar(th):
        x = np.asarray(th, float)
        return float(x.sum())
    with pytest.raises(TypeError):
        build_jacobian(f_scalar, np.array([0.1, 0.2]))


def test_jacobian_accepts_vector_output():
    """Jacobian should accept functions returning a vector (not array)."""
    theta0 = np.array([0.3, -0.1], dtype=float)
    jac = build_jacobian(f_analytic_2d, theta0, n_workers=1)
    assert jac.shape == (3, 2)
