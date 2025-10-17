"""Unit tests for hessian forecasting/calculus.py."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.calculus import build_hessian


def f_quadratic(theta, mat: np.ndarray, vec: np.ndarray, const: float = 0.0):
    """A simple quadratic function with known Hessian."""
    x = np.asarray(theta, float).reshape(-1)
    a = np.asarray(mat, float)
    b = np.asarray(vec, float).reshape(-1)
    return 0.5 * x @ a @ x + b @ x + float(const)


def f_quadratic_raw(theta, mat: np.ndarray):
    """A simple quadratic function with known Hessian, no linear or const term."""
    x = np.asarray(theta, float).reshape(-1)
    a = np.asarray(mat, float)
    return 0.5 * x @ a @ x


def f_nonfinite(theta):
    """Produces a non-finite output."""
    x = np.asarray(theta, float)
    return np.nan + x.sum()  # force NaN


def f_vector_out(theta):
    """Produces a vector output."""
    x = np.asarray(theta, float)
    return np.array([x[0] ** 2, x.sum()], dtype=float)  # not scalar → should raise


def f_cubic2d(theta):
    """A cubic function in 2D for numeric reference."""
    x, y = np.asarray(theta, float)
    return x**2 + 3.0 * x * y + 2.0 * y**3


def f_sum_squares(theta):
    """A simple sum-of-squares function."""
    x = np.asarray(theta, float).reshape(-1)
    return float(x @ x)


def rng_seed42(seed=42):
    """Get a random number generator with a fixed seed for reproducibility."""
    return np.random.default_rng(seed=seed)


def num_hessian(f, theta, eps=1e-5):
    """Central-difference numerical Hessian for reference."""
    theta = np.asarray(theta, float).reshape(-1)
    n = theta.size
    hess = np.empty((n, n), dtype=float)
    # diagonal terms
    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = 1.0
        fp = f(theta + eps * e)
        fm = f(theta - eps * e)
        f0 = f(theta)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (eps ** 2)
    # off-diagonals (mixed partials)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = 1.0
            ej[j] = 1.0
            fpp = f(theta + eps * ei + eps * ej)
            fpm = f(theta + eps * ei - eps * ej)
            fmp = f(theta - eps * ei + eps * ej)
            fmm = f(theta - eps * ei - eps * ej)
            val = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
            hess[i, j] = val
            hess[j, i] = val
    return hess

def test_hessian_from_raw_quadratic_nonsymmetric_matrix():
    """Check that the Hessian of a quadratic form matches the (symmetrized) matrix."""
    a = np.array([[1.0, 2.0, -1.0],
                  [0.0, 3.0,  4.0],
                  [5.0, 0.5,  2.0]], dtype=float)
    x0 = np.array([0.2, -0.1, 0.7], dtype=float)
    f = partial(f_quadratic_raw, mat=a)
    h = build_hessian(f, x0, n_workers=1)
    h_true = 0.5 * (a + a.T)
    assert np.allclose(h, h_true, atol=1e-10, rtol=0.0)


def test_hessian_matches_numeric_reference_on_cubic2d():
    """Check that the Hessian of a cubic function in 2D matches numeric reference."""
    x0 = np.array([0.3, -0.4], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=1)
    h_ref = num_hessian(f_cubic2d, x0, eps=1e-5)
    assert h.shape == (2, 2)
    assert np.allclose(h, h_ref, atol=5e-5, rtol=5e-6)


def test_hessian_workers_invariance():
    """Check that the Hessian is invariant to the number of workers."""
    x0 = np.array([0.25, -0.15], dtype=float)
    h1 = build_hessian(f_cubic2d, x0, n_workers=1)
    h2 = build_hessian(f_cubic2d, x0, n_workers=3)
    assert h1.shape == h2.shape == (2, 2)
    assert np.allclose(h1, h2, atol=2e-10, rtol=0.0)


def test_hessian_is_symmetric():
    """Check that the Hessian is symmetric."""
    x0 = np.array([0.1, 0.2], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=2)
    assert np.allclose(h, h.T, atol=1e-10)


def test_hessian_does_not_modify_input():
    """Check that the input vector is not modified by build_hessian."""
    x0 = np.array([0.1, 0.2, -0.3], dtype=float)
    x_copy = x0.copy()
    _ = build_hessian(f_sum_squares, x0, n_workers=1)
    assert np.array_equal(x0, x_copy)


def test_hessian_accepts_list_and_row_vector():
    """Check that the Hessian function accepts list and row vector inputs."""
    h1 = build_hessian(f_sum_squares, [0.3, -0.7])
    h2 = build_hessian(f_sum_squares, np.array([[0.3, -0.7]]))
    assert h1.shape == (2, 2)
    assert np.allclose(h1, h2)


def test_hessian_raises_on_vector_output():
    """Check that the Hessian function raises on non-scalar output."""
    with pytest.raises(TypeError):
        build_hessian(f_vector_out, np.array([0.2, 0.1]))


def test_hessian_raises_on_nonfinite_output():
    """Check that the Hessian function raises on non-finite output."""
    with pytest.raises((FloatingPointError, ValueError)):
        build_hessian(f_nonfinite, np.array([0.0, 1.0]))


def test_hessian_input_validation_empty_theta():
    """Input validation: empty theta should raise."""
    with pytest.raises(ValueError):
        build_hessian(f_sum_squares, np.array([]))
