"""Unit tests for hessian forecasting/calculus.py."""

from functools import partial

import numpy as np
import pytest

from derivkit.forecasting.calculus import build_hessian


def f_quadratic(theta, mat: np.ndarray, vec: np.ndarray, const: float = 0.0):
    """Quadratic scalar f(x) = 0.5 x^T A x + b^T x + c (returns float)."""
    x = np.asarray(theta, float).reshape(-1)
    a = np.asarray(mat, float)
    b = np.asarray(vec, float).reshape(-1)
    return 0.5 * x @ a @ x + b @ x + float(const)


def make_pure_quadratic(a: np.ndarray):
    """Test that Hessian of f(x)=0.5 x^T A x is A."""
    a = np.asarray(a, float)
    b = np.zeros(a.shape[0], dtype=float)
    return partial(f_quadratic, mat=a, vec=b, const=0.0)


def make_sum_squares(n: int):
    """Test that Hessian of f(x)=x^T x = sum_i x_i^2 is 2*I."""
    a = 2.0 * np.eye(n, dtype=float)
    b = np.zeros(n, dtype=float)
    return partial(f_quadratic, mat=a, vec=b, const=0.0)


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
        hess[i, i] = (fp - 2.0 * f0 + fm) / (eps**2)
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
            val = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            hess[i, j] = val
            hess[j, i] = val
    return hess


def test_hessian_quadratic_symmetrizes_matrix():
    """Test that Hessian of quadratic symmetrizes A."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))  # deliberately non-symmetric
    x0 = np.array([0.2, -0.1, 0.7], dtype=float)

    f = make_pure_quadratic(a)
    h = build_hessian(f, x0, n_workers=1)

    h_true = 0.5 * (a + a.T)
    assert np.allclose(h, h_true, atol=1e-10, rtol=0.0)

    # (Optional) show skew-symmetric parts don't matter:
    k = rng.normal(size=(3, 3))
    k = 0.5 * (k - k.T)  # skew-symmetric
    a2 = a + k  # same quadratic form; same Hessian
    f2 = make_pure_quadratic(a2)
    h2 = build_hessian(f2, x0, n_workers=1)
    assert np.allclose(h2, h_true, atol=1e-10, rtol=0.0)


def test_hessian_matches_numeric_reference_on_cubic2d():
    """Cubic 2D Hessian matches numeric reference."""
    x0 = np.array([0.3, -0.4], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=1)
    h_ref = num_hessian(f_cubic2d, x0, eps=1e-5)
    assert h.shape == (2, 2)
    assert np.allclose(h, h_ref, atol=5e-5, rtol=5e-6)


def test_hessian_workers_invariance():
    """Hessian is invariant to worker count."""
    x0 = np.array([0.25, -0.15], dtype=float)
    h1 = build_hessian(f_cubic2d, x0, n_workers=1)
    h2 = build_hessian(f_cubic2d, x0, n_workers=3)
    assert h1.shape == h2.shape == (2, 2)
    assert np.allclose(h1, h2, atol=2e-10, rtol=0.0)


def test_hessian_is_symmetric():
    """Hessian is symmetric."""
    x0 = np.array([0.1, 0.2], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=2)
    assert np.array_equal(h, h.T)


def test_hessian_does_not_modify_input():
    """Input vector is not modified by build_hessian."""
    x0 = np.array([0.1, 0.2, -0.3], dtype=float)
    x_copy = x0.copy()
    f_ss_3 = make_sum_squares(3)
    _ = build_hessian(f_ss_3, x0, n_workers=1)
    assert np.array_equal(x0, x_copy)


def test_hessian_accepts_list_and_row_vector():
    """Accepts list and row-vector inputs."""
    f_ss_2 = make_sum_squares(2)
    h1 = build_hessian(f_ss_2, [0.3, -0.7])
    h2 = build_hessian(f_ss_2, np.array([[0.3, -0.7]]))
    assert h1.shape == (2, 2)
    assert np.allclose(h1, h2)


def test_hessian_raises_on_vector_output():
    """Raises on non-scalar output."""
    with pytest.raises(TypeError):
        build_hessian(f_vector_out, np.array([0.2, 0.1]))


def test_hessian_raises_on_nonfinite_output():
    """Raises on non-finite output."""
    with pytest.raises((FloatingPointError, ValueError)):
        build_hessian(f_nonfinite, np.array([0.0, 1.0]))


def test_hessian_input_validation_empty_theta():
    """Raises on empty theta input."""
    with pytest.raises(ValueError):
        f_ss_1 = make_sum_squares(1)
        build_hessian(f_ss_1, np.array([]))
