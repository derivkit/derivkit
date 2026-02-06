"""Unit tests for hessian in calculus.py."""

from functools import partial

import numpy as np
import pytest

import derivkit.calculus.hessian as hessmod
from derivkit.calculus.hessian import build_hessian


def f_quadratic(theta, mat: np.ndarray, vec: np.ndarray, const: float = 0.0):
    """Quadratic scalar f(x) = 0.5 x^T A x + b^T x + c (returns float)."""
    x = np.asarray(theta, float).reshape(-1)
    a = np.asarray(mat, float)
    b = np.asarray(vec, float).reshape(-1)
    return 0.5 * x @ a @ x + b @ x + float(const)


def make_pure_quadratic(a: np.ndarray):
    """Tests that Hessian of f(x)=0.5 x^T A x is A."""
    a = np.asarray(a, float)
    b = np.zeros(a.shape[0], dtype=float)
    return partial(f_quadratic, mat=a, vec=b, const=0.0)


def make_sum_squares(n: int):
    """Tests that Hessian of f(x)=x^T x = sum_i x_i^2 is 2*I."""
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
    return np.array([x[0] ** 2, x.sum()], dtype=float)


def f_cubic2d(theta):
    """A cubic function in 2D for numeric reference."""
    x, y = np.asarray(theta, float)
    return x**2 + 3.0 * x * y + 2.0 * y**3


def rng_seed42():
    """Get a random number generator with a fixed seed (42) for reproducibility."""
    return np.random.default_rng(seed=42)


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


def _canonicalize_hessian(h, p: int):
    """Return Hessian as (out_dim, p, p), accepting common axis orders.

    This keeps tests robust if build_hessian returns (p, p), (out, p, p),
    (p, p, out), or (p, out, p). Scalar outputs are treated as out_dim=1.
    """
    h = np.asarray(h, float)

    if h.ndim == 2:
        if h.shape != (p, p):
            raise AssertionError(f"Expected (p, p) = {(p, p)}, got {h.shape}.")
        return h[None, :, :]

    if h.ndim != 3:
        raise AssertionError(
            f"Expected 2D or 3D Hessian, got ndim={h.ndim}, shape={h.shape}."
        )

    a, b, c = h.shape

    # (out, p, p)
    if b == p and c == p:
        return h

    # (p, p, out) -> (out, p, p)
    if a == p and b == p:
        return h.transpose(2, 0, 1)

    # (p, out, p) -> (out, p, p)
    if a == p and c == p:
        return h.transpose(1, 0, 2)

    # scalar-ish variants like (1,p,p) or (p,p,1)
    if (a, b, c) == (1, p, p):
        return h
    if (a, b, c) == (p, p, 1):
        return h.transpose(2, 0, 1)
    if (a, b, c) == (p, 1, p):
        return h.transpose(1, 0, 2)

    raise AssertionError(f"Unrecognized Hessian shape {h.shape} for p={p}.")


def test_hessian_quadratic_symmetrizes_matrix():
    """Tests that Hessian of quadratic symmetrizes A."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))  # deliberately non-symmetric
    x0 = np.array([0.2, -0.1, 0.7], dtype=float)

    f = make_pure_quadratic(a)
    h = build_hessian(f, x0, n_workers=1)

    h_true = 0.5 * (a + a.T)
    np.testing.assert_allclose(h, h_true, rtol=1e-10, atol=1e-8)

    # Show skew-symmetric parts don't matter:
    k = rng.normal(size=(3, 3))
    k = 0.5 * (k - k.T)  # skew-symmetric
    a2 = a + k  # same quadratic form; same Hessian
    f2 = make_pure_quadratic(a2)
    h2 = build_hessian(f2, x0, n_workers=1)

    np.testing.assert_allclose(h2, h_true, rtol=1e-10, atol=1e-8)


def test_hessian_matches_numeric_reference_on_cubic2d():
    """Cubic 2D Hessian matches numeric reference."""
    x0 = np.array([0.3, -0.4], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=1)
    h_ref = num_hessian(f_cubic2d, x0, eps=1e-5)

    h_c = _canonicalize_hessian(h, p=2)
    assert h_c.shape == (1, 2, 2)
    assert np.allclose(h_c[0], h_ref, atol=5e-5, rtol=5e-6)


def test_hessian_workers_invariance(extra_threads_ok):
    """Check that the Hessian is invariant to the number of workers."""
    x0 = np.array([0.25, -0.15], dtype=float)
    if not extra_threads_ok:
        pytest.skip("cannot spawn extra threads here")

    h1 = build_hessian(f_cubic2d, x0, n_workers=1)
    h2 = build_hessian(f_cubic2d, x0, n_workers=3)

    h1_c = _canonicalize_hessian(h1, p=2)
    h2_c = _canonicalize_hessian(h2, p=2)

    assert h1_c.shape == h2_c.shape == (1, 2, 2)
    assert np.allclose(h1_c, h2_c, atol=2e-10, rtol=0.0)


def test_hessian_is_symmetric():
    """Hessian is symmetric."""
    x0 = np.array([0.1, 0.2], dtype=float)
    h = build_hessian(f_cubic2d, x0, n_workers=2)

    h_c = _canonicalize_hessian(h, p=2)
    assert np.allclose(h_c[0], h_c[0].T, rtol=0.0, atol=0.0)


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
    assert np.allclose(h1, h2)


def test_hessian_vector_output_values_correct():
    """Vector-output Hessian has correct shape and numerical values.

    f_vector_out(theta) = [x0**2, x0 + x1]
      -> H[0] = [[2, 0],
                 [0, 0]]
         H[1] = zeros((2, 2))
    """
    x = np.array([0.2, 0.1])
    h = build_hessian(f_vector_out, x, n_workers=1)

    # ground truth as (out,p,p)
    h_true = np.zeros((2, 2, 2), dtype=float)
    h_true[0] = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=float)
    h_true[1] = np.zeros((2, 2), dtype=float)

    h_c = _canonicalize_hessian(h, p=2)
    assert h_c.shape == (2, 2, 2)
    assert np.allclose(h_c, h_true, rtol=0.0, atol=1e-8)


def test_hessian_raises_on_nonfinite_output():
    """Raises on non-finite output."""
    with pytest.raises(FloatingPointError):
        build_hessian(f_nonfinite, np.array([0.0, 1.0]))


def test_hessian_input_validation_empty_theta():
    """Raises on empty theta input."""
    with pytest.raises(ValueError):
        f_ss_1 = make_sum_squares(1)
        build_hessian(f_ss_1, np.array([]))


def poly_trig_simple(x):
    """A simple function to test parallel vs serial Hessian."""
    return float((x[0]**3) / 3 + 2 * x[0] * x[1] + np.cos(x[1]))


def test_build_hessian_parallel_equals_serial(extra_threads_ok):
    """Tests that parallel and serial Hessian computations yield the same result."""
    t = np.array([0.1, 0.3])
    if not extra_threads_ok:
        pytest.skip("cannot spawn extra threads here")

    hess1 = build_hessian(poly_trig_simple, t, n_workers=1)
    hess4 = build_hessian(poly_trig_simple, t, n_workers=8)
    np.testing.assert_allclose(hess4, hess1, rtol=1e-8, atol=1e-10)


def test_build_hessian_uses_cache_theta_function(monkeypatch):
    """Tests that build_hessian wraps the forward model with cache_theta_function."""
    calls = {"n": 0}
    seen_wrapped = {"ok": False}

    real = hessmod.cache_theta_function

    def spy_cache_theta_function(fn):
        calls["n"] += 1
        wrapped = real(fn)

        def wrapped_spy(theta):
            seen_wrapped["ok"] = True
            return wrapped(theta)

        return wrapped_spy

    monkeypatch.setattr(hessmod, "cache_theta_function", spy_cache_theta_function)

    x0 = np.array([0.2, -0.3], dtype=float)
    _ = build_hessian(f_cubic2d, x0, n_workers=1)

    assert calls["n"] == 1
    assert seen_wrapped["ok"]


def test_build_hessian_uses_cache_theta_function_for_tensor_outputs(monkeypatch):
    """Tests that tensor-output path also wraps the full forward model for caching."""
    calls = {"n": 0}

    real = hessmod.cache_theta_function

    def spy_cache_theta_function(fn):
        calls["n"] += 1
        return real(fn)

    monkeypatch.setattr(hessmod, "cache_theta_function", spy_cache_theta_function)

    def f_tensor(theta):
        x = np.asarray(theta, float)
        return np.array([x[0] ** 2, x.sum()], dtype=float)

    x0 = np.array([0.2, 0.1], dtype=float)
    _ = build_hessian(f_tensor, x0, n_workers=1)

    assert calls["n"] == 1
