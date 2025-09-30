"""Unit tests for forecasting/calculus.py (Jacobian)."""

import numpy as np
import pytest

from derivkit.forecasting.calculus import jacobian


def test_jacobian_linear_map():
    """Test jacobian on a linear map."""
    # f(theta) = A @ theta  =>  J = A (constant)
    A = np.array([[1.0, -2.0, 0.5],
                  [0.0,  3.0, 1.0]])
    def f(th):
        th = np.asarray(th, dtype=float)
        return A @ th
    theta0 = np.array([0.3, -0.7, 1.2])
    J = jacobian(f, theta0, n_workers=1)
    assert J.shape == (A.shape[0], theta0.size)
    assert np.allclose(J, A, atol=1e-12, rtol=0.0)


def test_jacobian_analytic():
    """Test jacobian on a function with known analytic Jacobian."""
    # f([x,y]) = [x^2, sin(y), x*y]
    # J = [[2x,   0 ],
    #      [ 0 , cos y],
    #      [ y ,  x  ]]
    def f(th):
        x, y = th
        return np.array([x**2, np.sin(y), x*y], dtype=float)
    x0, y0 = 0.4, -0.2
    theta0 = np.array([x0, y0], dtype=float)
    J = jacobian(f, theta0, n_workers=2)
    J_true = np.array([[2*x0, 0.0],
                       [0.0,  np.cos(y0)],
                       [y0,   x0]], dtype=float)
    assert J.shape == (3, 2)
    assert np.allclose(J, J_true, atol=1e-5, rtol=1e-6)


def test_jacobian_empty_theta_raises():
    """Test jacobian raises ValueError on empty theta0."""
    def f(th):
        return np.array([1.0])
    with pytest.raises(ValueError):
        jacobian(f, np.array([]))


def num_jacobian(f, theta, eps=1e-6):
    """Plain central-diff numerical Jacobian for test reference."""
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(f(theta), dtype=float)
    m, n = f0.size, theta.size
    J = np.empty((m, n), dtype=float)
    for j in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fp = np.asarray(f(tp), dtype=float)
        fm = np.asarray(f(tm), dtype=float)
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def test_jacobian_linear_random_matrix():
    """For a linear map f(θ)=Aθ, the Jacobian should equal A exactly."""
    rng = np.random.default_rng(42)
    A = rng.normal(size=(4, 3))
    def f(th):
        return A @ np.asarray(th, dtype=float)
    theta0 = rng.normal(size=3)
    J = jacobian(f, theta0, n_workers=1)
    assert J.shape == (4, 3)
    assert np.allclose(J, A, atol=1e-12, rtol=0.0)


def test_jacobian_matches_numeric_reference():
    """Nonlinear check vs a simple central-diff reference."""
    def f(th):
        x, y, z = th
        return np.array([
            x*y + np.sin(z),
            x**2 + np.cos(y),
            np.exp(z) * y
        ], dtype=float)

    theta0 = np.array([0.3, -0.7, 0.25], dtype=float)
    J = jacobian(f, theta0, n_workers=1)
    J_ref = num_jacobian(f, theta0, eps=1e-5)
    # Allow modest tolerance due to different step control in DerivativeKit
    assert J.shape == (3, 3)
    assert np.allclose(J, J_ref, atol=8e-5, rtol=5e-6)


def test_jacobian_single_output_vector_len1():
    """Vector-valued function of length 1 should still work and produce shape (1, n)."""
    def f(th):
        x, y = th
        return np.array([x**2 + y], dtype=float)  # length-1 vector
    theta0 = np.array([0.4, -0.2], dtype=float)
    J = jacobian(f, theta0, n_workers=1)
    J_true = np.array([[2*theta0[0], 1.0]])
    assert J.shape == (1, 2)
    assert np.allclose(J, J_true, atol=1e-6, rtol=1e-6)


def test_jacobian_workers_invariance():
    """Jacobian should be invariant to n_workers choice (within tolerance)."""
    def f(th):
        x, y = th
        return np.array([x**2, np.sin(y), x*y], dtype=float)
    theta0 = np.array([0.4, -0.2], dtype=float)
    J1 = jacobian(f, theta0, n_workers=1)
    J2 = jacobian(f, theta0, n_workers=3)
    assert J1.shape == J2.shape == (3, 2)
    assert np.allclose(J1, J2, atol=2e-6, rtol=2e-6)


def test_jacobian_shape_and_type_errors():
    """Input validation: empty theta should raise; non-array-like output should error upstream."""
    def f_vec(th):
        return np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        jacobian(f_vec, np.array([]))


def test_jacobian_chain_rule_linear_wrapper():
    """Check chain rule via a linear pre-transform: g(θ)=f(Bθ) ⇒ J_g = J_f|_{Bθ} · B."""
    B = np.array([[1.0, -1.0],
                  [0.5,  2.0]])
    def f(u):
        # f: R^2 -> R^2
        x, y = u
        return np.array([x**2 + y, np.sin(x) + np.cos(y)], dtype=float)

    def g(theta):
        return f(B @ np.asarray(theta, dtype=float))

    theta0 = np.array([0.2, -0.3], dtype=float)
    # J_g(θ) (computed)
    Jg = jacobian(g, theta0, n_workers=1)
    # J_f(u) at u=Bθ
    u0 = B @ theta0
    def f_wrt_u(u):
        return f(u)
    Jf = jacobian(f_wrt_u, u0, n_workers=1)  # shape (2,2)
    Jg_theory = Jf @ B
    assert np.allclose(Jg, Jg_theory, atol=8e-5, rtol=5e-6)


def test_jacobian_raises_on_nonfinite_output():
    """Jacobian should raise if function returns non-finite values."""
    def f(th):
        x, y = th
        return np.array([x, np.nan*x + y], dtype=float)
    with pytest.raises((FloatingPointError, ValueError)):
        jacobian(f, np.array([1.0, 2.0]))


def test_jacobian_does_not_modify_input():
    """Ensure jacobian does not modify theta0 in-place."""
    def f(th): return np.array([th[0] + th[1]])
    theta0 = np.array([0.1, 0.2])
    theta_copy = theta0.copy()
    _ = jacobian(f, theta0, n_workers=1)
    assert np.array_equal(theta0, theta_copy)


def test_jacobian_accepts_list_and_row_vector():
    """Ensure jacobian accepts list input and 2D row vector input."""
    def f(th):
        x, y = th
        return np.array([x + y, x - y])
    J1 = jacobian(f, [0.3, -0.7])
    J2 = jacobian(f, np.array([[0.3, -0.7]]))  # weird shape
    assert J1.shape == (2, 2)
    assert np.allclose(J1, J2)
