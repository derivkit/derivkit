"""Unit tests for forecasting/calculus.py (Jacobian)."""

import numpy as np
import pytest

from derivkit.forecasting.calculus import gradient, hessian, jacobian


def test_gradient_linear_form():
    """Test gradient on a linear form."""
    # f(theta) = c^T theta  =>  grad = c
    c = np.array([1.5, -2.0, 0.25], dtype=float)

    def f(th):
        th = np.asarray(th, dtype=float)
        return float(c @ th)

    theta0 = np.array([0.3, -0.7, 1.2])
    g = gradient(f, theta0, n_workers=1)
    assert g.shape == c.shape
    np.testing.assert_allclose(g, c, atol=1e-12, rtol=0.0)


def test_gradient_analytic_scalar():
    """Test gradient on a function with known analytic gradient."""
    # f(x, y) = x^2 + sin(y) + x*y
    # ∂f/∂x = 2x + y ; ∂f/∂y = cos(y) + x
    def f(th):
        x, y = th
        return float(x**2 + np.sin(y) + x * y)

    x0, y0 = 0.4, -0.2
    theta0 = np.array([x0, y0], dtype=float)
    g = gradient(f, theta0, n_workers=2)
    g_true = np.array([2 * x0 + y0, np.cos(y0) + x0], dtype=float)
    np.testing.assert_allclose(g, g_true, atol=1e-5, rtol=1e-7)


def test_gradient_equals_row_of_jacobian_for_scalar():
    """Teest that gradient matches Jacobian row for scalar-valued function."""
    # For scalar f: R^n -> R, grad f == J_f (row) of [f]
    def f(th):
        x, y, z = th
        return float(x**2 + np.sin(y) + np.exp(z) + x * y)

    theta0 = np.array([0.3, -0.7, 0.25], dtype=float)
    grad = gradient(f, theta0, n_workers=1)

    def f_vec(th):
        return np.array([f(th)], dtype=float)  # length-1 vector

    jac = jacobian(f_vec, theta0, n_workers=2)  # shape (1, n)
    np.testing.assert_allclose(grad, jac[0, :], atol=2e-6, rtol=2e-6)


def test_gradient_workers_invariance():
    """Test that gradient is invariant to n_workers choice (within tolerance)."""
    def f(th):
        x, y = th
        return float(x**2 + np.sin(y) + 0.5 * x * y)

    theta0 = np.array([0.3, -0.7])
    grad1 = gradient(f, theta0, n_workers=1)
    grad2 = gradient(f, theta0, n_workers=4)
    np.testing.assert_allclose(grad1, grad2, atol=2e-6, rtol=2e-6)


def test_gradient_empty_theta_raises():
    """Test gradient raises ValueError on empty theta0."""
    def f(th):
        return 1.0
    with pytest.raises(ValueError):
        gradient(f, np.array([]))


def test_gradient_raises_on_vector_output():
    """Input validation: gradient should raise if output is not scalar."""
    # gradient expects scalar-valued function
    def f_vec(th):
        return np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        gradient(f_vec, np.array([0.0, 1.0]))


def test_gradient_raises_on_nonfinite():
    """Gradient should raise if function returns non-finite values."""
    def f(th):
        x, y = th
        return float(x + (np.nan * y))
    with pytest.raises((FloatingPointError, TypeError, ValueError)):
        gradient(f, np.array([1.0, 2.0]))


def test_gradient_accepts_list_and_row_vector():
    """Ensure gradient accepts list input and 2D row vector input."""
    def f(th):
        x, y = th
        return float(x + y + x * y)

    grad1 = gradient(f, [0.3, -0.7])
    grad2 = gradient(f, np.array([[0.3, -0.7]]))  # odd shape -> reshaped inside
    np.testing.assert_allclose(grad1, grad2, atol=1e-12, rtol=0.0)


def test_gradient_does_not_modify_input():
    """Ensure gradient does not modify theta0 in-place."""
    def f(th):
        x, y = th
        return float(x + y + x * y)

    theta0 = np.array([0.1, 0.2])
    before = theta0.copy()
    _ = gradient(f, theta0, n_workers=1)
    assert np.array_equal(theta0, before)


def test_jacobian_linear_map():
    """Test jacobian on a linear map."""
    # f(theta) = A @ theta  =>  J = A (constant)
    mat = np.array([[1.0, -2.0, 0.5],
                  [0.0,  3.0, 1.0]])
    def f(th):
        th = np.asarray(th, dtype=float)
        return mat @ th
    theta0 = np.array([0.3, -0.7, 1.2])
    jac = jacobian(f, theta0, n_workers=1)
    assert jac.shape == (mat.shape[0], theta0.size)
    assert np.allclose(jac, mat, atol=1e-12, rtol=0.0)


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
    jac = jacobian(f, theta0, n_workers=2)
    jac_true = np.array([[2*x0, 0.0],
                       [0.0,  np.cos(y0)],
                       [y0,   x0]], dtype=float)
    assert jac.shape == (3, 2)
    assert np.allclose(jac, jac_true, atol=1e-5, rtol=1e-6)


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
    jac = np.empty((m, n), dtype=float)
    for j in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fp = np.asarray(f(tp), dtype=float)
        fm = np.asarray(f(tm), dtype=float)
        jac[:, j] = (fp - fm) / (2 * eps)
    return jac


def test_jacobian_linear_random_matrix():
    """For a linear map f(θ)=Aθ, the Jacobian should equal A exactly."""
    rng = np.random.default_rng(42)
    mat = rng.normal(size=(4, 3))
    def f(th):
        return mat @ np.asarray(th, dtype=float)
    theta0 = rng.normal(size=3)
    jac = jacobian(f, theta0, n_workers=1)
    assert jac.shape == (4, 3)
    assert np.allclose(jac, mat, atol=1e-12, rtol=0.0)


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
    jac = jacobian(f, theta0, n_workers=1)
    jac_ref = num_jacobian(f, theta0, eps=1e-5)
    # Allow modest tolerance due to different step control in DerivativeKit
    assert jac.shape == (3, 3)
    assert np.allclose(jac, jac_ref, atol=8e-5, rtol=5e-6)


def test_jacobian_single_output_vector_len1():
    """Vector-valued function of length 1 should still work and produce shape (1, n)."""
    def f(th):
        x, y = th
        return np.array([x**2 + y], dtype=float)  # length-1 vector
    theta0 = np.array([0.4, -0.2], dtype=float)
    jac = jacobian(f, theta0, n_workers=1)
    jac_true = np.array([[2*theta0[0], 1.0]])
    assert jac.shape == (1, 2)
    assert np.allclose(jac, jac_true, atol=1e-6, rtol=1e-6)


def test_jacobian_workers_invariance():
    """Jacobian should be invariant to n_workers choice (within tolerance)."""
    def f(th):
        x, y = th
        return np.array([x**2, np.sin(y), x*y], dtype=float)
    theta0 = np.array([0.4, -0.2], dtype=float)
    jac1 = jacobian(f, theta0, n_workers=1)
    jac2 = jacobian(f, theta0, n_workers=3)
    assert jac1.shape == jac2.shape == (3, 2)
    assert np.allclose(jac1, jac2, atol=2e-6, rtol=2e-6)


def test_jacobian_shape_and_type_errors():
    """Input validation: empty theta should raise; non-array-like output should error upstream."""
    def f_vec(th):
        return np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        jacobian(f_vec, np.array([]))


def test_jacobian_chain_rule_linear_wrapper():
    """Check chain rule via a linear pre-transform: g(θ)=f(Bθ) ⇒ J_g = J_f|_{Bθ} · B."""
    mat = np.array([[1.0, -1.0],
                  [0.5,  2.0]])
    def f(u):
        # f: R^2 -> R^2
        x, y = u
        return np.array([x**2 + y, np.sin(x) + np.cos(y)], dtype=float)

    def g(theta):
        return f(mat @ np.asarray(theta, dtype=float))

    theta0 = np.array([0.2, -0.3], dtype=float)
    # J_g(θ) (computed)
    jacg = jacobian(g, theta0, n_workers=1)
    # J_f(u) at u=Bθ
    u0 = mat @ theta0
    def f_wrt_u(u):
        return f(u)
    jacf = jacobian(f_wrt_u, u0, n_workers=1)  # shape (2,2)
    jacg_theory = jacf @ mat
    assert np.allclose(jacg, jacg_theory, atol=8e-5, rtol=5e-6)


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
    jac1 = jacobian(f, [0.3, -0.7])
    jac2 = jacobian(f, np.array([[0.3, -0.7]]))  # weird shape
    assert jac1.shape == (2, 2)
    assert np.allclose(jac1, jac2)


def test_hessian_quadratic_form():
    """For a quadratic form f(θ)=0.5 θ^T Q θ + b^T θ + c, Hessian should equal Q."""
    # f(theta) = 0.5 * theta^T Q theta + b^T theta + c
    # Hessian = Q (constant, symmetric)
    mat = np.array([[3.0, 1.0, -0.5],
                  [1.0, 4.0,  0.2],
                  [-0.5, 0.2, 2.5]], dtype=float)
    mat = 0.5 * (mat + mat.T)  # ensure symmetric
    arr = np.array([0.1, -0.2, 0.3], dtype=float)

    def f(th):
        th = np.asarray(th, dtype=float)
        return float(0.5 * th @ mat @ th + arr @ th + 1.23)

    theta0 = np.array([0.2, -0.1, 0.7], dtype=float)
    hess = hessian(f, theta0, n_workers=1)
    assert hess.shape == (3, 3)
    # symmetry
    assert np.allclose(hess, hess.T, atol=1e-12, rtol=0.0)
    # equals Q
    assert np.allclose(hess, mat, atol=1e-8, rtol=0.0)


def test_hessian_mixed_partials_equal():
    """For a smooth function, mixed partials should be equal: ∂²f/∂x∂y = ∂²f/∂y∂x."""
    # f(x,y) = x^2 * y + sin(x*y)
    # ∂²f/∂x∂y should equal ∂²f/∂y∂x
    def f(th):
        x, y = th
        return float(x**2 * y + np.sin(x * y))

    theta0 = np.array([0.4, -0.3], dtype=float)
    hess = hessian(f, theta0, n_workers=2)
    assert hess.shape == (2, 2)
    assert np.allclose(hess[0, 1], hess[1, 0], atol=5e-6, rtol=5e-6)


def test_hessian_analytic_small_tol():
    """Test hessian on a function with known analytic Hessian."""
    # f(x,y) = exp(x) + cos(y) + x*y
    # Hessian =
    # [[exp(x), 1],
    #  [1,    -cos(y)]]
    def f(th):
        x, y = th
        return float(np.exp(x) + np.cos(y) + x * y)

    x0, y0 = 0.1, -0.2
    theta0 = np.array([x0, y0], dtype=float)
    hess = hessian(f, theta0, n_workers=3)
    hess_true = np.array([[np.exp(x0), 1.0],
                       [1.0, -np.cos(y0)]], dtype=float)
    assert hess.shape == (2, 2)
    assert np.allclose(hess, hess_true, atol=2e-5, rtol=2e-6)


def test_hessian_raises_on_vector_output():
    """Input validation: Hessian should raise if output is not scalar."""
    # hessian expects scalar-valued function
    def f_vec(th):
        return np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        hessian(f_vec, np.array([0.0, 1.0]))


def test_hessian_empty_theta_raises():
    """Test hessian raises ValueError on empty theta0."""
    def f(th):
        return 1.0
    with pytest.raises(ValueError):
        hessian(f, np.array([]))


def test_hessian_nonfinite_raises():
    """Hessian should raise if function returns non-finite values."""
    def f(th):
        x, y = th
        return float((x + y) * np.nan)
    with pytest.raises((FloatingPointError, TypeError, ValueError)):
        hessian(f, np.array([0.3, 0.2]))


def test_hessian_does_not_modify_input():
    """Ensure hessian does not modify theta0 in-place."""
    def f(th):
        x, y, z = th
        return float(x**2 + y**2 + z**2 + x * y - y * z)
    theta0 = np.array([0.1, 0.2, -0.3])
    theta_copy = theta0.copy()
    _ = hessian(f, theta0, n_workers=1)
    assert np.array_equal(theta0, theta_copy)
