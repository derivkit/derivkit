"""Unit tests for Gaussian Process based differentiation in DerivativeKit."""

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit
from derivkit.gaussian_process.core import (
    gp_choose_hyperparams,
    gp_derivative,
    gp_fit,
)
from derivkit.gaussian_process.kernels import get_kernel


def f(x):
    """Test function: sin(x)."""
    return np.sin(x)


def df(x):
    """Analytic first derivative: cos(x)."""
    return np.cos(x)


def d2f(x):
    """Analytic second derivative: -sin(x)."""
    return -np.sin(x)


def test_gp_first_derivative_close_to_truth():
    """Tests that GP derivative is close to analytic derivative for sin(x)."""
    x0 = 0.3
    dk = DerivativeKit(f, x0)
    d_gp = dk.differentiate(
        method="gp",
        order=1,
        kernel="rbf",
        kernel_params={"length_scale": 0.4, "output_scale": 1.0},
        samples=None, n_points=15, spacing="auto", base_abs=0.5,
        normalize=True, optimize=False, return_variance=False,
    )
    assert abs(d_gp - df(x0)) < 1e-2  # loosen/tighten as you like

@pytest.mark.parametrize("x0", [0.0, 0.3, 0.9])
def test_gp_second_derivative_close_to_truth(x0):
    """Tests that GP second derivative is close to analytic second derivative for sin(x)."""
    dk = DerivativeKit(f, float(x0))
    n_points = 121
    base_abs = 1.5
    kernel_params = {"length_scale": 0.4, "output_scale": 1.0}

    # --- 1) Run through DerivativeKit (your usual path)
    d2_gp, d2_var = dk.differentiate(
        method="gp",
        order=2,
        kernel="rbf",
        kernel_params=kernel_params,
        n_points=n_points,
        spacing="chebyshev",
        base_abs=base_abs,
        noise_variance=1e-6,
        normalize=False,
        optimize=True,
        return_variance=True,
    )
    truth = d2f(x0)
    err = float(abs(d2_gp - truth))
    sigma = float(np.sqrt(d2_var))
    print("\n[DK PATH]")
    print(f"x0={x0:.3f}  d2_gp={float(d2_gp):+.6e}  truth={truth:+.6e}  "
          f"err={err:.6e}  sigma={sigma:.6e}  max(5e-3,5σ)={max(5e-3, 5*sigma):.6e}")

    # --- 2) Build the exact grid we expect and replicate the GP by hand
    def _cheb_nodes(n, half, center):
        k = np.arange(n)
        # Chebyshev-Gauss nodes on [-1,1]; center included for odd n after affine map
        xi = np.cos((2 * k + 1) * np.pi / (2 * n))
        return center + half * xi

    x = _cheb_nodes(n_points, base_abs, float(x0)).reshape(-1, 1)
    y = f(x.ravel())

    # Let the GP choose hyperparams on this exact window
    theta_opt, noise_opt = gp_choose_hyperparams(
        x, y,
        kernel="rbf",
        init_params=kernel_params,
        init_noise=1e-6,
        normalize=False,
        n_restarts=8,
        random_state=123,
    )

    state = gp_fit(
        x, y,
        kernel="rbf",
        kernel_params=theta_opt,
        noise_variance=noise_opt,
        normalize=False,
        jitter=1e-12,
        variance_floor=1e-18,
    )
    d2_mu_manual, d2_var_manual = gp_derivative(state, np.array([x0]), order=2, axis=0)

    # Diagnostics: window, theta*, noise*, cov conditioning
    ker = get_kernel("rbf")
    cov = ker.cov_value_value(x, x, theta_opt) + (noise_opt + state["jitter"]) * np.eye(len(x))
    eig = np.linalg.eigvalsh(cov)
    cond = (eig.max() / max(eig.min(), 1e-300))
    print("[MANUAL GP]")
    print(f"window=[{x.min():+.3f},{x.max():+.3f}]  span={x.max()-x.min():.3f}  n={len(x)}")
    print(f"theta_opt={theta_opt}  noise_opt={noise_opt:.3e}  cond(K)≈{cond:.3e}  "
          f"min_eig={eig.min():.3e}  max_eig={eig.max():.3e}")
    print(f"d2_mu_manual={float(d2_mu_manual):+.6e}  d2_var_manual={float(d2_var_manual):.6e}  "
          f"err_manual={abs(float(d2_mu_manual)-truth):.6e}  "
          f"sigma_manual={float(np.sqrt(d2_var_manual)):.6e}")

    # 3) Assertion: keep the variance-aware band you wanted
    tol_floor = 5e-3
    assert err <= max(tol_floor, 5.0 * sigma)
