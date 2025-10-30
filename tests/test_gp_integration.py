"""Tests for Gaussian Process based differentiation in DerivativeKit."""

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def f(x): return np.sin(x)
def df(x): return np.cos(x)
def d2f(x): return -np.sin(x)

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

def test_gp_second_derivative_close_to_truth():
    """Tests that GP second derivative is close to analytic second derivative for sin(x)."""
    x0 = 0.3
    dk = DerivativeKit(f, x0)
    d2_gp = dk.differentiate(
        method="gp",
        order=2,
        kernel="rbf",
        kernel_params={"length_scale": 0.4, "output_scale": 1.0},
        samples=None, n_points=21, spacing="auto", base_abs=0.6,
        normalize=True, optimize=False, return_variance=False,
    )
    assert abs(d2_gp - d2f(x0)) < 5e-2
