"""Unit tests for RBF kernel in Gaussian Process module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from derivkit.gaussian_process.kernels.rbf_kernel import RBFKernel

RNG = np.random.default_rng(0)

def _params(length_scale=0.7, output_scale=1.3):
    """Helper to create parameter dicts for RBF kernel tests."""
    return {"length_scale": float(length_scale), "output_scale": float(output_scale)}

def _params_ard(ell_vec, output_scale=1.3):
    """Helper to create ARD parameter dicts for RBF kernel tests."""
    return {"length_scale": np.asarray(ell_vec, dtype=np.float64), "output_scale": float(output_scale)}

def test_value_value_basic_shapes_and_symmetry():
    """Basic shape and symmetry test for cov(value(x), value(x))."""
    ker = RBFKernel()
    x = RNG.normal(size=(5, 3))
    y = RNG.normal(size=(7, 3))
    p = _params(length_scale=0.9, output_scale=2.0)

    cov_xy = ker.cov_value_value(x, y, p)
    cov_yx = ker.cov_value_value(y, x, p)

    assert cov_xy.shape == (5, 7)
    assert cov_yx.shape == (7, 5)
    assert_allclose(cov_xy, cov_yx.T, rtol=0, atol=0)

def test_value_grad_antisymmetry_by_argument_swap():
    """Swapping arguments flips sign for cov(value(x), grad(y))."""
    ker = RBFKernel()
    x = RNG.normal(size=(6, 2))
    y = RNG.normal(size=(4, 2))
    p = _params(length_scale=0.8, output_scale=1.7)

    cov_v_g = ker.cov_value_grad(x, y, p, axis=1)
    # By swapping arguments, semantics flip to cov(value(y), grad(x));
    # For RBF, this should change sign.
    cov_v_g_swapped = ker.cov_value_grad(y, x, p, axis=1)

    assert cov_v_g.shape == (6, 4)
    assert_allclose(cov_v_g, -cov_v_g_swapped.T, rtol=0, atol=0)

def test_grad_grad_basic_and_same_point_limit():
    """Basic shape test and same-point limit for cov(grad(x), grad(x))."""
    ker = RBFKernel()
    x = RNG.normal(size=(5, 1))
    p = _params(length_scale=0.6, output_scale=1.2)

    cov_gg = ker.cov_grad_grad(x, x, p, axis=0)
    assert cov_gg.shape == (5, 5)

    # same-point diagonal: (1/ell^2) * s^2
    ell2 = p["length_scale"] ** 2
    s2 = p["output_scale"] ** 2
    expected_diag = s2 * (1.0 / ell2)
    assert_allclose(np.diag(cov_gg), expected_diag, rtol=0, atol=1e-12)

def test_value_hessdiag_same_point_limit():
    """Same-point limit for cov(value(x), hessdiag(x))."""
    ker = RBFKernel()
    x = RNG.normal(size=(7, 1))
    p = _params(length_scale=0.5, output_scale=2.0)

    cov_h = ker.cov_value_hessdiag(x, x, p, axis=0)
    assert cov_h.shape == (7, 7)

    # same point: - s^2 / ell^2 (since delta=0)
    ell2 = p["length_scale"] ** 2
    s2 = p["output_scale"] ** 2
    expected_diag = -s2 * (1.0 / ell2)
    assert_allclose(np.diag(cov_h), expected_diag, rtol=0, atol=1e-12)

def test_hessdiag_variance_samepoint():
    """Variance of hessdiag at same points."""
    ker = RBFKernel()
    x = RNG.normal(size=(4, 3))
    p = _params(length_scale=0.9, output_scale=1.1)

    v = ker.cov_hessdiag_samepoint(x, p, axis=2)
    assert v.shape == (4, 4)
    # strictly diagonal
    assert_allclose(v, np.diag(np.diag(v)), rtol=0, atol=0)

    # value = s^2 * 3 / ell^4 on each diagonal
    ell2 = p["length_scale"] ** 2
    s2 = p["output_scale"] ** 2
    expected = s2 * 3.0 / (ell2 ** 2)
    assert_allclose(np.diag(v), expected, rtol=0, atol=1e-12)

def test_ard_matches_isotropic_when_all_dims_equal():
    """Checks that ARD with equal length-scales matches isotropic case."""
    ker = RBFKernel()
    x = RNG.normal(size=(5, 3))
    y = RNG.normal(size=(6, 3))
    ell = 0.7
    s2 = 1.4

    p_iso = _params(length_scale=ell, output_scale=s2)
    p_ard = _params_ard([ell, ell, ell], output_scale=s2)

    cov_iso = ker.cov_value_value(x, y, p_iso)
    cov_ard = ker.cov_value_value(x, y, p_ard)
    assert_allclose(cov_iso, cov_ard, rtol=0, atol=0)

def test_value_grad_formula_matches_manual_expression_1d():
    """Checks that cov(value, grad) matches manual expression in 1D."""
    ker = RBFKernel()
    x = np.linspace(-1.1, 1.3, 7)[:, None]
    y = np.linspace(-0.9, 1.0, 5)[:, None]
    p = _params(length_scale=0.6, output_scale=2.3)

    cov_vv = ker.cov_value_value(x, y, p)
    cov_v_g = ker.cov_value_grad(x, y, p, axis=0)

    ell2 = p["length_scale"] ** 2
    delta = x[:, None, 0] - y[None, :, 0]
    cov_v_g_manual = (delta / ell2) * cov_vv

    assert_allclose(cov_v_g, cov_v_g_manual, rtol=0, atol=1e-12)

def test_grad_grad_formula_matches_manual_expression_1d():
    """Checks that cov(grad, grad) matches manual expression in 1D."""
    ker = RBFKernel()
    x = np.linspace(-1.0, 1.0, 6)[:, None]
    y = np.linspace(-0.8, 1.1, 4)[:, None]
    p = _params(length_scale=0.9, output_scale=1.9)

    cov_vv = ker.cov_value_value(x, y, p)
    cov_gg = ker.cov_grad_grad(x, y, p, axis=0)

    ell2 = p["length_scale"] ** 2
    delta2 = (x[:, None, 0] - y[None, :, 0]) ** 2
    cov_gg_manual = (1.0 / ell2 - delta2 / (ell2 ** 2)) * cov_vv

    assert_allclose(cov_gg, cov_gg_manual, rtol=0, atol=1e-12)

def test_value_hessdiag_formula_matches_manual_expression_1d():
    """Checks that cov(value, hessdiag) matches manual expression in 1D."""
    ker = RBFKernel()
    x = np.linspace(-1.0, 1.0, 5)[:, None]
    y = np.linspace(-0.6, 1.2, 3)[:, None]
    p = _params(length_scale=0.8, output_scale=1.5)

    cov_vv = ker.cov_value_value(x, y, p)
    cov_h = ker.cov_value_hessdiag(x, y, p, axis=0)

    ell2 = p["length_scale"] ** 2
    delta2 = (x[:, None, 0] - y[None, :, 0]) ** 2
    cov_h_manual = (delta2 / (ell2 ** 2) - 1.0 / ell2) * cov_vv

    assert_allclose(cov_h, cov_h_manual, rtol=0, atol=1e-12)

def test_value_grad_zero_on_same_points():
    """Tests that cov(value(x), grad(x)) is zero on the diagonal."""
    ker = RBFKernel()
    x = RNG.normal(size=(8, 2))
    p = _params(length_scale=0.6, output_scale=1.1)
    cov_v_g = ker.cov_value_grad(x, x, p, axis=1)
    # Only the diagonal is guaranteed to be ~0.
    assert_allclose(np.diag(cov_v_g), 0.0, rtol=0, atol=1e-14)


def test_axis_out_of_range_raises():
    """Axis out of range raises exception."""
    ker = RBFKernel()
    x = RNG.normal(size=(3, 2))
    p = _params(length_scale=0.6, output_scale=1.0)
    with pytest.raises(Exception):
        ker.cov_value_grad(x, x, p, axis=5)  # invalid axis
    with pytest.raises(Exception):
        ker.cov_grad_grad(x, x, p, axis=5)
    with pytest.raises(Exception):
        ker.cov_value_hessdiag(x, x, p, axis=5)
    with pytest.raises(Exception):
        ker.cov_hessdiag_samepoint(x, p, axis=5)

def test_input_dims_mismatch_raises():
    """Mismatched input dimensions raise exception."""
    ker = RBFKernel()
    x = RNG.normal(size=(3, 2))
    y = RNG.normal(size=(4, 3))  # mismatched n_dims
    p = _params(length_scale=0.6, output_scale=1.0)
    with pytest.raises(Exception):
        ker.cov_value_value(x, y, p)
