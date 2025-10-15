"""Tests for parallel execution in LikelihoodExpansion."""

from __future__ import annotations

import numpy as np

from derivkit.forecasting.expansions import LikelihoodExpansion


def observables_fn(theta):
    """Deterministic mapping (not index-sensitive)."""
    x = float(np.sum(theta))
    return np.array([x, x**2, np.cos(x)])


def observables_fn_indexed(theta):
    """Deterministic mapping that is index-sensitive (exposes ordering issues)."""
    return np.array([float(np.dot(theta, np.arange(1, len(theta) + 1)))])


def test_order1_equivalent_across_workers():
    """Order-1 Fisher is identical across worker counts."""
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 8), np.eye(3))
    F1 = le.get_forecast_tensors(1, n_workers=1)
    F4 = le.get_forecast_tensors(1, n_workers=4)
    np.testing.assert_allclose(F1, F4, rtol=0, atol=1e-12)


def test_order2_equivalent_across_workers():
    """Order-2 DALI tensors are identical across worker counts."""
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 6), np.eye(3))
    G1, H1 = le.get_forecast_tensors(2, n_workers=1)
    G4, H4 = le.get_forecast_tensors(2, n_workers=4)
    np.testing.assert_allclose(G1, G4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H1, H4, rtol=0, atol=1e-12)


def test_ordering_preserved_order1():
    """Derivative ordering is preserved across worker counts."""
    le = LikelihoodExpansion(observables_fn_indexed, np.ones(5), np.eye(1))
    d1_serial = le._get_derivatives(order=1, n_workers=1)
    d1_threaded = le._get_derivatives(order=1, n_workers=4)
    np.testing.assert_allclose(d1_serial, d1_threaded, rtol=0, atol=0)


def test_ordering_preserved_order2():
    """Derivative ordering is preserved across worker counts."""
    le = LikelihoodExpansion(observables_fn_indexed, np.ones(6), np.eye(1))
    d2_serial = le._get_derivatives(order=2, n_workers=1)
    d2_threaded = le._get_derivatives(order=2, n_workers=4)
    np.testing.assert_allclose(d2_serial, d2_threaded, rtol=0, atol=0)

def test_small_workload_falls_back_to_serial():
    """Small workloads should fall back to serial execution."""
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 2), np.eye(3))
    F = le.get_forecast_tensors(1, n_workers=4)
    assert F.shape == (2, 2)

    G, H = le.get_forecast_tensors(2, n_workers=4)
    assert G.shape == (2, 2, 2)
    assert H.shape == (2, 2, 2, 2)

    # Don’t assert zeros — the function produces nonzero derivatives.
    # Instead check it’s finite and identical to serial:
    F1 = le.get_forecast_tensors(1, n_workers=1)
    np.testing.assert_allclose(F, F1, rtol=0, atol=1e-12)

    G1, H1 = le.get_forecast_tensors(2, n_workers=1)
    np.testing.assert_allclose(G, G1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H, H1, rtol=0, atol=1e-12)


def test_non_numeric_output_raises():
    """Non-numeric outputs should raise an error."""
    def bad_fn(theta):
        return "not a number"

    le = LikelihoodExpansion(bad_fn, np.linspace(0.1, 1.0, 4), np.eye(3))
    try:
        le.get_forecast_tensors(1, n_workers=4)
    except Exception as e:
        assert isinstance(e, (TypeError, ValueError))
    else:
        assert False, "Expected an exception for non-numeric output"

def test_incorrect_output_shape_raises():
    """Outputs of incorrect shape should raise an error."""
    def bad_shape_fn(theta):
        return np.array([[1, 2], [3, 4]])  # 2D instead of 1D

    le = LikelihoodExpansion(bad_shape_fn, np.linspace(0.1, 1.0, 4), np.eye(3))
    try:
        le.get_forecast_tensors(1, n_workers=4)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        assert False, "Expected a ValueError for incorrect output shape"

def test_large_workload_parallel_execution():
    """Large workloads should execute in parallel."""
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 10.0, 100), np.eye(3))
    F = le.get_forecast_tensors(1, n_workers=8)
    assert F.shape == (100, 100)

    G, H = le.get_forecast_tensors(2, n_workers=8)
    assert G.shape == (100, 100, 100)
    assert H.shape == (100, 100, 100, 100)

    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(H))


def observables_fn2(theta):
    """Deterministic mapping (not index-sensitive)."""
    s = float(np.sum(theta))
    return np.array([s, s*s])

def test_nworkers_edge_values_behave_serial():
    """n_workers values of None, 0, and -n should behave like n_workers=1."""
    le = LikelihoodExpansion(observables_fn2, np.linspace(0.1, 1.0, 7), np.eye(2))
    F_serial = le.get_forecast_tensors(1, n_workers=1)
    for n in (0, None, -3):
        F_edge = le.get_forecast_tensors(1, n_workers=n)
        np.testing.assert_allclose(F_serial, F_edge, rtol=0, atol=1e-12)
