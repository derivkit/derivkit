"""Tests for parallel execution in LikelihoodExpansion."""

from __future__ import annotations

import os

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def observables_fn(theta):
    """Deterministic mapping (not index-sensitive)."""
    x = float(np.sum(theta))
    return np.array([x, x**2, np.cos(x)])


def observables_fn_indexed(theta):
    """Deterministic mapping that is index-sensitive (exposes ordering issues)."""
    return np.array([float(np.dot(theta, np.arange(1, len(theta) + 1)))])


def test_order1_equivalent_across_workers(extra_threads_ok):
    """Order-1 Fisher is identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 8), np.eye(3))
    F1 = le.get_forecast_tensors(1, n_workers=1)
    F2 = le.get_forecast_tensors(1, n_workers=2)
    np.testing.assert_allclose(F1, F2, rtol=0, atol=1e-12)


def test_order2_equivalent_across_workers(extra_threads_ok):
    """Order-2 DALI tensors are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 6), np.eye(3))
    G1, H1 = le.get_forecast_tensors(2, n_workers=1)
    G2, H2 = le.get_forecast_tensors(2, n_workers=2)
    np.testing.assert_allclose(G1, G2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H1, H2, rtol=0, atol=1e-12)


def test_ordering_preserved_order1(extra_threads_ok):
    """Derivative ordering is preserved across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn_indexed, np.ones(5), np.eye(1))
    d1_serial = le._get_derivatives(order=1, n_workers=1)
    d1_threaded = le._get_derivatives(order=1, n_workers=2)
    np.testing.assert_allclose(d1_serial, d1_threaded, rtol=0, atol=0)


def test_ordering_preserved_order2(extra_threads_ok):
    """Derivative ordering is preserved across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn_indexed, np.ones(6), np.eye(1))
    d2_serial = le._get_derivatives(order=2, n_workers=1)
    d2_threaded = le._get_derivatives(order=2, n_workers=2)
    np.testing.assert_allclose(d2_serial, d2_threaded, rtol=0, atol=0)


def test_small_workload_falls_back_to_serial(extra_threads_ok):
    """Small workloads should fall back to serial execution."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 2), np.eye(3))
    F = le.get_forecast_tensors(1, n_workers=2)
    assert F.shape == (2, 2)

    G, H = le.get_forecast_tensors(2, n_workers=2)
    assert G.shape == (2, 2, 2)
    assert H.shape == (2, 2, 2, 2)

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


def test_large_workload_parallel_execution(extra_threads_ok, threads_ok):
    """Large workloads should execute in parallel."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    # Pick the highest feasible worker count up to a sensible cap.
    n_cpu = os.cpu_count() or 2
    preferred = min(8, max(2, n_cpu // 2))
    for k in range(preferred, 1, -1):
        if threads_ok(k):
            n_workers = k
            break
    else:
        pytest.skip("cannot spawn >=2 threads despite extra_threads_ok")

    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 10.0, 100), np.eye(3))
    F = le.get_forecast_tensors(1, n_workers=n_workers)
    assert F.shape == (100, 100)

    G, H = le.get_forecast_tensors(2, n_workers=n_workers)
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


def test_parallel_is_deterministic_across_runs_order1(extra_threads_ok):
    """Test that order-1 tensors are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 32), np.eye(3))
    y_a = le.get_forecast_tensors(1, n_workers=3)
    y_b = le.get_forecast_tensors(1, n_workers=3)
    np.testing.assert_allclose(y_a, y_b, rtol=0, atol=0)


def test_parallel_matches_serial_for_prime_length(extra_threads_ok):
    """Ensure parallel and serial results match even when workloads can't split evenly.

    A prime-length input (97) guarantees that the data cannot be divided evenly
    among worker threads, which stresses chunking logic and verifies that uneven
    partitions do not alter numerical results.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    n = 97  # prime to stress uneven splits
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 5.0, n), np.eye(3))
    y_serial = le.get_forecast_tensors(1, n_workers=1)
    y_par = le.get_forecast_tensors(1, n_workers=4)
    np.testing.assert_allclose(y_par, y_serial, rtol=0, atol=0)


def test_parallel_caps_or_handles_too_many_workers(extra_threads_ok, threads_ok):
    """Requesting more workers than CPUs should still behave correctly."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    n_cpu = os.cpu_count() or 2
    cap = min(8, max(2, n_cpu // 2))

    # pick the highest feasible k ≤ cap
    for k in range(cap, 1, -1):
        if threads_ok(k):
            n_workers = k
            break
    else:
        pytest.skip("cannot spawn >=2 threads despite extra_threads_ok")

    # Request far more than feasible to ensure capping/handling kicks in
    excessive_workers = max(n_workers * 4, n_workers + 1)

    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 2.0, 21), np.eye(3))
    y_serial = le.get_forecast_tensors(1, n_workers=1)
    y_par = le.get_forecast_tensors(1, n_workers=excessive_workers)

    np.testing.assert_allclose(y_par, y_serial, rtol=0, atol=0)


def test_parallel_is_deterministic_across_runs_order2(extra_threads_ok):
    """Order-2 tensors are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    le = LikelihoodExpansion(observables_fn, np.linspace(0.1, 1.0, 16), np.eye(3))
    g1, h1 = le.get_forecast_tensors(2, n_workers=2)
    g2, h2 = le.get_forecast_tensors(2, n_workers=2)
    np.testing.assert_allclose(g1, g2, rtol=0, atol=0)
    np.testing.assert_allclose(h1, h2, rtol=0, atol=0)


def test_parallel_preserves_order_with_prime_length_order2(extra_threads_ok):
    """Verify derivative ordering is preserved even with uneven parallel splits (order=2).

    Using a prime-length input guarantees the workload cannot be divided evenly among
    worker threads. This stresses the chunking/merge logic for second-order derivatives
    (which involve cross-terms and more complex indexing) and ensures that stitching
    results back together does not permute or misalign derivative entries.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    n = 59
    le = LikelihoodExpansion(observables_fn_indexed, np.ones(n), np.eye(1))
    d2_serial = le._get_derivatives(order=2, n_workers=1)
    d2_par = le._get_derivatives(order=2, n_workers=3)
    np.testing.assert_allclose(d2_serial, d2_par, rtol=0, atol=0)
