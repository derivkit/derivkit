"""Tests for parallel execution in forecast tensor computations."""

from __future__ import annotations

import os

import numpy as np
import pytest

from derivkit.forecasting.forecast_core import (
    _get_derivatives,
    get_forecast_tensors,
)


def observables_fn(theta):
    """Returns a deterministic mapping (not index-sensitive)."""
    x = float(np.sum(theta))
    return np.array([x, x**2, np.cos(x)])


def observables_fn_indexed(theta):
    """Returns an index-sensitive mapping."""
    return np.array([float(np.dot(theta, np.arange(1, len(theta) + 1)))])


def observables_fn2(theta):
    """Returns a simple deterministic mapping."""
    s = float(np.sum(theta))
    return np.array([s, s*s])


def bad_fn(theta):
    """Returns a non-numeric output."""
    _ = theta
    return "not a number"


def bad_shape_fn(theta):
    """Returns an output of incorrect shape."""
    _ = theta
    return np.array([[1, 2], [3, 4]])  # 2D instead of 1D


def test_order1_equivalent_across_workers(extra_threads_ok):
    """Tests that order-1 DALI tensors are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    theta0 = np.linspace(0.1, 1.0, 8)
    cov = np.eye(3)
    fisher1 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=1)
    fisher2 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=2)
    np.testing.assert_allclose(fisher1, fisher2, rtol=0, atol=1e-12)


def test_order2_equivalent_across_workers(extra_threads_ok):
    """Tests that order-2 DALI tensors are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    theta0 = np.linspace(0.1, 1.0, 6)
    cov = np.eye(3)
    g1, h1 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=1)
    g2, h2 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=2)
    np.testing.assert_allclose(g1, g2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h1, h2, rtol=0, atol=1e-12)


def test_ordering_preserved_order1(extra_threads_ok):
    """Tests that derivative ordering is preserved across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    theta0 = np.ones(5)
    cov = np.eye(1)
    d1_serial = _get_derivatives(observables_fn_indexed, theta0, cov, order=1, n_workers=1)
    d1_threaded = _get_derivatives(observables_fn_indexed, theta0, cov, order=1, n_workers=2)

    np.testing.assert_allclose(d1_serial, d1_threaded, rtol=0, atol=0)


def test_ordering_preserved_order2(extra_threads_ok):
    """Tests that derivative ordering is preserved across worker counts (order=2)."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    theta0 = np.ones(6)
    cov = np.eye(1)
    d2_serial = _get_derivatives(observables_fn_indexed, theta0, cov, order=2, n_workers=1)
    d2_threaded = _get_derivatives(observables_fn_indexed, theta0, cov, order=2, n_workers=2)

    np.testing.assert_allclose(d2_serial, d2_threaded, rtol=0, atol=0)


def test_small_workload_falls_back_to_serial(extra_threads_ok):
    """Tests that small workloads fall back to serial execution."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 2)
    cov = np.eye(3)

    f = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=2)
    assert f.shape == (2, 2)

    g, h = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=2)
    assert g.shape == (2, 2, 2)
    assert h.shape == (2, 2, 2, 2)

    f1 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=1)
    np.testing.assert_allclose(f, f1, rtol=0, atol=1e-12)

    g1, h1 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=1)
    np.testing.assert_allclose(g, g1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h, h1, rtol=0, atol=1e-12)


def test_non_numeric_output_raises():
    """Tests that non-numeric outputs raise an error."""
    theta0 = np.linspace(0.1, 1.0, 4)
    cov = np.eye(3)

    try:
        get_forecast_tensors(bad_fn, theta0, cov, forecast_order=1, n_workers=4)
    except Exception as e:
        assert isinstance(e, (TypeError, ValueError))
    else:
        assert False, "Expected an exception for non-numeric output"


def test_incorrect_output_shape_raises():
    """Tests that outputs of incorrect shape raise an error."""
    theta0 = np.linspace(0.1, 1.0, 4)
    cov = np.eye(3)

    try:
        get_forecast_tensors(bad_shape_fn, theta0, cov, forecast_order=1, n_workers=4)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        assert False, "Expected a ValueError for incorrect output shape"


def test_large_workload_parallel_execution(extra_threads_ok, threads_ok):
    """Tests that large workloads execute in parallel and produce valid outputs."""
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

    n = int(os.getenv("DK_TEST_PARALLEL_N", "24"))
    theta0 = np.linspace(0.1, 10.0, n)
    cov = np.eye(3)

    f = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=n_workers)
    g, h = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=n_workers)

    assert f.shape == (n, n)
    assert g.shape == (n, n, n)
    assert h.shape == (n, n, n, n)

    assert np.all(np.isfinite(f))
    assert np.all(np.isfinite(g))
    assert np.all(np.isfinite(h))


def test_nworkers_edge_values_behave_serial():
    """Tests that n_workers edge values (0, None, negative) behave like serial execution."""
    theta0 = np.linspace(0.1, 1.0, 7)
    cov = np.eye(2)

    f_serial = get_forecast_tensors(observables_fn2, theta0, cov, forecast_order=1, n_workers=1)

    for n in (0, None, -3):
        f_edge = get_forecast_tensors(observables_fn2, theta0, cov, forecast_order=1, n_workers=n)
        np.testing.assert_allclose(f_serial, f_edge, rtol=0, atol=1e-12)


def test_parallel_is_deterministic_across_runs_order1(extra_threads_ok):
    """Tests that order-1 tensors are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 32)
    cov = np.eye(3)

    y_a = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=3)
    y_b = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=3)

    np.testing.assert_allclose(y_a, y_b, rtol=0, atol=0)


def test_parallel_matches_serial_for_prime_length(extra_threads_ok):
    """Ensures parallel and serial results match even when workloads can't split evenly.

    A prime-length input (97) guarantees that the data cannot be divided evenly
    among worker threads, which stresses chunking logic and verifies that uneven
    partitions do not alter numerical results.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")
    n = 97  # prime to stress uneven splits
    theta0 = np.linspace(0.1, 5.0, n)
    cov = np.eye(3)

    y_serial = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=1)
    y_par = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=4)

    np.testing.assert_allclose(y_par, y_serial, rtol=0, atol=0)


def test_parallel_caps_or_handles_too_many_workers(extra_threads_ok, threads_ok):
    """Tests that requesting too many workers is capped/handled gracefully."""
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

    excessive_workers = max(n_workers * 4, n_workers + 1)

    theta0 = np.linspace(0.1, 2.0, 21)
    cov = np.eye(3)

    y_serial = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=1)
    y_par = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=1, n_workers=excessive_workers)

    np.testing.assert_allclose(y_par, y_serial, rtol=0, atol=0)


def test_parallel_is_deterministic_across_runs_order2(extra_threads_ok):
    """Tests that order-2 tensors are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 16)
    cov = np.eye(3)

    g1, h1 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=2)
    g2, h2 = get_forecast_tensors(observables_fn, theta0, cov, forecast_order=2, n_workers=2)

    np.testing.assert_allclose(g1, g2, rtol=0, atol=0)
    np.testing.assert_allclose(h1, h2, rtol=0, atol=0)


def test_parallel_preserves_order_with_prime_length_order2(extra_threads_ok):
    """Verifys derivative ordering is preserved even with uneven parallel splits (order=2).

    Using a prime-length input guarantees the workload cannot be divided evenly among
    worker threads. This stresses the chunking/merge logic for second-order derivatives
    (which involve cross-terms and more complex indexing) and ensures that stitching
    results back together does not permute or misalign derivative entries.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    n = int(os.getenv("DK_TEST_PRIME_N", "31"))
    theta0 = np.ones(n)
    cov = np.eye(1)

    d2_serial = _get_derivatives(observables_fn_indexed, theta0, cov, order=2, n_workers=1)
    d2_par = _get_derivatives(observables_fn_indexed, theta0, cov, order=2, n_workers=3)

    np.testing.assert_allclose(d2_serial, d2_par, rtol=0, atol=0)
