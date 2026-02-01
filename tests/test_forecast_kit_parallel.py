"""Tests for parallel execution in forecast tensor computations.

These tests exercise the public ForecastKit API (not forecast_core internals)
to verify that:

- Fisher and DALI tensors are identical across worker counts,
- parallel execution is deterministic across repeated runs,
- uneven chunking (prime-length workloads) does not change results, and
- invalid model outputs are rejected with clear exceptions.

Conventions
-----------
ForecastKit follows the introduced-at-order convention for DALI tensors:

- ``dali[1] == (F,)``
- ``dali[2] == (D1, D2)``
- ``dali[3] == (T1, T2, T3)``

All tensor axes have length ``p = len(theta0)``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit


def observables_fn(theta):
    """Returns a deterministic mapping (not index-sensitive)."""
    x = float(np.sum(theta))
    return np.array([x, x**2, np.cos(x)], dtype=float)


def observables_fn_indexed(theta):
    """Returns an index-sensitive mapping."""
    theta = np.asarray(theta, dtype=float)
    w = np.arange(1, len(theta) + 1, dtype=float)
    return np.array([float(np.dot(theta, w))], dtype=float)


def observables_fn2(theta):
    """Returns a simple deterministic mapping."""
    s = float(np.sum(theta))
    return np.array([s, s * s], dtype=float)


def bad_fn(theta):
    """Returns a non-numeric output."""
    _ = theta
    return "not a number"


def bad_shape_fn(theta):
    """Returns an output of incorrect shape."""
    _ = theta
    return np.array([[1, 2], [3, 4]], dtype=float)  # 2D instead of 1D


def test_order1_equivalent_across_workers(extra_threads_ok):
    """Tests that Fisher matrices are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 8)
    cov = np.eye(3)

    fk1 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)
    fk2 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f1 = fk1.fisher(n_workers=1)
    f2 = fk2.fisher(n_workers=2)

    np.testing.assert_allclose(f1, f2, rtol=0, atol=1e-12)


def test_order2_equivalent_across_workers(extra_threads_ok):
    """Tests that order-2 DALI tensors are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 6)
    cov = np.eye(3)

    fk1 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)
    fk2 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    dali1 = fk1.dali(forecast_order=2, n_workers=1)
    dali2 = fk2.dali(forecast_order=2, n_workers=2)

    f1 = dali1[1][0]
    f2 = dali2[1][0]
    d1_1, d2_1 = dali1[2]
    d1_2, d2_2 = dali2[2]

    np.testing.assert_allclose(f1, f2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d1_1, d1_2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d2_1, d2_2, rtol=0, atol=1e-12)


def test_order3_equivalent_across_workers(extra_threads_ok):
    """Tests that order-3 DALI tensors are identical across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 5)
    cov = np.eye(3)

    fk1 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)
    fk2 = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    dali1 = fk1.dali(forecast_order=3, n_workers=1)
    dali2 = fk2.dali(forecast_order=3, n_workers=2)

    f1 = dali1[1][0]
    f2 = dali2[1][0]
    d1_1, d2_1 = dali1[2]
    d1_2, d2_2 = dali2[2]
    t1_1, t2_1, t3_1 = dali1[3]
    t1_2, t2_2, t3_2 = dali2[3]

    np.testing.assert_allclose(f1, f2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d1_1, d1_2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d2_1, d2_2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t1_1, t1_2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t2_1, t2_2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(t3_1, t3_2, rtol=0, atol=1e-12)


def test_ordering_preserved_order1(extra_threads_ok):
    """Tests that Fisher is invariant for index-sensitive models across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.ones(5)
    cov = np.eye(1)

    fk1 = ForecastKit(function=observables_fn_indexed, theta0=theta0, cov=cov)
    fk2 = ForecastKit(function=observables_fn_indexed, theta0=theta0, cov=cov)

    f_serial = fk1.fisher(n_workers=1)
    f_threaded = fk2.fisher(n_workers=2)

    np.testing.assert_allclose(f_serial, f_threaded, rtol=0, atol=0)


def test_ordering_preserved_order2(extra_threads_ok):
    """Tests that order-2 DALI tensors are invariant for index-sensitive models across worker counts."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.ones(6)
    cov = np.eye(1)

    fk1 = ForecastKit(function=observables_fn_indexed, theta0=theta0, cov=cov)
    fk2 = ForecastKit(function=observables_fn_indexed, theta0=theta0, cov=cov)

    dali_serial = fk1.dali(forecast_order=2, n_workers=1)
    dali_threaded = fk2.dali(forecast_order=2, n_workers=2)

    f1 = dali_serial[1][0]
    f2 = dali_threaded[1][0]
    d1_1, d2_1 = dali_serial[2]
    d1_2, d2_2 = dali_threaded[2]

    np.testing.assert_allclose(f1, f2, rtol=0, atol=0)
    np.testing.assert_allclose(d1_1, d1_2, rtol=0, atol=0)
    np.testing.assert_allclose(d2_1, d2_2, rtol=0, atol=0)


def test_small_workload_behaves_like_serial(extra_threads_ok):
    """Tests that small workloads behave like serial execution even if n_workers>1."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 2)
    cov = np.eye(3)

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f_par = fk.fisher(n_workers=2)
    assert f_par.shape == (2, 2)

    dali_par = fk.dali(forecast_order=2, n_workers=2)
    f2 = dali_par[1][0]
    d1, d2 = dali_par[2]
    assert f2.shape == (2, 2)
    assert d1.shape == (2, 2, 2)
    assert d2.shape == (2, 2, 2, 2)

    f_ser = fk.fisher(n_workers=1)
    np.testing.assert_allclose(f_par, f_ser, rtol=0, atol=1e-12)

    dali_ser = fk.dali(forecast_order=2, n_workers=1)
    np.testing.assert_allclose(dali_par[1][0], dali_ser[1][0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(dali_par[2][0], dali_ser[2][0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(dali_par[2][1], dali_ser[2][1], rtol=0, atol=1e-12)


def test_non_numeric_output_raises():
    """Tests that non-numeric outputs raise an error."""
    theta0 = np.linspace(0.1, 1.0, 4)
    cov = np.eye(3)

    fk = ForecastKit(function=bad_fn, theta0=theta0, cov=cov)

    with pytest.raises((TypeError, ValueError)):
        _ = fk.fisher(n_workers=4)


def test_incorrect_output_shape_raises():
    """Tests that outputs of incorrect shape raise an error."""
    theta0 = np.linspace(0.1, 1.0, 4)
    cov = np.eye(3)

    fk = ForecastKit(function=bad_shape_fn, theta0=theta0, cov=cov)

    with pytest.raises(ValueError, match="shape"):
        _ = fk.fisher(n_workers=4)


def test_large_workload_parallel_execution(extra_threads_ok, threads_ok):
    """Tests that large workloads execute in parallel and produce finite outputs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

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

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f = fk.fisher(n_workers=n_workers)
    dali = fk.dali(forecast_order=2, n_workers=n_workers)

    f2 = dali[1][0]
    d1, d2 = dali[2]

    assert f.shape == (n, n)
    assert f2.shape == (n, n)
    assert d1.shape == (n, n, n)
    assert d2.shape == (n, n, n, n)

    assert np.all(np.isfinite(f))
    assert np.all(np.isfinite(f2))
    assert np.all(np.isfinite(d1))
    assert np.all(np.isfinite(d2))


def test_nworkers_edge_values_behave_serial():
    """Tests that n_workers edge values (``0``, ``None``, negative) behave like serial execution."""
    theta0 = np.linspace(0.1, 1.0, 7)
    cov = np.eye(2)

    fk = ForecastKit(function=observables_fn2, theta0=theta0, cov=cov)

    f_serial = fk.fisher(n_workers=1)

    for n_workers in (0, None, -3):
        f_edge = fk.fisher(n_workers=n_workers)  # normalize_workers should coerce to serial
        np.testing.assert_allclose(f_serial, f_edge, rtol=0, atol=1e-12)


def test_parallel_is_deterministic_across_runs_order1(extra_threads_ok):
    """Tests that Fisher matrices are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 32)
    cov = np.eye(3)

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f_a = fk.fisher(n_workers=3)
    f_b = fk.fisher(n_workers=3)

    np.testing.assert_allclose(f_a, f_b, rtol=0, atol=0)


def test_parallel_matches_serial_for_prime_length(extra_threads_ok):
    """Ensures parallel and serial Fisher results match for prime-length inputs.

    A prime-length input (97) guarantees the workload cannot split evenly among
    workers, stressing chunking/merge logic.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    n = 97
    theta0 = np.linspace(0.1, 5.0, n)
    cov = np.eye(3)

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f_serial = fk.fisher(n_workers=1)
    f_par = fk.fisher(n_workers=4)

    np.testing.assert_allclose(f_par, f_serial, rtol=0, atol=0)


def test_parallel_caps_or_handles_too_many_workers(extra_threads_ok, threads_ok):
    """Tests that requesting too many workers is capped/handled gracefully."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    n_cpu = os.cpu_count() or 2
    cap = min(8, max(2, n_cpu // 2))

    for k in range(cap, 1, -1):
        if threads_ok(k):
            feasible = k
            break
    else:
        pytest.skip("cannot spawn >=2 threads despite extra_threads_ok")

    excessive_workers = max(feasible * 4, feasible + 1)

    theta0 = np.linspace(0.1, 2.0, 21)
    cov = np.eye(3)

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    f_serial = fk.fisher(n_workers=1)
    f_par = fk.fisher(n_workers=excessive_workers)

    np.testing.assert_allclose(f_par, f_serial, rtol=0, atol=0)


def test_parallel_is_deterministic_across_runs_order2(extra_threads_ok):
    """Tests that order-2 DALI tensors are stable across repeated parallel runs."""
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    theta0 = np.linspace(0.1, 1.0, 16)
    cov = np.eye(3)

    fk = ForecastKit(function=observables_fn, theta0=theta0, cov=cov)

    dali_a = fk.dali(forecast_order=2, n_workers=2)
    dali_b = fk.dali(forecast_order=2, n_workers=2)

    f_a = dali_a[1][0]
    f_b = dali_b[1][0]
    d1_a, d2_a = dali_a[2]
    d1_b, d2_b = dali_b[2]

    np.testing.assert_allclose(f_a, f_b, rtol=0, atol=0)
    np.testing.assert_allclose(d1_a, d1_b, rtol=0, atol=0)
    np.testing.assert_allclose(d2_a, d2_b, rtol=0, atol=0)


def test_parallel_preserves_order_with_prime_length_order2(extra_threads_ok):
    """Verifies order-2 DALI tensors match between serial and uneven parallel splits.

    Using a prime-length input stresses chunking and merge logic for second-order
    derivatives and ensures that stitching results back together does not permute
    parameter ordering.
    """
    if not extra_threads_ok:
        pytest.skip("no extra threads available")

    n = int(os.getenv("DK_TEST_PRIME_N", "31"))
    theta0 = np.ones(n)
    cov = np.eye(1)

    fk = ForecastKit(function=observables_fn_indexed, theta0=theta0, cov=cov)

    dali_serial = fk.dali(forecast_order=2, n_workers=1)
    dali_par = fk.dali(forecast_order=2, n_workers=3)

    f1 = dali_serial[1][0]
    f2 = dali_par[1][0]
    d1_1, d2_1 = dali_serial[2]
    d1_2, d2_2 = dali_par[2]

    np.testing.assert_allclose(f1, f2, rtol=0, atol=0)
    np.testing.assert_allclose(d1_1, d1_2, rtol=0, atol=0)
    np.testing.assert_allclose(d2_1, d2_2, rtol=0, atol=0)
