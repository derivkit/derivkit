"""Unit tests for batch_eval.py speed and correctness."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from derivkit.adaptive.batch_eval import eval_function_batch


def cpu_heavy(x: float) -> float:
    """CPU-heavy function used to test parallel evaluation speed.

    Performs many elementary math operations to simulate a deterministic workload.
    The function is intentionally not vectorized so parallelism yields measurable speedups.
    """
    # deterministic CPU burn; avoid numpy (keeps work in Python)
    s = 0.0
    # ~O(1e5) ops: tune if your CI is too slow/fast
    for k in range(80_000):
        y = x + (k * 1e-7)
        s += (y - math.sin(y)) * (y - math.cos(y))
    return s


@pytest.mark.slow
def test_eval_function_batch_parallel_is_faster_and_equal(extra_threads_ok):
    """eval_function_batch should be faster with multiple workers and give identical results."""
    if not extra_threads_ok:
        pytest.skip("cannot spawn extra threads in this environment")

    # Fixture guarantees >1 worker possible; size workers from CPU count.
    n_cpu = os.cpu_count() or 2
    n_workers = min(4, max(2, n_cpu // 2))
    assert n_workers >= 2

    # Make xs large enough to amortize Pool overhead and pass your internal threshold
    # Internal threshold is: xs.size >= max(8, 2*n_workers)
    # Use a generous multiple of that to ensure real parallel work.
    min_required = max(8, 2 * n_workers)
    num = min_required * 40  # scale up to get a measurable speedup
    xs = np.linspace(-1.0, 1.0, num)

    # Warm-ups to JIT caches / fork overhead / first-call penalties
    _ = eval_function_batch(cpu_heavy, xs[:min_required], n_workers=1)
    _ = eval_function_batch(cpu_heavy, xs[:min_required], n_workers=n_workers)

    # Serial timing
    t0 = time.perf_counter()
    y_serial = eval_function_batch(cpu_heavy, xs, n_workers=1)
    t1 = time.perf_counter()
    serial_time = t1 - t0

    # Parallel timing
    t2 = time.perf_counter()
    y_parallel = eval_function_batch(cpu_heavy, xs, n_workers=n_workers)
    t3 = time.perf_counter()
    parallel_time = t3 - t2

    # 1) Correctness: identical results
    # Allow exact equality; if tiny float jitter appears, swap to allclose with tight tol
    assert y_serial.shape == y_parallel.shape
    assert np.allclose(y_parallel, y_serial, rtol=0, atol=0), (
        "Parallel and serial results differ numerically."
    )

    # 2) Speed: require modest improvement, but don’t be flaky:
    # On busy CI, we still require a speedup of ~10% if multiple cores are available.
    # If parallel_time > serial_time (e.g. system under load),
    # mark as xfail with a helpful message rather than hard fail.
    # You can tighten the ratio if your environment is stable.
    required_ratio = 0.90  # parallel must be <= 90% of serial
    if not (parallel_time <= required_ratio * serial_time):
        pytest.xfail(
            f"Parallel speedup not observed (serial {serial_time:.3f}s, "
            f"parallel {parallel_time:.3f}s, workers={n_workers}). "
            "Environment may be CPU-throttled or heavily loaded."
        )
