"""Tests for derivkit.utils.concurrency."""

from __future__ import annotations

import os
import time

from derivkit.utils.concurrency import (
    normalize_workers,
    parallel_execute,
)


def _identity(x: int) -> int:
    """Identity function used for testing."""
    return x


def test_normalize_workers_various_inputs():
    """Tests that normalize_workers handles various inputs correctly."""
    assert normalize_workers(1) == 1
    assert normalize_workers(4) == 4
    assert normalize_workers(0) == 1
    assert normalize_workers(-3) == 1
    assert normalize_workers(None) == 1
    assert normalize_workers(2.7) == 2


def test_parallel_execute_backend_processes_ok():
    """Tests that backend='processes' returns the correct results (not performance)."""
    out = parallel_execute(_identity, [(1,), (2,)], n_workers=2, backend="processes")
    assert sorted(out) == [1, 2]


def _pid_worker_sleep(_x: int) -> int:
    """Returns PID after a short sleep (module-scope for pickling)."""
    time.sleep(0.05)
    return os.getpid()


def test_parallel_execute_uses_multiple_processes_when_n_workers_gt_1():
    """Tests that parallel_execute uses multiple processes when n_workers > 1."""
    n = 4
    out = parallel_execute(
        _pid_worker_sleep,
        [(i,) for i in range(n)],
        n_workers=2,
        backend="processes",
    )
    assert len(set(out)) > 1
