"""Unit tests for `utils.thread_safety`."""

from __future__ import annotations

import threading
import time
from typing import Any

from derivkit.utils.thread_safety import wrap_with_lock


def test_wrap_with_lock_returns_none_when_fn_is_none() -> None:
    """Tests that wrap_with_lock returns None when fn is None."""
    wrapped = wrap_with_lock(None)
    assert wrapped is None


def test_wrap_with_lock_basic_call_passthrough() -> None:
    """Tests that wrap_with_lock passes through the function call."""
    def f(x: int, y: int = 2) -> int:
        return x + y

    wrapped = wrap_with_lock(f)
    assert wrapped is not None
    assert wrapped(3) == 5
    assert wrapped(3, y=10) == 13


def test_wrap_with_lock_uses_provided_lock_object() -> None:
    """Tests that wrap_with_lock uses the provided lock object."""
    lock = threading.Lock()
    acquired_inside: dict[str, Any] = {"value": None}

    def f() -> bool:
        # If wrap_with_lock is using *this* lock, it must already be held here.
        acquired_inside["value"] = lock.acquire(blocking=False)
        # If we could acquire it, then it was NOT held => wrong lock used.
        if acquired_inside["value"] is True:
            lock.release()
        return True

    wrapped = wrap_with_lock(f, lock=lock)
    assert wrapped is not None
    assert wrapped() is True
    assert acquired_inside["value"] is False


def test_wrap_with_lock_serializes_concurrent_calls() -> None:
    """Tests that wrap_with_lock serializes concurrent calls."""
    in_region = 0
    max_in_region = 0
    mu = threading.Lock()

    def f(delay_s: float) -> None:
        nonlocal in_region, max_in_region
        with mu:
            in_region += 1
            max_in_region = max(max_in_region, in_region)
        time.sleep(delay_s)
        with mu:
            in_region -= 1

    wrapped = wrap_with_lock(f)  # uses internal RLock
    assert wrapped is not None

    n = 8
    delay_s = 0.02
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        wrapped(delay_s)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_in_region == 1


def test_wrap_with_lock_allows_independent_locks_to_run_concurrently() -> None:
    """Tests that wrap_with_lock can be used with independent locks."""
    in_region = 0
    max_in_region = 0
    mu = threading.Lock()

    def f(delay_s: float) -> None:
        nonlocal in_region, max_in_region
        with mu:
            in_region += 1
            max_in_region = max(max_in_region, in_region)
        time.sleep(delay_s)
        with mu:
            in_region -= 1

    w1 = wrap_with_lock(f, lock=threading.RLock())
    w2 = wrap_with_lock(f, lock=threading.RLock())
    assert w1 is not None and w2 is not None

    barrier = threading.Barrier(2)

    def t1() -> None:
        barrier.wait()
        w1(0.05)

    def t2() -> None:
        barrier.wait()
        w2(0.05)

    a = threading.Thread(target=t1)
    b = threading.Thread(target=t2)
    a.start()
    b.start()
    a.join()
    b.join()

    assert max_in_region >= 2


def test_wrap_with_lock_allows_reentrant_calls_with_rlock() -> None:
    """Tests that wrap_with_lock works with RLock objects."""
    lock = threading.RLock()

    def f() -> int:
        with lock:
            return 1

    wrapped = wrap_with_lock(f, lock=lock)
    assert wrapped is not None
    assert wrapped() == 1
