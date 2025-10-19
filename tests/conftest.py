"""Pytest configuration file with a fixture to check thread spawning capability."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

__all__ = ["extra_threads_ok"]


@pytest.fixture(scope="session")
def threads_ok():
    """Return a callable: threads_ok(n) -> True iff we can start >= n threads."""
    def _can_spawn(n: int = 2, timeout: float = 1.0) -> bool:
        try:
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(lambda: None) for _ in range(n)]
                for f in futs:
                    f.result(timeout=timeout)
            return True
        except (RuntimeError, MemoryError, OSError, TimeoutError):
            return False
    return _can_spawn


@pytest.fixture(scope="session")
def extra_threads_ok(threads_ok):
    """Convenience: True iff we can start >= 2 threads (common case)."""
    return threads_ok(2)
