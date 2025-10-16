"""Pytest configuration file with a fixture to check thread spawning capability."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

__all__ = ["extra_threads_ok"]


@pytest.fixture(scope="session")
def extra_threads_ok():
    """True if we can actually start >=2 threads; False otherwise."""
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(lambda: None) for _ in range(2)]
            for f in futs:
                f.result(timeout=1)
        return True
    except (RuntimeError, MemoryError, OSError, TimeoutError):
        # RuntimeError: can't start new thread
        # MemoryError/OSError: system/resource limits
        # TimeoutError: threads didn't start or hang
        return False
