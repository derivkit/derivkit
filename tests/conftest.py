"""Pytest configuration file with a fixture to check thread spawning capability."""

import inspect
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

import derivkit.utils.concurrency as conc

__all__ = ["extra_threads_ok"]


@pytest.fixture(autouse=True, scope="session")
def _limit_blas_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def threads_ok():
    """Return a callable that checks thread-spawning capability.

    The returned function has signature `check(n=2, timeout=1.0) -> bool` and
    returns True if at least `n` threads can be started and joined within `timeout`.
    """
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

# --- default to serial inside DerivKit unless test opts in via @pytest.mark.parallel ---
@pytest.fixture(autouse=True)
def _serial_by_default(request, monkeypatch):
    """Force n_workers=1 in DerivKit internals unless test is marked @pytest.mark.parallel."""
    if request.node.get_closest_marker("parallel"):
        return  # allow the test to exercise true parallel behavior

    try:


        if hasattr(conc, "parallel_execute"):
            orig = conc.parallel_execute
            sig = inspect.signature(orig)
            allowed = set(sig.parameters.keys())

            def _wrapped(*args, **kwargs):
                # ensure n_workers=1 regardless of call style
                if "n_workers" in allowed:
                    kwargs["n_workers"] = 1
                else:
                    # fallback: if third positional is n_workers, override it
                    if len(args) >= 3:
                        args = list(args)
                        args[2] = 1
                        args = tuple(args)
                # drop any kwargs the real function doesn't accept
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                return orig(*args, **kwargs)

            monkeypatch.setattr(conc, "parallel_execute", _wrapped, raising=True)

        # Optional: normalize module-level defaults if they exist
        if hasattr(conc, "DEFAULT_BACKEND"):
            monkeypatch.setattr(conc, "DEFAULT_BACKEND", "thread", raising=False)
        if hasattr(conc, "DEFAULT_WORKERS"):
            monkeypatch.setattr(conc, "DEFAULT_WORKERS", 1, raising=False)

    except Exception:
        pass
