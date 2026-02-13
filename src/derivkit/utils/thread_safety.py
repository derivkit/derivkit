"""Thread safety utilities."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

def wrap_with_lock(
    fn: Callable[..., T] | None,
    lock: Any = None,
) -> Callable[..., T] | None:
    """Wraps a function call with a lock."""
    if fn is None:
        return None
    lk = lock if lock is not None else threading.RLock()

    def wrapped(*args: Any, **kwargs: Any) -> T:
        """Wrapped function call."""
        with lk:
            return fn(*args, **kwargs)

    return wrapped
