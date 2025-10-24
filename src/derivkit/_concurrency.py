"""Concurrency management for derivative computations."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager

__all__ = [
    "set_default_inner_derivative_workers",
    "set_inner_derivative_workers",
    "resolve_inner_from_outer",
    "_inner_workers_var",
]

# Context-var and default
_inner_workers_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "derivkit_inner_workers", default=None
)
_DEFAULT_INNER_WORKERS: int | None = None


def set_default_inner_derivative_workers(n: int | None) -> None:
    """Sets the module-wide default for inner derivative workers.

    Args:
        n: Number of inner derivative workers, or None for automatic policy.

    Returns:
        None
    """
    global _DEFAULT_INNER_WORKERS
    _DEFAULT_INNER_WORKERS = None if n is None else int(n)


@contextmanager
def set_inner_derivative_workers(n: int | None):
    """Sets the inner derivative workers in a context.

    Args:
        n: Number of inner derivative workers, or None for automatic policy.

    Yields:

    """
    token = _inner_workers_var.set(None if n is None else int(n))
    try:
        yield
    finally:
        _inner_workers_var.reset(token)


def _int_env(name: str) -> int | None:
    """Reads a positive integer from an environment variable, or None if unset/invalid.

    Args:
        name: Environment variable name.

    Returns:
        Positive integer value, or None.
    """
    v = os.getenv(name)
    if not v:
        return None
    try:
        i = int(v)
        return i if i > 0 else None
    except ValueError:
        return None

def _detect_hw_threads() -> int:
    """Detects the number of hardware threads, capped by relevant environment variables.

    Returns:
        Number of hardware threads (at least 1).
    """
    hints = [
        _int_env("OMP_NUM_THREADS"),
        _int_env("MKL_NUM_THREADS"),
        _int_env("OPENBLAS_NUM_THREADS"),
        _int_env("VECLIB_MAXIMUM_THREADS"),
        _int_env("NUMEXPR_NUM_THREADS"),
    ]
    env_cap = min([h for h in hints if h is not None], default=None)
    hw = os.cpu_count() or 1
    return max(1, min(hw, env_cap) if env_cap else hw)


def resolve_inner_from_outer(w_params: int) -> int | None:
    """Resolves the number of inner derivative workers based on outer workers and defaults.

    Args:
        w_params: Number of outer derivative workers.

    Returns:
        Number of inner derivative workers, or None for automatic policy.
    """
    w = _inner_workers_var.get()
    if w is not None:
        return w
    if _DEFAULT_INNER_WORKERS is not None:
        return _DEFAULT_INNER_WORKERS
    cores = _detect_hw_threads()
    if w_params > 1:
        return min(4, max(1, cores // w_params))
    return min(4, cores)
