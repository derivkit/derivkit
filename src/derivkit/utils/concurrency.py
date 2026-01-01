"""Concurrency management for derivative computations."""

from __future__ import annotations

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence, Tuple

__all__ = [
    "set_default_inner_derivative_workers",
    "set_inner_derivative_workers",
    "resolve_inner_from_outer",
    "parallel_execute",
    "_inner_workers_var",
    "normalize_workers",
    "resolve_workers",
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
def set_inner_derivative_workers(n: int | None) -> Iterator[int | None]:
    """Temporarily sets the number of inner derivative workers.

    Args:
        n: Number of inner derivative workers, or ``None`` for automatic policy.

    Yields:
        int | None: The previous worker setting (restored on exit).
    """
    prev = _inner_workers_var.get()
    token = _inner_workers_var.set(None if n is None else int(n))
    try:
        yield prev
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


def parallel_execute(
    worker: Callable[..., Any],
    arg_tuples: Sequence[Tuple[Any, ...]],
    *,
    outer_workers: int = 1,
    inner_workers: int | None = None,
) -> list[Any]:
    """Runs ``worker(*args)`` for each tuple in arg_tuples with outer threads.

    Inner worker setting is applied to the context, so calls inside worker
    will see the resolved inner worker count.
    """
    with set_inner_derivative_workers(inner_workers):
        if outer_workers > 1:
            with ThreadPoolExecutor(max_workers=outer_workers) as ex:
                futures = []
                for args in arg_tuples:
                    # Each task gets its own copy of the current context
                    ctx = contextvars.copy_context()
                    futures.append(ex.submit(ctx.run, worker, *args))
                return [f.result() for f in futures]
        else:
            return [worker(*args) for args in arg_tuples]


def normalize_workers(
    n_workers: Any
) -> int:
    """Ensures n_workers is a positive integer, defaulting to 1.

    Args:
        n_workers: Input number of workers (can be None, float, negative, etc.)

    Returns:
        int: A positive integer number of workers (at least 1).

    Raises:
        None: Invalid inputs are coerced to 1.
    """
    try:
        n = int(n_workers)
    except (TypeError, ValueError):
        n = 1
    return 1 if n < 1 else n


def resolve_workers(
    n_workers: Any,
    dk_kwargs: dict[str, Any],
) -> tuple[int, int | None, dict[str, Any]]:
    """Decides how parallel work is split between outer calculus routines and the inner derivative engine.

    Outer workers parallelize across independent derivative tasks (e.g. parameters,
    output components, Hessian entries). Inner workers control parallelism inside
    each derivative evaluation (within DerivativeKit).

    If both levels spawn workers simultaneously, nested parallelism can cause
    oversubscription. By default, the inner worker count is derived from the
    outer worker count to avoid that. You can override this by passing
    ``inner_workers=<int>`` via ``dk_kwargs``.

    Args:
        n_workers: Number of outer workers. If ``None``, defaults to 1.
        dk_kwargs: Keyword arguments forwarded to DerivativeKit.differentiate.
            May include ``inner_workers`` to override the default policy.

    Returns:
        (outer_workers, inner_workers, dk_kwargs_cleaned), where ``dk_kwargs_cleaned``
        has any ``inner_workers`` entry removed.
    """
    dk_kwargs_cleaned = dict(dk_kwargs)
    inner_override = dk_kwargs_cleaned.pop("inner_workers", None)

    outer = normalize_workers(n_workers)
    if inner_override is None:
        inner = resolve_inner_from_outer(outer)
    else:
        inner = normalize_workers(inner_override)

    return outer, inner, dk_kwargs_cleaned
