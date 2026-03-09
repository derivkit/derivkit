"""Concurrency management for derivative computations."""

from __future__ import annotations

import contextvars
from multiprocessing import Pool
from typing import Any, Callable, Sequence, Tuple

__all__ = [
    "parallel_execute",
    "normalize_workers",
]

_BACKENDS = ["processes"]

# Context-var and default
_inner_workers_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "derivkit_inner_workers", default=None
)


def parallel_execute(
    worker: Callable[..., Any],
    arg_tuples: Sequence[Tuple[Any, ...]],
    *,
    n_workers: int = 1,
    backend: str = "processes",  # TODO: implement  MPI or Dask?
) -> list[Any]:
    """Applies a function to groups of arguments in parallel.

    Inner worker setting is applied to the context, so calls inside worker
    will see the resolved inner worker count.

    Args:
        worker: Function applied to each entry in ``arg_tuples`` (called as ``worker(*args)``).
        arg_tuples: Argument tuples; each tuple is expanded into one ``worker(*args)`` call.
        n_workers: Parallelism level for outer execution.
        backend: Parallel backend.
            - "processes": use multiprocessing (spawn-based), running each
              worker in a separate Python process. This avoids GIL contention
              and is safer for native/OpenMP/BLAS stacks.


    Returns:
        List of worker return values.
    """
    backend_l = str(backend).lower()
    if backend_l not in _BACKENDS:
        raise NotImplementedError(
            f"parallel_execute backend={backend!r} not supported yet."
            f" Use one of: {', '.join(_BACKENDS)}."
        )

    if n_workers <= 1:  # No need of using a parallel backend
        return [worker(*args) for args in arg_tuples]

    elif backend_l == "processes":
        with Pool(processes=normalize_workers(n_workers)) as pool:
            return pool.starmap(worker, arg_tuples)


def normalize_workers(n_workers: Any) -> int:
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
