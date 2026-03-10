"""Concurrency management for derivative computations."""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Sequence, Tuple

__all__ = [
    "parallel_execute",
    "normalize_workers",
]

_BACKENDS = ["dask", "processes"]

# Context-var and default
_inner_workers_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "derivkit_inner_workers", default=None
)


def parallel_execute(
    worker: Callable[..., Any],
    arg_tuples: Sequence[Tuple[Any, ...]],
    *,
    n_workers: int = 1,
    backend: str = "dask",
) -> list[Any]:
    """Applies a function to groups of arguments in parallel.

    Inner worker setting is applied to the context, so calls inside worker
    will see the resolved inner worker count.

    Args:
        worker: Function applied to each entry in ``arg_tuples`` (called as ``worker(*args)``).
        arg_tuples: Argument tuples; each tuple is expanded into one ``worker(*args)`` call.
        n_workers: Parallelism level for outer execution.
        backend: Parallel backend.
            - "dask" (default): build a Dask task graph and execute it with
              Dask's scheduler. Control the scheduler via
              ``dask.config.set(scheduler=...)`` or by instantiating a
              ``dask.distributed.Client`` (recommended for HPC clusters).
              Supports nested parallelism: tasks may themselves call
              ``dask.compute`` without deadlock.
            - "processes": use ``multiprocessing`` (spawn-based), running each
              worker in a separate Python process. Avoids GIL contention but
              does **not** support nested parallelism.

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

    if backend_l == "dask":
        return _parallel_execute_dask(worker, arg_tuples)

    # backend_l == "processes"
    from multiprocessing import Pool

    with Pool(processes=normalize_workers(n_workers)) as pool:
        return pool.starmap(worker, arg_tuples)


def _parallel_execute_dask(
    worker: Callable[..., Any],
    arg_tuples: Sequence[Tuple[Any, ...]],
) -> list[Any]:
    """Execute tasks in parallel using Dask delayed.

    Builds a Dask task graph and computes all results. The scheduler is chosen
    by Dask based on the active configuration:

    - **Default** (no client): threaded scheduler. Good for NumPy/BLAS
      workloads that release the GIL; avoids inter-process overhead.
    - **``dask.config.set(scheduler="processes")``**: process-based scheduler.
      Use for CPU-bound pure-Python work that does not release the GIL.
    - **``dask.distributed.Client``**: distributed scheduler. Recommended for
      HPC clusters. Nested ``dask.compute`` calls inside workers are fully
      supported.

    Args:
        worker: Callable applied to each set of arguments.
        arg_tuples: Sequence of argument tuples.

    Returns:
        List of results in the same order as ``arg_tuples``.
    """
    import dask

    tasks = [dask.delayed(worker)(*args) for args in arg_tuples]
    return tasks


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
