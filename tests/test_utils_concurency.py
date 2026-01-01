"""Tests for derivkit.utils.concurrency."""

from __future__ import annotations

import contextvars

import pytest

from derivkit.utils import concurrency as conc
from derivkit.utils.concurrency import normalize_workers, resolve_workers


def _reset_inner_var() -> None:
    """Reset the contextvar to its default before each test that needs it."""
    # ContextVar has a default=None, so we just set it explicitly
    conc._inner_workers_var.set(None)


def test_set_default_inner_derivative_workers_overrides_auto(monkeypatch):
    """set_default_inner_derivative_workers should override automatic policy."""
    _reset_inner_var()
    # Make sure we don't inherit a previous default
    monkeypatch.setattr(conc, "_DEFAULT_INNER_WORKERS", None, raising=False)

    conc.set_default_inner_derivative_workers(3)
    # Internal state updated
    assert conc._DEFAULT_INNER_WORKERS == 3

    # resolve_inner_from_outer should now use the default, regardless of outer workers
    assert conc.resolve_inner_from_outer(w_params=1) == 3
    assert conc.resolve_inner_from_outer(w_params=4) == 3


def test_set_inner_derivative_workers_context_manager_restores_previous():
    """Context manager should temporarily set the value and restore it on exit."""
    _reset_inner_var()

    var: contextvars.ContextVar[int | None] = conc._inner_workers_var
    prev = var.get()  # usually None, but don't assume

    with conc.set_inner_derivative_workers(5) as returned_prev:
        # The context manager yields the previous value
        assert returned_prev == prev
        # Inside the context, the value is updated
        assert var.get() == 5

    # After exiting, previous value is restored
    assert var.get() == prev


def test_resolve_inner_from_outer_uses_auto_policy(monkeypatch):
    """resolve_inner_from_outer should use hardware-based heuristic when no overrides."""
    _reset_inner_var()
    # Ensure no default override
    monkeypatch.setattr(conc, "_DEFAULT_INNER_WORKERS", None, raising=False)
    # Make hardware detection deterministic
    monkeypatch.setattr(conc, "_detect_hw_threads", lambda: 8)

    # For multiple outer workers, split cores and cap at 4
    assert conc.resolve_inner_from_outer(w_params=2) == 4  # min(4, 8 // 2) = 4
    # For single outer worker, just cap by 4
    assert conc.resolve_inner_from_outer(w_params=1) == 4  # min(4, 8) = 4


def test_resolve_inner_from_outer_prefers_context_then_default(monkeypatch):
    """Precedence: contextvar > default > auto heuristic."""
    _reset_inner_var()
    monkeypatch.setattr(conc, "_detect_hw_threads", lambda: 16)
    monkeypatch.setattr(conc, "_DEFAULT_INNER_WORKERS", None, raising=False)

    # 1) Context override should win
    with conc.set_inner_derivative_workers(7):
        assert conc.resolve_inner_from_outer(w_params=3) == 7

    # 2) With no context override, default should be used
    conc.set_default_inner_derivative_workers(5)
    assert conc._DEFAULT_INNER_WORKERS == 5
    assert conc.resolve_inner_from_outer(w_params=3) == 5


def test_parallel_execute_sequential_propagates_inner_workers():
    """parallel_execute should set the inner worker context for sequential execution."""
    _reset_inner_var()

    def worker(x):
        # Worker sees the inner worker setting through the contextvar
        return x, conc._inner_workers_var.get()

    results = conc.parallel_execute(
        worker,
        arg_tuples=[(1,), (2,)],
        outer_workers=1,
        inner_workers=3,
    )

    assert results == [(1, 3), (2, 3)]
    # After the call, the context should be reset to its previous value (here: None)
    assert conc._inner_workers_var.get() is None


def test_parallel_execute_threaded_propagates_inner_workers():
    """parallel_execute should set inner worker context inside thread workers."""
    _reset_inner_var()

    def worker(x):
        # Called inside ThreadPoolExecutor worker threads
        return x, conc._inner_workers_var.get()

    results = conc.parallel_execute(
        worker,
        arg_tuples=[(10,), (20,)],
        outer_workers=2,
        inner_workers=4,
    )

    # Order is preserved because we collect futures in submission order
    assert results == [(10, 4), (20, 4)]
    assert conc._inner_workers_var.get() is None


def test_normalize_workers_various_inputs():
    """Tests that normalize_workers handles various inputs correctly."""
    assert normalize_workers(1) == 1
    assert normalize_workers(4) == 4
    assert normalize_workers(0) == 1
    assert normalize_workers(-3) == 1
    assert normalize_workers(None) == 1
    assert normalize_workers(2.7) == 2


def test_resolve_workers_defaults_to_serial(monkeypatch):
    """Tests that default outer_workers=1 and inner policy is used."""
    monkeypatch.setattr(
        "derivkit.utils.concurrency.resolve_inner_from_outer",
        lambda outer: 42,
    )

    dk_kwargs = {"foo": "bar"}
    outer, inner, cleaned = resolve_workers(None, dk_kwargs)

    assert outer == 1
    assert inner == 42
    assert cleaned == {"foo": "bar"}
    assert dk_kwargs == {"foo": "bar"}


@pytest.mark.parametrize("n_workers", [0, -3, "nope", 1.2])
def test_resolve_workers_normalizes_outer_workers(monkeypatch, n_workers):
    """Tests that invalid outer_workers are normalized to 1."""
    monkeypatch.setattr(
        "derivkit.utils.concurrency.resolve_inner_from_outer",
        lambda outer: 7,
    )

    outer, inner, cleaned = resolve_workers(n_workers, {"x": 1})
    assert outer >= 1
    assert inner == 7
    assert cleaned == {"x": 1}


def test_resolve_workers_inner_override_is_respected(monkeypatch):
    """Tests that valid inner_workers override is respected."""
    monkeypatch.setattr(
        "derivkit.utils.concurrency.resolve_inner_from_outer",
        lambda outer: 999,
    )

    dk_kwargs = {"inner_workers": 3, "alpha": 1.0}
    outer, inner, cleaned = resolve_workers(4, dk_kwargs)

    assert outer == 4
    assert inner == 3
    assert cleaned == {"alpha": 1.0}

    # New contract: resolve_workers does not mutate caller kwargs.
    assert dk_kwargs == {"inner_workers": 3, "alpha": 1.0}


@pytest.mark.parametrize("inner_override", [0, -2, "bad"])
def test_resolve_workers_invalid_inner_override_falls_back_to_one(monkeypatch, inner_override):
    """Tests that invalid inner_workers override falls back to single worker."""
    monkeypatch.setattr(
        "derivkit.utils.concurrency.resolve_inner_from_outer",
        lambda outer: 5,
    )

    dk_kwargs = {"beta": 2.0}
    if inner_override is not None:
        dk_kwargs["inner_workers"] = inner_override

    outer, inner, cleaned = resolve_workers(2, dk_kwargs)

    assert outer == 2
    if inner_override is None:
        assert inner == 5
    else:
        assert inner == 1
    assert cleaned == {"beta": 2.0}
