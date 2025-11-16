"""Unit tests for finite difference extrapolation utilities."""

from __future__ import annotations

import numpy as np

import derivkit.finite.extrapolators as fe
from derivkit.finite.extrapolators import (
    adaptive_richardson_fd,
    adaptive_ridders_fd,
    fixed_richardson_fd,
    fixed_ridders_fd,
)


def single_finite_constant(order: int, h: float, num_points: int, n_workers: int):
    """Finite difference stub that always returns a scalar constant."""
    return 2.5  # arbitrary but fixed


def single_finite_constant_vector(order: int, h: float, num_points: int, n_workers: int):
    """Finite difference stub that always returns a vector constant."""
    return np.array([1.0, -3.0, 4.5], dtype=float)


RICHARDSON_VALUES: list[float] = []
RICHARDSON_INDEX: int = 0

RIDDERS_VALUES: list[tuple[float, float]] = []
RIDDERS_INDEX: int = 0


def richardson_stub(base_values, p: int, r: float):
    """Stub for richardson_extrapolate using a pre-defined value sequence."""
    global RICHARDSON_INDEX
    value = RICHARDSON_VALUES[RICHARDSON_INDEX]
    RICHARDSON_INDEX += 1
    return value


def ridders_stub(base_values, p: int, r: float):
    """Stub for ridders_extrapolate using a pre-defined (value, err) sequence."""
    global RIDDERS_INDEX
    value, err = RIDDERS_VALUES[RIDDERS_INDEX]
    RIDDERS_INDEX += 1
    return value, err


def test_fixed_richardson_fd_scalar_no_error():
    """Tests that fixed_richardson_fd should return a stable value for constant estimates."""
    est = fixed_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        levels=3,
        p=2,
        r=2.0,
        return_error=False,
    )
    assert np.isclose(est, 2.5)


def test_fixed_richardson_fd_scalar_with_error():
    """Tests that fixed_richardson_fd returns a value and zero error when levels == 2."""
    est, err = fixed_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        levels=2,   # must be >= 2 now
        p=2,
        r=2.0,
        return_error=True,
    )
    assert np.isclose(est, 2.5)
    err_arr = np.asarray(err, dtype=float)
    assert err_arr.shape == np.asarray(est).shape
    assert np.all(err_arr == 0.0)


def test_fixed_richardson_fd_vector_output():
    """Tests that fixed_richardson_fd should handle vector-valued finite-difference estimates."""
    est = fixed_richardson_fd(
        single_finite_constant_vector,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        levels=3,
        p=2,
        r=2.0,
        return_error=False,
    )
    est_arr = np.asarray(est, dtype=float)
    assert est_arr.shape == (3,)
    assert np.allclose(est_arr, np.array([1.0, -3.0, 4.5]))


def test_fixed_ridders_fd_scalar_no_error(monkeypatch):
    """Tests that fixed_ridders_fd should return a stable value for constant estimates."""
    est = fixed_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        levels=3,
        p=2,
        r=2.0,
        return_error=False,
    )
    # For a constant sequence, Ridders should also return that constant.
    assert np.isclose(est, 2.5)


def test_fixed_ridders_fd_scalar_with_error(monkeypatch):
    """Tests that with a stubbed ridders_extrapolate, fixed_ridders_fd should forward value+err."""
    global RIDDERS_VALUES, RIDDERS_INDEX
    RIDDERS_VALUES = [(5.0, 0.1)]
    RIDDERS_INDEX = 0

    monkeypatch.setattr(fe, "ridders_extrapolate", ridders_stub)

    est, err = fixed_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        levels=2,
        p=2,
        r=2.0,
        return_error=True,
    )

    assert np.isclose(est, 5.0)
    assert np.isclose(err, 0.1)


def test_adaptive_richardson_fd_converges_to_latest_estimate(monkeypatch):
    """Tests that adaptive_richardson_fd should accept the latest estimate when converged."""
    global RICHARDSON_VALUES, RICHARDSON_INDEX
    # Sequence of extrapolated values: first sets best_est, second triggers convergence.
    RICHARDSON_VALUES = [1.0, 1.0]
    RICHARDSON_INDEX = 0

    monkeypatch.setattr(fe, "richardson_extrapolate", richardson_stub)

    est = adaptive_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=False,
    )

    assert np.isclose(est, 1.0)


def test_adaptive_richardson_fd_falls_back_on_divergence(monkeypatch):
    """Tests that if extrapolated values diverge badly, adaptive_richardson_fd should fall back."""
    global RICHARDSON_VALUES, RICHARDSON_INDEX
    # First value becomes best_est; second is far away -> triggers divergence branch.
    RICHARDSON_VALUES = [1.0, 100.0]
    RICHARDSON_INDEX = 0

    monkeypatch.setattr(fe, "richardson_extrapolate", richardson_stub)

    est = adaptive_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=False,
    )

    # On divergence, function returns previous best_est (1.0)
    assert np.isclose(est, 1.0)


def test_adaptive_richardson_fd_max_levels_without_convergence():
    """Tests that if max_levels < min_levels, we should fall back to last base value."""
    est = adaptive_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=1,
        min_levels=2,
        r=2.0,
        rtol=1e-8,
        atol=0.0,
        return_error=False,
    )

    # In this regime, best_est is taken as base_values[-1] = single_finite_constant(...)
    assert np.isclose(est, 2.5)


def test_adaptive_richardson_fd_return_error_on_convergence(monkeypatch):
    """Tests that on convergence with return_error=True, error should be non-negative and finite."""
    global RICHARDSON_VALUES, RICHARDSON_INDEX
    RICHARDSON_VALUES = [3.0, 3.0]
    RICHARDSON_INDEX = 0

    monkeypatch.setattr(fe, "richardson_extrapolate", richardson_stub)

    est, err = adaptive_richardson_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=True,
    )

    est_arr = np.asarray(est, dtype=float)
    err_arr = np.asarray(err, dtype=float)

    assert np.allclose(est_arr, 3.0)
    # In the convergence branch, error is either zeros or last_err, both non-negative.
    assert np.all(err_arr >= 0.0)
    assert np.all(np.isfinite(err_arr))


def test_adaptive_ridders_fd_converges_to_latest_estimate(monkeypatch):
    """Tests that adaptive_ridders_fd should accept the latest estimate when converged."""
    global RIDDERS_VALUES, RIDDERS_INDEX
    # Two extrapolated values; second triggers convergence (same as first).
    RIDDERS_VALUES = [(1.0, 0.1), (1.0, 0.05)]
    RIDDERS_INDEX = 0

    monkeypatch.setattr(fe, "ridders_extrapolate", ridders_stub)

    est = adaptive_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=False,
    )

    assert np.isclose(est, 1.0)


def test_adaptive_ridders_fd_falls_back_on_divergence(monkeypatch):
    """Tests that if Ridders estimates diverge badly, adaptive_ridders_fd falls back to previous best."""
    global RIDDERS_VALUES, RIDDERS_INDEX
    # First value becomes best_est; second is far away, forcing divergence.
    RIDDERS_VALUES = [(1.0, 0.1), (50.0, 2.0)]
    RIDDERS_INDEX = 0

    monkeypatch.setattr(fe, "ridders_extrapolate", ridders_stub)

    est = adaptive_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=False,
    )

    # On divergence, function returns previous best_est (1.0)
    assert np.isclose(est, 1.0)


def test_adaptive_ridders_fd_max_levels_without_convergence():
    """Tests that if max_levels < min_levels, we fall back to last base value."""
    est = adaptive_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=1,
        min_levels=2,
        r=2.0,
        rtol=1e-8,
        atol=0.0,
        return_error=False,
    )

    assert np.isclose(est, 2.5)


def test_adaptive_ridders_fd_return_error_on_convergence(monkeypatch):
    """Tests that on convergence with return_error=True, error from ridders_extrapolate is propagated."""
    global RIDDERS_VALUES, RIDDERS_INDEX
    # First call sets best_est; second converges with smaller error.
    RIDDERS_VALUES = [(4.0, 0.2), (4.0, 0.05)]
    RIDDERS_INDEX = 0

    monkeypatch.setattr(fe, "ridders_extrapolate", ridders_stub)

    est, err = adaptive_ridders_fd(
        single_finite_constant,
        order=1,
        stepsize=0.1,
        num_points=5,
        n_workers=1,
        p=2,
        max_levels=3,
        min_levels=2,
        r=2.0,
        rtol=1e-3,
        atol=0.0,
        return_error=True,
    )

    assert np.isclose(est, 4.0)
    assert np.isclose(err, 0.05)
