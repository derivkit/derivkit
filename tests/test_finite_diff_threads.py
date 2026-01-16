"""Tests for finite difference derivative calculator with multithreading support."""

import math
import multiprocessing as mp

import numpy as np
import pytest

from derivkit.derivatives.finite.finite_difference import (
    FiniteDifferenceDerivative,
)


def f_log1p_x2(x: float) -> float:
    """Returns log(1 + x^2)."""
    return math.log1p(x * x)


def df_log1p_x2(x: float) -> float:
    """Returns derivative of log(1 + x^2): 2x / (1 + x^2)."""
    return (2.0 * x) / (1.0 + x * x)


def f_cubic(x: float) -> float:
    """Returns x^3."""
    return x * x * x


def d2f_cubic(x: float) -> float:
    """Returns second derivative of x^3: 6x."""
    return 6.0 * x


def f_vector(x: float) -> np.ndarray:
    """Returns a vector-valued function [x, x^2, sin(x)]."""
    return np.array([x, x * x, math.sin(x)], dtype=float)


def f_linear(x: float) -> float:
    """Returns a linear function f(x) = x."""
    return x


def d2f_sin(x: float) -> float:
    """Returns second derivative of sin(x): -sin(x)."""
    return -math.sin(x)


@pytest.mark.parametrize(
    "f,df,x0",
    [
        (math.sin, math.cos, -2.0),
        (math.exp, math.exp, -1.5),
        (f_log1p_x2, df_log1p_x2, -2.0),
    ],
)
@pytest.mark.parametrize("num_points", [5, 7, 9])
def test_fd_first_derivative_matches_truth(f, df, x0, num_points):
    """Tests that first derivatives computed via FD match analytic first derivatives."""
    fd = FiniteDifferenceDerivative(f, x0)
    deriv = fd.differentiate(order=1, stepsize=1e-3, num_points=num_points, n_workers=4)
    assert abs(deriv - df(x0)) < 1e-6


@pytest.mark.parametrize(
    "f,d2f,x0",
    [
        (math.sin, d2f_sin, 0.3),
        (math.exp, math.exp, -0.7),
        (f_cubic, d2f_cubic, 0.4),
    ],
)
@pytest.mark.parametrize("num_points", [5, 7, 9])
def test_fd_second_derivative_matches_truth(f, d2f, x0, num_points):
    """Tests that second derivatives computed via FD match analytic second derivatives."""
    fd = FiniteDifferenceDerivative(f, x0)
    d2 = fd.differentiate(order=2, stepsize=1e-3, num_points=num_points, n_workers=8)
    assert abs(d2 - d2f(x0)) < 1e-5


def test_fd_vector_output_componentwise():
    """Tests that FD can handle vector-valued functions."""
    x0 = 0.25
    fd = FiniteDifferenceDerivative(f_vector, x0)
    got = fd.differentiate(order=1, stepsize=1e-3, num_points=5, n_workers=8)
    truth = np.array([1.0, 2 * x0, math.cos(x0)], dtype=float)
    assert np.allclose(got, truth, rtol=0, atol=1e-6)


def _daemon_worker(q, x0):
    """Tests that FD works inside a daemon process."""
    try:
        fd = FiniteDifferenceDerivative(math.sin, x0)
        d = fd.differentiate(order=1, stepsize=1e-3, num_points=5, n_workers=8)
        q.put(("ok", d))
    except Exception as e:
        q.put(("err", repr(e)))


def test_fd_parallel_works_inside_daemon_process():
    """Tests that FD works correctly inside a daemon process."""
    q = mp.Queue()
    x0 = 0.2
    p = mp.Process(target=_daemon_worker, args=(q, x0))
    p.daemon = True  # emulate being inside another process pool / runner
    p.start()
    p.join(timeout=10)
    assert not p.is_alive(), "daemon subprocess hung"
    status, payload = q.get_nowait()
    assert status == "ok", f"FD crashed in daemon context: {payload}"
    assert abs(payload - math.cos(x0)) < 1e-5


def test_fd_ordering_stability_linear_function():
    """Tests that FD returns correct derivative on linear function (stability test)."""
    x0 = 0.3
    fd = FiniteDifferenceDerivative(f_linear, x0)
    deriv = fd.differentiate(order=1, stepsize=1e-3, num_points=9, n_workers=32)
    assert abs(deriv - 1.0) < 1e-12


def test_fd_rejects_bad_stepsize():
    """Tests that FD rejects non-positive stepsizes."""
    with pytest.raises(ValueError):
        fd = FiniteDifferenceDerivative(math.sin, 0.0)
        fd.differentiate(order=1, stepsize=0.0, num_points=5)


def test_fd_rejects_unsupported_combo():
    """Tests that FD rejects unsupported (stencil size, order) combinations."""
    with pytest.raises(ValueError):
        fd = FiniteDifferenceDerivative(math.sin, 0.0)
        fd.differentiate(order=3, stepsize=1e-3, num_points=3)
