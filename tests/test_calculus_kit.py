"""Tests for CalculusKit class."""

import threading
import time

import numpy as np

from derivkit.calculus_kit import CalculusKit


def scalar_function(x):
    """Scalar-valued function for gradient/Hessian tests."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def vector_function(x):
    """Vector-valued function for Jacobian tests."""
    return np.asarray(x, dtype=float)


class RecordingGradient:
    """Records inputs to gradient builder."""
    def __init__(self):
        """Initialises recorder."""
        self.func = None
        self.x0 = None
        self.args = None
        self.kwargs = None

    def __call__(self, func, x0, *args, **kwargs):
        """Records inputs and returns a fixed dummy gradient."""
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.kwargs = kwargs
        return np.array([1.23, 4.56])


class RecordingJacobian:
    """Records inputs to Jacobian builder."""
    def __init__(self):
        """Initialises recorder."""
        self.func = None
        self.x0 = None
        self.args = None
        self.kwargs = None

    def __call__(self, func, x0, *args, **kwargs):
        """Records inputs and returns dummy Jacobian."""
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.kwargs = kwargs
        return np.full((2, 2), 7.0)


class RecordingHessian:
    """Records inputs to Hessian builder."""
    def __init__(self):
        """Initialises recorder."""
        self.func = None
        self.x0 = None
        self.args = None
        self.kwargs = None

    def __call__(self, func, x0, *args, **kwargs):
        """Records inputs and returns dummy Hessian."""
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.kwargs = kwargs
        return np.ones((3, 3))


class RecordingHessianDiag:
    """Records inputs to Hessian diagonal builder."""
    def __init__(self):
        """Initialises recorder."""
        self.func = None
        self.x0 = None
        self.args = None
        self.kwargs = None

    def __call__(self, func, x0, *args, **kwargs):
        """Records inputs and returns dummy Hessian diagonal."""
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.kwargs = kwargs
        return np.array([9.0, 8.0])


class RecordingHyperHessian:
    """Records inputs to hyper-Hessian builder."""
    def __init__(self):
        """Initialises recorder."""
        self.func = None
        self.x0 = None
        self.args = None
        self.kwargs = None

    def __call__(self, func, x0, *args, **kwargs):
        """Records inputs and returns dummy hyper-Hessian."""
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.kwargs = kwargs
        return np.ones((3, 3, 3))


def test_calculuskit_stores_function_and_x0():
    """Tests that CalculusKit stores the input function and x0 correctly."""
    x0 = [0.1, 0.2]
    ck = CalculusKit(scalar_function, x0)

    assert ck.function is scalar_function
    assert isinstance(ck.x0, np.ndarray)
    np.testing.assert_allclose(ck.x0, np.asarray(x0, dtype=float))


def test_gradient_delegates_to_build_gradient(monkeypatch):
    """Tests that gradient() delegates to build_gradient()."""
    recorder = RecordingGradient()
    monkeypatch.setattr(
        "derivkit.calculus_kit.build_gradient",
        recorder,
        raising=True,
    )

    x0 = [0.5, 1.5]
    ck = CalculusKit(scalar_function, x0)

    out = ck.gradient("extra", method="adaptive", n_workers=3)

    assert recorder.func is scalar_function
    np.testing.assert_allclose(recorder.x0, np.asarray(x0, dtype=float))
    assert recorder.args == ("extra",)
    assert recorder.kwargs == {"method": "adaptive", "n_workers": 3}

    np.testing.assert_allclose(out, np.array([1.23, 4.56]))


def test_jacobian_delegates_to_build_jacobian(monkeypatch):
    """Tests that jacobian() delegates to build_jacobian()."""
    recorder = RecordingJacobian()
    monkeypatch.setattr(
        "derivkit.calculus_kit.build_jacobian",
        recorder,
        raising=True,
    )

    x0 = np.array([1.0, 2.0])
    ck = CalculusKit(vector_function, x0)

    out = ck.jacobian(foo="bar")

    assert recorder.func is vector_function
    np.testing.assert_allclose(recorder.x0, x0)
    assert recorder.args == ()
    assert recorder.kwargs == {"foo": "bar"}
    np.testing.assert_allclose(out, np.full((2, 2), 7.0))


def test_hessian_delegates_to_build_hessian(monkeypatch):
    """Tests that hessian() delegates to build_hessian()."""
    recorder = RecordingHessian()
    monkeypatch.setattr(
        "derivkit.calculus_kit.build_hessian",
        recorder,
        raising=True,
    )

    x0 = [0.0, 1.0, 2.0]
    ck = CalculusKit(scalar_function, x0)

    out = ck.hessian(mode="full")

    assert recorder.func is scalar_function
    np.testing.assert_allclose(recorder.x0, np.asarray(x0, dtype=float))
    assert recorder.args == ()
    assert recorder.kwargs == {"mode": "full"}
    np.testing.assert_allclose(out, np.ones((3, 3)))


def test_hessian_diag_delegates_to_build_hessian_diag(monkeypatch):
    """Tests that hessian_diag() delegates to build_hessian_diag()."""
    recorder = RecordingHessianDiag()
    monkeypatch.setattr(
        "derivkit.calculus_kit.build_hessian_diag",
        recorder,
        raising=True,
    )

    x0 = [0.3, -0.7]
    ck = CalculusKit(scalar_function, x0)

    out = ck.hessian_diag(eps=1e-4)

    assert recorder.func is scalar_function
    np.testing.assert_allclose(recorder.x0, np.asarray(x0, dtype=float))
    assert recorder.args == ()
    assert recorder.kwargs == {"eps": 1e-4}
    np.testing.assert_allclose(out, np.array([9.0, 8.0]))


def test_hyper_hessian_delegates_to_build_hyper_hessian(monkeypatch):
    """Tests that hyper_hessian() delegates to build_hyper_hessian()."""
    recorder = RecordingHyperHessian()
    monkeypatch.setattr(
        "derivkit.calculus_kit.build_hyper_hessian",
        recorder,
        raising=True,
    )

    x0 = [0.0, 1.0, 2.0]
    ck = CalculusKit(scalar_function, x0)

    out = ck.hyper_hessian(ordering="ijk")

    assert recorder.func is scalar_function
    np.testing.assert_allclose(recorder.x0, np.asarray(x0, dtype=float))
    assert recorder.args == ()
    assert recorder.kwargs == {"ordering": "ijk"}
    np.testing.assert_allclose(out, np.ones((3, 3, 3)))


def _make_concurrency_probe(*, sleep_s: float = 0.03):
    """Returns (fn, get_max_active) to measure peak concurrent executions with a sleep delay."""
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fn(x):
        """Sleeps briefly and tracks concurrent entries before returning sum of squares."""
        x = np.asarray(x, dtype=float)

        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

        time.sleep(sleep_s)

        with state_lock:
            state["active"] -= 1

        return float(np.sum(x**2))

    def get_max_active():
        """Returns the maximum number of overlapping fn calls observed."""
        with state_lock:
            return int(state["max_active"])

    return fn, get_max_active


def _run_concurrent_calls(callable_, *, n_threads: int = 40):
    """Runs callable_ concurrently in threads and waits for completion."""
    start = threading.Barrier(n_threads + 1)

    def worker():
        start.wait()
        callable_([1.0, 2.0, 3.0])

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()

    start.wait()  # release all workers
    for t in threads:
        t.join()


def test_thread_safe_wraps_function_with_given_lock(monkeypatch):
    """Tests that thread_safe=True wraps function with given lock."""
    lock = threading.RLock()
    called = {}

    def fake_wrap_with_lock(func, lock=None):
        called["func"] = func
        called["lock"] = lock

        def wrapped(x):
            return func(x)

        return wrapped

    monkeypatch.setattr(
        "derivkit.calculus_kit.wrap_with_lock",
        fake_wrap_with_lock,
        raising=True,
    )

    ck = CalculusKit(lambda x: float(np.sum(np.asarray(x, float) ** 2)), [0.1], thread_safe=True, thread_lock=lock)

    assert called["func"] is not None
    assert called["lock"] is lock
    assert ck.function is not called["func"]  # wrapped


def test_thread_safe_serializes_direct_function_calls():
    """Tests that thread_safe=True serializes direct function calls."""
    fn, get_max_active = _make_concurrency_probe(sleep_s=0.03)
    ck = CalculusKit(fn, [0.0], thread_safe=True)

    _run_concurrent_calls(ck.function, n_threads=50)

    assert get_max_active() == 1


def test_thread_unsafe_allows_overlapping_function_calls():
    """Tests that thread_safe=False allows overlapping function calls."""
    fn, get_max_active = _make_concurrency_probe(sleep_s=0.03)
    ck = CalculusKit(fn, [0.0], thread_safe=False)

    _run_concurrent_calls(ck.function, n_threads=50)

    assert get_max_active() > 1
