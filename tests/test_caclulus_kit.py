"""Tests for CalculusKit class."""

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


def test_calculuskit_stores_function_and_x0():
    """Tests that CalculusKit stores the input function and x0 correctly."""
    x0 = [0.1, 0.2]
    ck = CalculusKit(scalar_function, x0)

    assert ck.function is scalar_function
    assert isinstance(ck.x0, np.ndarray)
    assert ck.x0.dtype == float
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

    # Delegation checks
    assert recorder.func is scalar_function
    np.testing.assert_allclose(recorder.x0, np.asarray(x0, dtype=float))
    assert recorder.args == ("extra",)
    assert recorder.kwargs == {"method": "adaptive", "n_workers": 3}

    # Return passthrough
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
