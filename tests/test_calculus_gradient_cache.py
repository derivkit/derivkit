"""Unit tests for caching in ``derivkit.calculus.gradient``."""

import numpy as np

from derivkit.calculus.gradient import build_gradient


def test_gradient_cache_does_not_change_result():
    """Tests that gradient caching does not change the result."""
    def model(theta):
        """Mock scalar-valued model function."""
        x, y, z = theta
        return x * y + z**2 + np.sin(x)

    theta0 = np.array([0.3, -0.5, 0.8])

    grad_no_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )

    grad_with_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        grad_with_cache,
        grad_no_cache,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gradient_accepts_cache_init_kwargs():
    """Tests that gradient caching accepts cache initialization kwargs."""
    def model(theta):
        """Mock scalar-valued model function."""
        x, y = theta
        return x**2 + x * y + np.cos(y)

    theta0 = np.array([0.2, 0.4])

    grad = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_number_decimal_places": 12,
            "cache_maxsize": 128,
            "cache_copy": True,
        },
    )

    assert grad.shape == (2,)
    assert np.isfinite(grad).all()


def test_gradient_cache_reduces_model_evaluations_for_repeated_call():
    """Tests that gradient caching reduces model evaluations."""
    calls = {"count": 0}

    def model(theta):
        """Mock scalar-valued model function."""
        calls["count"] += 1
        x, y = theta
        return x**2 + np.sin(x) * y + y**2

    theta0 = np.array([0.7, -0.2])

    grad_no_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )
    calls_no_cache = calls["count"]

    calls["count"] = 0

    grad_with_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_with_cache = calls["count"]

    np.testing.assert_allclose(grad_with_cache, grad_no_cache)
    assert calls_with_cache < calls_no_cache


def test_gradient_cache_number_decimal_places_can_increase_hits():
    """Tests that cache rounding can increase gradient cache hits."""
    calls = {"count": 0}

    def model(theta):
        """Mock scalar-valued model function."""
        calls["count"] += 1
        x, y = theta
        return np.exp(x) + x * y - y**2

    theta0 = np.array([0.31, -0.27])

    build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_number_decimal_places": None,
        },
    )
    calls_exact = calls["count"]

    calls["count"] = 0

    build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_number_decimal_places": 12,
        },
    )
    calls_rounded = calls["count"]

    assert calls_rounded <= calls_exact


def test_gradient_cache_disable_matches_explicit_true_behavior():
    """Tests that default gradient cache behavior matches explicit caching."""
    calls = {"count": 0}

    def model(theta):
        """Mock scalar-valued model function."""
        calls["count"] += 1
        x, y = theta
        return x + y + x * y

    theta0 = np.array([0.25, 0.75])

    grad_default = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
    )
    calls_default = calls["count"]

    calls["count"] = 0

    grad_explicit = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_explicit = calls["count"]

    np.testing.assert_allclose(grad_default, grad_explicit)
    assert calls_explicit == calls_default


def test_gradient_cache_parallel_and_serial_agree():
    """Tests that cached gradients agree between serial and parallel execution."""
    def model(theta):
        """Mock scalar-valued model function."""
        x, y, z = theta
        return x + y + z + x * y - np.cos(z)

    theta0 = np.array([0.4, -0.3, 0.2])

    grad_serial = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    grad_parallel = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=2,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        grad_parallel,
        grad_serial,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gradient_cache_small_maxsize_preserves_correctness():
    """Tests that a small gradient cache preserves correctness."""
    def model(theta):
        """Mock scalar-valued model function."""
        x, y = theta
        return x**3 + y + np.cos(x * y)

    theta0 = np.array([0.6, -0.1])

    grad_small_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_maxsize": 2,
        },
    )

    grad_large_cache = build_gradient(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_maxsize": 4096,
        },
    )

    np.testing.assert_allclose(
        grad_small_cache,
        grad_large_cache,
        rtol=1e-10,
        atol=1e-12,
    )
