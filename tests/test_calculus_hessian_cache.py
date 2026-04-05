"""Unit tests for caching inside ``derivkit.calculus.hessian``."""

import numpy as np

from derivkit.calculus.hessian import build_hessian, build_hessian_diag


def test_hessian_cache_does_not_change_result():
    """Tests that Hessian caching does not change the result."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return x**2 * y + np.sin(z) + x * z

    theta0 = np.array([0.3, -0.5, 0.8])

    hess_no_cache = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )

    hess_with_cache = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        hess_with_cache, hess_no_cache, rtol=1e-10, atol=1e-12
    )


def test_hessian_cache_reduces_model_evaluations():
    """Tests that Hessian caching reduces model evaluations."""
    calls = {"count": 0}

    def model(theta):
        calls["count"] += 1
        x, y, z = theta
        return x**2 * y + np.sin(z) + x * z

    theta0 = np.array([0.3, -0.5, 0.8])

    hess_no_cache = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )
    calls_no_cache = calls["count"]

    calls["count"] = 0

    hess_with_cache = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_with_cache = calls["count"]

    np.testing.assert_allclose(hess_with_cache, hess_no_cache)
    assert calls_with_cache < calls_no_cache


def test_hessian_cache_accepts_cache_init_kwargs():
    """Tests that Hessian caching accepts cache initialization kwargs."""
    def model(theta):
        """Mock model function."""
        x, y = theta
        return x**3 + x * y + np.cos(y)

    theta0 = np.array([0.2, 0.4])

    hess = build_hessian(
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

    assert hess.shape == (2, 2)
    assert np.isfinite(hess).all()


def test_hessian_cache_parallel_and_serial_agree():
    """Tests that cached Hessians agree between serial and parallel execution."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return x * y + y * z + np.sin(x)

    theta0 = np.array([0.4, -0.3, 0.2])

    hess_serial = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    hess_parallel = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=2,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        hess_parallel, hess_serial, rtol=1e-10, atol=1e-12
    )


def test_hessian_cache_small_maxsize_preserves_correctness():
    """Tests that a small Hessian cache preserves correctness."""
    def model(theta):
        """Mock model function."""
        x, y = theta
        return x**4 + x * y + np.exp(y)

    theta0 = np.array([0.6, -0.1])

    hess_small_cache = build_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_maxsize": 2,
        },
    )

    hess_large_cache = build_hessian(
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
        hess_small_cache, hess_large_cache, rtol=1e-10, atol=1e-12
    )


def test_hessian_diag_cache_does_not_change_result():
    """Tests that diagonal Hessian caching does not change the result."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return x**2 * y + np.sin(z) + x * z

    theta0 = np.array([0.3, -0.5, 0.8])

    diag_no_cache = build_hessian_diag(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )

    diag_with_cache = build_hessian_diag(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        diag_with_cache, diag_no_cache, rtol=1e-10, atol=1e-12
    )
