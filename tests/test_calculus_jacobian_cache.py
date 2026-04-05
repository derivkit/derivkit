"""Unit tests for caching in ``derivkit.calculus.jacobian``."""

import numpy as np

from derivkit.calculus.jacobian import build_jacobian


def test_jacobian_cache_does_not_change_result():
    """Tests that Jacobian caching does not change the result."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return np.array([
            x * y + z,
            x**2 + np.cos(y),
            z**3 - x,
        ])

    theta0 = np.array([0.3, -0.5, 0.8])

    jac_no_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )

    jac_with_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(jac_with_cache, jac_no_cache, rtol=1e-10, atol=1e-12)


def test_jacobian_accepts_cache_init_kwargs():
    """Tests that Jacobian caching accepts cache initialization kwargs."""
    def model(theta):
        """Mock model function."""
        x, y = theta
        return np.array([x + y, x * y])

    theta0 = np.array([0.2, 0.4])

    jac = build_jacobian(
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

    assert jac.shape == (2, 2)
    assert np.isfinite(jac).all()


def test_jacobian_cache_reduces_model_evaluations_for_repeated_call():
    """Tests that Jacobian caching reduces model evaluations."""
    calls = {"count": 0}

    def model(theta):
        """Mock model function."""
        calls["count"] += 1
        x, y = theta
        return np.array([
            x**2 + y,
            np.sin(x) + x * y,
        ])

    theta0 = np.array([0.7, -0.2])

    jac_no_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )
    calls_no_cache = calls["count"]

    calls["count"] = 0

    jac_with_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_with_cache = calls["count"]

    np.testing.assert_allclose(jac_with_cache, jac_no_cache)
    assert calls_with_cache < calls_no_cache


def test_jacobian_cache_number_decimal_places_can_increase_hits():
    """Tests that cache rounding can increase Jacobian cache hits."""
    calls = {"count": 0}

    def model(theta):
        """Mock model function."""
        calls["count"] += 1
        x, y = theta
        return np.array([
            np.exp(x) + y,
            x - y**2,
        ])

    theta0 = np.array([0.31, -0.27])

    build_jacobian(
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

    build_jacobian(
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


def test_jacobian_cache_disable_matches_explicit_false_behavior():
    """Tests that default Jacobian cache behavior matches explicit caching."""
    calls = {"count": 0}

    def model(theta):
        """Mock model function."""
        calls["count"] += 1
        x, y = theta
        return np.array([
            x + y,
            x * y,
        ])

    theta0 = np.array([0.25, 0.75])

    jac_default = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
    )
    calls_default = calls["count"]

    calls["count"] = 0

    jac_explicit = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_explicit = calls["count"]

    np.testing.assert_allclose(jac_default, jac_explicit)
    assert calls_explicit == calls_default


def test_jacobian_cache_parallel_and_serial_agree():
    """Tests that cached Jacobians agree between serial and parallel execution."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return np.array([
            x + y + z,
            x * y + z,
            np.sin(x) + np.cos(y) - z,
        ])

    theta0 = np.array([0.4, -0.3, 0.2])

    jac_serial = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    jac_parallel = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=2,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(jac_parallel, jac_serial, rtol=1e-10, atol=1e-12)


def test_jacobian_cache_small_maxsize_preserves_correctness():
    """Tests that a small Jacobian cache preserves correctness."""
    def model(theta):
        """Mock model function."""
        x, y = theta
        return np.array([
            x**3 + y,
            np.cos(x * y),
        ])

    theta0 = np.array([0.6, -0.1])

    jac_small_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_maxsize": 2,
        },
    )

    jac_large_cache = build_jacobian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={
            "use_input_cache": True,
            "cache_maxsize": 4096,
        },
    )

    np.testing.assert_allclose(jac_small_cache, jac_large_cache, rtol=1e-10, atol=1e-12)
