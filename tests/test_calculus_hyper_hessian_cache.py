"""Unit tests for caching in ``derivkit.calculus_hyper_hessian``."""

import numpy as np

from derivkit.calculus.hyper_hessian import build_hyper_hessian


def test_hyper_hessian_cache_does_not_change_result():
    """Tests that hyper-Hessian caching does not change the result."""
    def model(theta):
        """Mock model function."""
        x, y, z = theta
        return x**2 * y + x * z + np.sin(z)

    theta0 = np.array([0.3, -0.5, 0.8])

    hh_no_cache = build_hyper_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )

    hh_with_cache = build_hyper_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )

    np.testing.assert_allclose(
        hh_with_cache, hh_no_cache, rtol=1e-10, atol=1e-12
    )


def test_hyper_hessian_cache_reduces_model_evaluations():
    """Tests that hyper-Hessian caching reduces model evaluations."""
    calls = {"count": 0}

    def model(theta):
        """Mock model function."""
        calls["count"] += 1
        x, y, z = theta
        return x**2 * y + x * z + np.sin(z)

    theta0 = np.array([0.3, -0.5, 0.8])

    hh_no_cache = build_hyper_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": False},
    )
    calls_no_cache = calls["count"]

    calls["count"] = 0

    hh_with_cache = build_hyper_hessian(
        model,
        theta0,
        method="finite",
        n_workers=1,
        dk_init_kwargs={"use_input_cache": True},
    )
    calls_with_cache = calls["count"]

    np.testing.assert_allclose(hh_with_cache, hh_no_cache)
    assert calls_with_cache < calls_no_cache


def test_hyper_hessian_accepts_cache_init_kwargs():
    """Tests that hyper-Hessian caching accepts cache initialization kwargs."""
    def model(theta):
        """Mock model function."""
        x, y = theta
        return x**3 + x * y + np.cos(y)

    theta0 = np.array([0.2, 0.4])

    hh = build_hyper_hessian(
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

    assert hh.shape == (2, 2, 2)
    assert np.isfinite(hh).all()
