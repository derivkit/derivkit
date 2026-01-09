"""Unit tests for sampling utilities in local_polynomial_derivative.sampling."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivatives.local_polynomial_derivative.local_polynomial_derivative import (
    build_samples,
)


class DummyConfig:
    """Minimal config for build_samples; only rel_steps is used."""

    def __init__(self, rel_steps):
        """Initializes DummyConfig with rel_steps."""
        self.rel_steps = rel_steps


def f_square(x: float) -> float:
    """Scalar function: x^2."""
    return x**2


def f_identity(x: float) -> float:
    """Scalar function: identity."""
    return x


def f_vector_sin_cos(x: float) -> np.ndarray:
    """Vector-valued function: [sin(x), cos(x)]."""
    return np.array([np.sin(x), np.cos(x)], dtype=float)


def test_build_samples_symmetric_around_zero_scalar_output():
    """Tests that for x0=0, sampling should be symmetric ±rel_steps and sorted."""
    cfg = DummyConfig(rel_steps=[0.1, 0.2])
    x0 = 0.0

    xs, ys = build_samples(f_square, x0, cfg, n_workers=1)

    # Expect symmetric sample points
    expected_xs = np.array([-0.2, -0.1, 0.1, 0.2])
    assert xs.shape == expected_xs.shape
    assert np.allclose(xs, expected_xs)

    # Scalar output should give (n_samples, 1) array
    assert ys.shape == (xs.size, 1)
    assert np.allclose(ys[:, 0], xs**2)


def test_build_samples_scaled_around_nonzero_x0():
    """Tests that for x0!=0, sampling should be multiplicative x0*(1±rel_steps)."""
    cfg = DummyConfig(rel_steps=[0.1, 0.2])
    x0 = 2.0

    xs, ys = build_samples(f_square, x0, cfg, n_workers=1)

    # raw steps: x0*(1 - 0.2), x0*(1 - 0.1), x0*(1 + 0.1), x0*(1 + 0.2)
    expected_xs = np.array([
        x0 * (1.0 - 0.2),
        x0 * (1.0 - 0.1),
        x0 * (1.0 + 0.1),
        x0 * (1.0 + 0.2),
    ])
    expected_xs.sort()

    assert xs.shape == expected_xs.shape
    assert np.allclose(xs, expected_xs)

    # Check function evaluations
    assert ys.shape == (xs.size, 1)
    assert np.allclose(ys[:, 0], xs**2)


def test_build_samples_vector_output_and_parallel_matches_serial():
    """Tests that vector outputs and parallel evaluation should give same result as serial."""
    cfg = DummyConfig(rel_steps=[0.05, 0.1, 0.2])
    x0 = 1.5

    xs1, ys1 = build_samples(f_vector_sin_cos, x0, cfg, n_workers=1)
    xs2, ys2 = build_samples(f_vector_sin_cos, x0, cfg, n_workers=4)

    # xs should be identical and sorted
    assert np.allclose(xs1, xs2)
    assert np.all(np.diff(xs1) >= 0)

    # Same shapes and values
    assert ys1.shape == ys2.shape
    assert ys1.shape[0] == xs1.size
    assert ys1.shape[1] == 2  # two components
    assert np.allclose(ys1, ys2)


def test_build_samples_deduplicates_and_sorts_steps():
    """Tests that duplicate/zero rel_steps should lead to unique, sorted sample points."""
    # Includes duplicates and a zero step
    cfg = DummyConfig(rel_steps=[0.0, 0.1, 0.1, 0.2])
    x0 = 1.0

    xs, ys = build_samples(f_identity, x0, cfg, n_workers=1)

    raw_steps = np.array([0.0, 0.1, 0.1, 0.2], dtype=float)
    expected_xs = np.unique(
        x0 * (1.0 + np.concatenate([-raw_steps, raw_steps]))
    )
    expected_xs.sort()

    assert np.allclose(xs, expected_xs)
    assert np.all(np.diff(xs) >= 0)  # sorted

    # y=x, so ys should just repeat xs in a (n_samples,1) layout
    assert ys.shape == (xs.size, 1)
    assert np.allclose(ys[:, 0], xs)


@pytest.mark.parametrize(
    "rel_steps",
    [
        [],                  # empty list
        np.array([[]]),      # 2D array
    ],
)
def test_build_samples_invalid_rel_steps_raises(rel_steps):
    """Tests that non-1D or empty rel_steps should raise a ValueError."""
    cfg = DummyConfig(rel_steps=rel_steps)

    with pytest.raises(ValueError):
        build_samples(f_identity, x0=0.0, config=cfg, n_workers=1)
