"""Unit tests for LocalPolyConfig."""

from __future__ import annotations

import numpy as np

from derivkit.local_polynomial_derivative.local_poly_config import (
    LocalPolyConfig,
)


def test_local_poly_config_defaults():
    """Tests that default constructor should set documented defaults."""
    cfg = LocalPolyConfig()

    # rel_steps normalized to tuple of floats
    assert isinstance(cfg.rel_steps, tuple)
    assert cfg.rel_steps == (0.01, 0.02, 0.04, 0.08)

    assert cfg.tol_rel == 0.01
    assert cfg.tol_abs == 1e-10
    assert cfg.min_samples == 9
    assert cfg.max_trim == 10
    assert cfg.max_degree == 7
    assert cfg.center is True


def test_local_poly_config_rel_steps_from_list():
    """Tests that rel_steps passed as list should be converted to tuple of floats."""
    rel_steps = [0.1, 0.2, 0.4]
    cfg = LocalPolyConfig(rel_steps=rel_steps)

    assert isinstance(cfg.rel_steps, tuple)
    assert cfg.rel_steps == (0.1, 0.2, 0.4)
    for s in cfg.rel_steps:
        assert isinstance(s, float)


def test_local_poly_config_rel_steps_from_numpy_array():
    """Tests that rel_steps from a numpy array should also become a tuple of floats."""
    rel_steps = np.array([0.01, 0.03, 0.07], dtype=np.float64)
    cfg = LocalPolyConfig(rel_steps=rel_steps)

    assert isinstance(cfg.rel_steps, tuple)
    assert cfg.rel_steps == (0.01, 0.03, 0.07)
    for s in cfg.rel_steps:
        assert isinstance(s, float)


def test_local_poly_config_rel_steps_from_generator():
    """Tests that rel_steps from a generator should be consumed into a tuple."""
    def gen():
        for v in (0.05, 0.1, 0.2):
            yield v

    cfg = LocalPolyConfig(rel_steps=gen())

    assert isinstance(cfg.rel_steps, tuple)
    assert cfg.rel_steps == (0.05, 0.1, 0.2)
    for s in cfg.rel_steps:
        assert isinstance(s, float)


def test_local_poly_config_custom_parameters():
    """Tests that custom parameters should be stored exactly as given."""
    cfg = LocalPolyConfig(
        rel_steps=(0.02, 0.03),
        tol_rel=5e-3,
        tol_abs=1e-8,
        min_samples=5,
        max_trim=3,
        max_degree=4,
        center=False,
    )

    assert cfg.rel_steps == (0.02, 0.03)
    assert cfg.tol_rel == 5e-3
    assert cfg.tol_abs == 1e-8
    assert cfg.min_samples == 5
    assert cfg.max_trim == 3
    assert cfg.max_degree == 4
    assert cfg.center is False
