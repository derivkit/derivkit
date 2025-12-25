"""Unit tests for derivkit.autodiff.jax_utils module."""

from __future__ import annotations

import pytest
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from derivkit.autodiff.jax_utils import (
    apply_array_nd,
    apply_scalar_1d,
    apply_scalar_nd,
    require_jax,
)


def test_require_jax_noop_when_available() -> None:
    """Tests that require_jax does not raise when JAX is available."""
    require_jax()


def test_apply_scalar_1d_returns_scalar_jax_array() -> None:
    """Tests that apply_scalar_1d returns a scalar JAX array."""
    out = apply_scalar_1d(lambda x: x * x, "apply_scalar_1d", jnp.array(2.0))
    assert out.shape == ()
    assert float(out) == pytest.approx(4.0)


def test_apply_scalar_1d_raises_if_output_not_scalar() -> None:
    """Tests that apply_scalar_1d raises TypeError if output is not scalar."""
    with pytest.raises(TypeError):
        apply_scalar_1d(
            lambda x: jnp.array([x, x + 1.0]),
            "apply_scalar_1d",
            jnp.array(2.0),
        )


def test_apply_scalar_nd_returns_scalar_jax_array() -> None:
    """Tests that apply_scalar_nd returns a scalar JAX array."""
    theta = jnp.array([1.0, 2.0, 3.0])
    out = apply_scalar_nd(lambda t: jnp.sum(t), "apply_scalar_nd", theta)
    assert float(out) == pytest.approx(6.0)


def test_apply_scalar_nd_raises_if_output_not_scalar() -> None:
    """Tests that apply_scalar_nd raises TypeError if output is not scalar."""
    theta = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        apply_scalar_nd(lambda t: t, "apply_scalar_nd", theta)


def test_apply_array_nd_returns_array_jax_array() -> None:
    """Tests that apply_array_nd returns an array JAX array."""
    theta = jnp.array([1.0, 2.0, 3.0])
    out = apply_array_nd(lambda t: jnp.stack([t, 2.0 * t]), "apply_array_nd", theta)
    assert out.shape == (2, 3)


def test_apply_array_nd_raises_if_output_is_scalar() -> None:
    """Tests that apply_array_nd raises TypeError if output is scalar."""
    theta = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        apply_array_nd(lambda t: jnp.sum(t), "apply_array_nd", theta)
