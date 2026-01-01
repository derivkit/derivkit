"""Unit tests for JAX-based automatic differentiation engine integration."""

from __future__ import annotations

import pytest

from derivkit.autodiff.jax_autodiff import (
    AutodiffDerivative,
    register_jax_autodiff_backend,
)
from derivkit.autodiff.jax_utils import AutodiffUnavailable, require_jax
from derivkit.derivative_kit import DerivativeKit


def _skip_if_no_jax() -> None:
    """Skips the test if JAX is not installed."""
    try:
        require_jax()
    except AutodiffUnavailable:
        pytest.skip('JAX not installed; install with `pip install "derivkit[jax]"`.')


def square_function(x: float) -> float:
    """Square function."""
    return x**2


def cubic_function(x: float) -> float:
    """Cubic function."""
    return x**3


def test_autodiff_derivative_engine_first_order() -> None:
    """Tests that AutodiffDerivative computes first-order derivatives correctly."""
    _skip_if_no_jax()

    eng = AutodiffDerivative(square_function, 3.0)

    val = eng.differentiate(order=1)
    assert val == pytest.approx(6.0)


def test_autodiff_derivative_engine_higher_order() -> None:
    """Tests that AutodiffDerivative computes higher-order derivatives correctly."""
    _skip_if_no_jax()

    eng = AutodiffDerivative(cubic_function, 2.0)

    val = eng.differentiate(order=2)
    assert val == pytest.approx(12.0)


def test_register_jax_autodiff_backend_allows_use_in_derivativekit() -> None:
    """Tests that registering JAX autodiff backend allows its use in DerivativeKit."""
    _skip_if_no_jax()

    register_jax_autodiff_backend(name="autodiff_test")

    dk = DerivativeKit(square_function, 3.0)

    val = dk.differentiate(method="autodiff_test", order=1)
    assert val == pytest.approx(6.0)


def test_register_jax_autodiff_backend_alias() -> None:
    """Tests that registering JAX autodiff backend with alias works in DerivativeKit."""
    _skip_if_no_jax()

    register_jax_autodiff_backend(name="autodiff_alias_test", aliases=("jax_test",))

    dk = DerivativeKit(square_function, 4.0)

    val = dk.differentiate(method="jax_test", order=1)
    assert val == pytest.approx(8.0)
