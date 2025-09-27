"""Tests for LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def test_order():
    """High derivative orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        like._get_derivatives(order=3)
    with pytest.raises(ValueError):
        like._get_derivatives(order=np.random.randint(low=4, high=30))


def test_forecast_order():
    """High forecast orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))

    with pytest.raises(ValueError):
        like.get_forecast_tensors(forecast_order=3)

    with pytest.raises(ValueError):
        like.get_forecast_tensors(forecast_order=np.random.randint(low=4, high=30))
