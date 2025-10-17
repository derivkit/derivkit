"""Unit tests for forecasting/calculus.py."""

import numpy as np
import pytest

import derivkit.forecasting.likelihoods as dkl


def test_gaussian_likelihood():
    """Test that build_gaussian_likelihood handles input and output correctly."""
    # Check that 3D data raises ValueError
    with pytest.raises(ValueError):
        data = np.ones((1, 1, 1))
        model_parameter = np.ones(1)
        cov = np.ones((1, 1))
        dkl.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that 2D model_parameter raises ValueError
    with pytest.raises(ValueError):
        data = np.ones((1, 1))
        model_parameter = np.ones((1, 2))
        cov = np.ones((1, 1))
        dkl.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that model_parameter.size must match axis 0 of data
    with pytest.raises(ValueError):
        data = np.ones((2, 1))
        model_parameter = np.ones((1))
        cov = np.ones((1, 1))
        dkl.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that cov must be square with axes length equal to model_parameters
    with pytest.raises(ValueError):
        data = np.ones((3, 10))
        model_parameter = np.ones((2))
        cov = np.ones((3, 2))
        dkl.build_gaussian_likelihood(data, model_parameter, cov)
    with pytest.raises(ValueError):
        data = np.ones((3, 10))
        model_parameter = np.ones((3))
        cov = np.ones((2, 2))
        dkl.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that function outputs have the correct types
    data = np.array([[1, 2, 3], [4, 3, 1]])
    model_parameters = np.array([1, 2])
    cov = np.eye(2)
    result = dkl.build_gaussian_likelihood(data, model_parameters, cov)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result, tuple) and \
        all(isinstance(el, np.ndarray) for el in result[0])
    assert isinstance(result[1], np.ndarray)
