"""Unit tests for forecasting/calculus.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import derivkit.forecasting.likelihoods as dkl


def test_gaussian_likelihood():
    """Test that build_gaussian handles input and output correctly."""
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

@pytest.mark.parametrize(
  (
    "input_data, "
    "input_parameters, "
    "expected_output_data, "
    "expected_output_likelihood"
  ),
  [
    # Test case to check that scalars can be used to generate probabilities.
    pytest.param(
        2,
        1.4,
        np.array([2]),
        np.array([0.24166502])
    ),
    # Test case to check that samples of data can be used to generate
    # probabilities in a single distribution.
    pytest.param(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        2.4,
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array(
            [2.17723088e-01, 2.61267705e-01, 2.09014164e-01, 1.25408499e-01,
             6.01960793e-02, 2.40784317e-02, 8.25546231e-03, 2.47663869e-03,
             6.60436985e-04, 1.58504876e-04]
        ),
    ),
    # Test case to check that the output shape is determined by the
    # shape of the ``model_parameters`` input.
    pytest.param(
        np.array([1, 2]),
        np.array([3]),
        np.array([[1], [2]]),
        np.array([[0.14936121], [0.22404181]])
    ),
    # Test case to check that single probabilities can be drawn from
    # an ensemble of distributions.
    pytest.param(
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        np.array([[1, 2, 3, 4, 5, 6]]),
        np.array([
            [9.04837418e-02, 1.63746151e-02, 3.33368199e-03,
             7.15008049e-04, 1.57950693e-04, 3.55629940e-05]
        ]),
    ),
    # Test case to check that probabilities can be drawn from ensembles
    # on a 2D grid.
    pytest.param(
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        np.array([[[1, 2, 3], [4, 5, 6]]]),
        np.array([[
            [9.04837418e-02, 1.63746151e-02, 3.33368199e-03],
            [7.15008049e-04, 1.57950693e-04, 3.55629940e-05]
        ]]),
    ),
    # Test case to check that multiple samples can be drawn from the
    # same 2d grid.
    pytest.param(
        np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        ),
        np.array(
            [
                [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
            ]
        ),
        np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        ),
        np.array(
            [
                [
                    [9.04837418e-02, 1.63746151e-02, 3.33368199e-03],
                    [7.15008049e-04, 1.57950693e-04, 3.55629940e-05]
                ],
                [
                    [1.79531234e-11, 5.19829050e-11, 4.01827740e-11],
                    [1.93695302e-11, 7.41937101e-12, 2.49402815e-12]
                ]
            ]
        ),
    ),
    # Test case to check that the same
    pytest.param(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        np.array(
            [
                [0.1, 0.2, 0.3,], [0.4, 0.5, 0.6]
            ]
        ),
        np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]
            ]
        ),
        np.array(
            [
                [
                    [9.04837418e-02, 1.63746151e-02, 3.33368199e-03],
                    [7.15008049e-04, 1.57950693e-04, 3.55629940e-05]
                ],
                [
                    [1.79531234e-11, 5.19829050e-11, 4.01827740e-11],
                    [1.93695302e-11, 7.41937101e-12, 2.49402815e-12]
                ]
            ]
        ),
    ),
  ]
)

def test_poissonian_likelihood_negative_parameters(
    input_data,
    input_parameters,
    expected_output_data,
    expected_output_likelihood
):
    """Tests that build_poissonian_likelihood produces the expected output.

    Test cases were originally made for examples in the build_poissonian docstring.
    """
    test_output, test_likelihood \
        = dkl.build_poissonian_likelihood(input_data, input_parameters)

    assert_allclose(test_output, expected_output_data)
    assert_allclose(test_likelihood, expected_output_likelihood)

@pytest.mark.parametrize(
  "test_data, test_parameters",
  [
    (1, -1),
    (1, np.inf),
    (1, np.nan),
    ([1, 2, 3], [0.1, 0.2, np.inf]),
    ([1, 2, 3], [0.1, 0.2, np.nan]),
    ([1, 2, 3], [0.1, 0.2, -1.2]),
    ([[1, 2, 3], [4, 5, 6]], -1),
    ([[1, 2, 3], [4, 5, 6]], [[0.1, 0.2, 0.3], [0.4, -0.1, 0.6]]),
    ([[1, 2, 3], [4, 5, 6]], [[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]]),
  ]
)
def test_poissonian_likelihood_incompatible_shapes(test_data, test_parameters):
    """Tests negative parameter value exception triggers."""
    with pytest.raises(ValueError):
        dkl.build_poissonian_likelihood(test_data, test_parameters)

@pytest.mark.parametrize(
  "test_data, test_parameters",
  [
    (1, [1, 2]),
    ([1, 2, 3], [1, 2]),
    ([1, 2, 3, 4, 5], [1, 2]),
  ]
)
def test_poissonian_likelihood_raises(test_data, test_parameters):
    """Tests build_poissonian_likelihood shape mismatch exception triggers."""
    with pytest.raises(ValueError):
        dkl.build_poissonian_likelihood(test_data, test_parameters)
