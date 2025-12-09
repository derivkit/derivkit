"""Unit tests for forecasting.likelihoods.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import derivkit.likelihoods.gaussian as gauss


def test_gaussian_likelihood_basic_shape_checks():
    """Checks that build_gaussian_likelihood raises exceptions for bad input shapes."""
    # Check that 3D data raises a ValueError
    with pytest.raises(ValueError):
        data = np.ones((1, 1, 1))
        model_parameter = np.ones(1)
        cov = np.ones((1, 1))
        gauss.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that 2D model parameters raise a ValueError
    with pytest.raises(ValueError):
        data = np.ones((1, 1))
        model_parameter = np.ones((1, 2))
        cov = np.ones((1, 1))
        gauss.build_gaussian_likelihood(data, model_parameter, cov)

    # Check that mismatched data and parameter sizes raise a ValueError
    with pytest.raises(ValueError):
        data = np.ones((2, 1))
        model_parameter = np.ones(1)
        cov = np.ones((1, 1))
        gauss.build_gaussian_likelihood(data, model_parameter, cov)


def test_gaussian_likelihood_accepts_scalar_diag_full_cov():
    """Tests that build_gaussian_likelihood works for scalar, diag, and full covariances."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 1.0]])
    mu = np.array([1.0, 2.0])

    # A scalar covariance should be interpreted as a multiple of the identity matrix
    grids, pdf = gauss.build_gaussian_likelihood(data, mu, 2.0)
    assert isinstance(grids, tuple) and len(grids) == 2
    assert pdf.shape == (data.shape[1], data.shape[1])
    assert np.isfinite(pdf).all()

    # A 1D vector should be treated as a diagonal covariance
    grids, pdf = gauss.build_gaussian_likelihood(data, mu, np.array([1.0, 0.5]))
    assert np.isfinite(pdf).all()

    # A full 2D covariance matrix should be handled directly
    cov = np.array([[1.0, 0.2], [0.2, 0.5]])
    grids, pdf = gauss.build_gaussian_likelihood(data, mu, cov)
    assert np.isfinite(pdf).all()

def test_gaussian_likelihood_nonfinite_raises():
    """Tests that non-finite inputs in Gaussian likelihood raise ValueError."""
    data = np.array([[0.0, 1.0]])
    mu = np.array([0.0])

    # Non-finite tests for data
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(np.append(data, np.inf), mu, np.array([np.nan]))
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(np.append(data, np.nan), mu, np.array([[np.inf]]))

    # Non-finite tests for model_parameters
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(data, np.append(mu, np.nan), np.array([np.nan]))
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(data, np.append(mu, np.inf), np.array([[np.inf]]))

    # Non-finite tests for cov
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(data, mu, np.array([np.nan]))
    with pytest.raises(ValueError):
        gauss.build_gaussian_likelihood(data, mu, np.array([[np.inf]]))


def test_gaussian_likelihood_output_types():
    """Tests that build_gaussian_likelihood returns correct types."""
    data = np.array([[1, 2, 3], [4, 3, 1]])
    model_parameters = np.array([1, 2])
    cov = np.eye(2)
    result = gauss.build_gaussian_likelihood(data, model_parameters, cov)
    assert isinstance(result, tuple) and len(result) == 2
    grids, pdf = result
    assert isinstance(grids, tuple) and all(isinstance(g, np.ndarray) for g in grids)
    assert isinstance(pdf, np.ndarray)


@pytest.mark.parametrize(
  (
    "input_data, "
    "input_parameters, "
    "input_cov, "
    "expected_output_data, "
    "expected_output_likelihood"
  ),
  [
    pytest.param(
        np.array([2]),
        np.array([1.4]),
        np.array([2]),
        (np.array([2.]),),
        np.array([0.2578152274047408])
    ),
    pytest.param(
        np.array([-2, -1, 0, 1, 2]),
        np.array([1.4]),
        np.array([2]),
        (np.array([-2., -1., 0., 1., 2.]),),
        np.array([0.01567776, 0.06683609, 0.17281872, 0.2710337 , 0.25781523])
    ),
    pytest.param(
        np.array([-2, -1, 0, 1, 2]),
        np.array([1.4]),
        np.array([2]),
        (np.array([-2., -1., 0., 1., 2.]),),
        np.array([0.01567776, 0.06683609, 0.17281872, 0.2710337 , 0.25781523])
    ),
    pytest.param(
        np.array(
            [
                [-2, -1, 0, 1, 2],
                [0, 1, 2, 3, 4]
            ]
        ),
        np.array([1.4, 2]),
        2,
        (
            np.array(
              [
                [-2., -2., -2., -2., -2.],
                [-1., -1., -1., -1., -1.],
                [ 0.,  0.,  0.,  0.,  0.],
                [ 1.,  1.,  1.,  1.,  1.],
                [ 2.,  2.,  2.,  2.,  2.]
              ]
            ),
            np.array(
              [
                [0., 1., 2., 3., 4.],
                [0., 1., 2., 3., 4.],
                [0., 1., 2., 3., 4.],
                [0., 1., 2., 3., 4.],
                [0., 1., 2., 3., 4.]
              ]
            )
        ),
        np.array(
          [
            [0.00162699, 0.00344434, 0.00442261, 0.00344434, 0.00162699],
            [0.00693604, 0.0146836 , 0.01885411, 0.0146836 , 0.00693604],
            [0.01793459, 0.03796752, 0.04875126, 0.03796752, 0.01793459],
            [0.02812703, 0.05954492, 0.07645719, 0.05954492, 0.02812703],
            [0.02675526, 0.05664088, 0.07272833, 0.05664088, 0.02675526]
          ]
        )
    )
  ]
)
class TestGaussOutput:
    """A container for Gaussian likelihood output tests.

    All tests in this class have access to the same parameters.
    """
    def test_gaussian_likelihood_output(
        self,
        input_data,
        input_parameters,
        input_cov,
        expected_output_data,
        expected_output_likelihood
   ):
        """Tests that build_gaussian_likelihood produces the expected output."""
        test_output, test_likelihood = gauss.build_gaussian_likelihood(
            input_data, input_parameters, input_cov
        )

        assert_allclose(test_output, expected_output_data, rtol=2e-6)
        assert_allclose(test_likelihood, expected_output_likelihood, rtol=2e-6)


    def test_log_likelihood(
        self,
        input_data,
        input_parameters,
        input_cov,
        expected_output_data,
        expected_output_likelihood
    ):
       """Tests that Gaussian log-likelihoods are computed correctly."""
       _, test_likelihood = gauss.build_gaussian_likelihood(
          input_data,
          input_parameters,
          input_cov,
          return_log=True
       )
       assert_allclose(np.log(expected_output_likelihood), test_likelihood, rtol=2e-6)
