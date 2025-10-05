"""Basic likelihood functions for forecasting."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

def build_gaussian_likelihood(
    data: np.ndarray,
    model_parameters: np.ndarray,
    cov: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Constructs the Gaussian likelihood function.

    Args:
        data: a 1D or 2D array representing the given data values. It is
            expected that axis 0 represents different samples of data while
            axis 1 represents the data values.
        model_parameters: a 1D array representing the theoretical values
            of the model parameters.
        cov: the covariance matrix. Must be either a 1D or 2D array.
            If a 1D array is supplied it must be reshapable to a 2D array
            consistent with the length of model_parameters.

    Returns:
        ``np.ndarray``: an array containing the coordinate grids constructed
            from the data.
        ``np.ndarray``: an array containing the values of the Gaussian
            likelihood function on the coodinate grid.

    Raises:
        ValueError: raised if

            - data is not a 1D or 2D array,
            - model_parameters is not a 1D array,
            - the length of model_parameters is not equal to axis 1 of data,
            - cov is a 1D array and cannot be reshaped as a 2D array compatible
              with data and model_parameters.
            - cov is not a 2D array compatible with data and model_parameters.

    Examples:
        A 1D Gaussian likelihood:
            >>> data = np.linspace(-10, 10, 100)[np.newaxis, :]
            >>> model_parameters = np.array([1])
            >>> cov = np.array([[2]])
            >>> x, y = build_gaussian(data, model_parameters, cov)
            >>> plt.scatter(x, y)
        A 2D Gaussian likelihood:
            >>> data = np.asarray((
            ...     np.linspace(-10, 10, 30),
            ...     np.linspace(3, 6, 30),
            ... ))
            >>> model_parameters = np.array([0, 4])
            >>> cov = np.array([[1, 0.2], [0.2, 0.3]])
            >>> grid, pdf = build_gaussian(data, model_parameters, cov)
            >>> plt.contour(*grid, pdf)
    """
    # The data is expected to be 2D. However, 1D is allowed, since it can be
    # embedded in a 2D space.
    _data = np.copy(data)
    number_samples = _data.shape[0]
    number_model_parameters = model_parameters.shape[0]
    if _data.ndim == 1:
        _data = np.array([[*_data]])
    elif _data.ndim > 2:
        raise ValueError(
            f"data must be a 2D array, but is a {data.ndim}D array."
        )
    if model_parameters.ndim != 1:
        raise ValueError(
            "model_parameters must be a 1D array, "
            f"but is a {model_parameters.ndim}D array."
        )
    if number_samples != number_model_parameters:
        raise ValueError(
            "There must be as many model parameters as there are samples of data."
            f" Number of model parameters: {number_model_parameters}. "
            f"Types of data: {number_samples}."
        )
    square_shape = (number_samples, number_model_parameters)
    if cov.ndim == 1:
        cov = cov.reshape(square_shape)
    elif cov.shape != square_shape:
        raise ValueError(
            "cov must be a square 2D array of shape "
            f"{square_shape}. Actual shape {cov.shape}."
        )

    # The data are coordinate vectors, which have to be extended into a
    # coordinate grid (meshgrid). The grids are then combined to give a
    # box of coordinates (dstack), which is then sent to the PDF. The
    # indexing in meshgrid should ensure that the ordering of the grids
    # corresponds to the ordering of the original data.
    coordinate_grids = np.meshgrid(*_data, indexing="ij")
    coordinate_box = np.dstack(coordinate_grids)
    distribution = multivariate_normal(mean=model_parameters, cov=cov)
    return coordinate_grids, distribution.pdf(coordinate_box)


def poisson(*args, **kwargs):
    """This is a placeholder for a Poisson likelihood function."""
    raise NotImplementedError
def binomial(*args, **kwargs):
    """This is a placeholder for a Binomial likelihood function."""
    raise NotImplementedError
def multinomial(*args, **kwargs):
    """This is a placeholder for a Multinomial likelihood function."""
    raise NotImplementedError
def student_t(*args, **kwargs):
    """This is a placeholder for a Student's t-distribution likelihood function."""
    raise NotImplementedError
def sellentin_heavens(*args, **kwargs):
    """This is a placeholder for the Sellentin-Heavens likelihood function."""
    raise NotImplementedError
