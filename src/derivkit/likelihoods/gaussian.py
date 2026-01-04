"""Gaussian likelihood function module."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import multivariate_normal

from derivkit.utils.linalg import normalize_covariance

__all__ = [
    "build_gaussian_likelihood",
]


def build_gaussian_likelihood(
    data: ArrayLike,
    model_parameters: ArrayLike,
    cov: ArrayLike,
    return_log: bool = True,
    ) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Constructs the Gaussian likelihood function.

    Args:
        data: a 1D or 2D array representing the given data values. It is
            expected that axis 0 represents different samples of data while
            axis 1 represents the data values.
        model_parameters: a 1D array representing the theoretical values
            of the model parameters.
        cov: covariance matrix. May be a scalar, a 1D vector of diagonal variances,
            or a full 2D covariance matrix. It will be symmetrised and normalized
            internally to ensure compatibility with the data and model_parameters.
        return_log: when set to ``True``, the function will compute the
            log-likelihood instead.

    Returns:
        A tuple:
          - coordinate_grids: tuple of 1D arrays giving the evaluation coordinates
            for each dimension (one array per dimension), ordered consistently with
            the first axis of ``data``.
          - probability_density: ndarray with the values of the multivariate
            Gaussian probability density function evaluated on the Cartesian
            product of those coordinates.

    Raises:
        ValueError: raised if
            - data is not 1D or 2D,
            - model_parameters is not 1D,
            - the number of samples in data does not match the number of
              model parameters,
            - model_parameters contain non-finite values,
            - cov cannot be normalized to a valid covariance matrix.

    Examples:
        A 1D Gaussian likelihood:
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from derivkit.likelihoods.gaussian import build_gaussian_likelihood
            >>> data = np.linspace(-10, 10, 100)[np.newaxis, :]
            >>> model_parameters = np.array([1.0])
            >>> cov = np.array([[2.0]])
            >>> x_grid, pdf = build_gaussian_likelihood(data, model_parameters, cov)
            >>> plt.plot(x_grid[0], pdf[0])  # doctest: +SKIP
        A 2D Gaussian likelihood:
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> data = np.asarray((np.linspace(-10, 10, 30), np.linspace(3, 6, 30)))
            >>> model_parameters = np.array([0.0, 4.0])
            >>> cov = np.array([[1.0, 0.2], [0.2, 0.3]])
            >>> # Build coordinate arrays and evaluate the probability density on their
            >>> # Cartesian product. The indexing ensures the coordinate order matches
            >>> # the order in ``data``.
            >>> grid, probability_density = build_gaussian_likelihood(data, model_parameters, cov)
            >>> plt.contour(*grid, probability_density)  # doctest: +SKIP
    """
    # The data is expected to be 2D. However, 1D is allowed, since it can be
    # embedded in a 2D space.
    _data = np.array(data, dtype=float, copy=True)
    if not np.isfinite(_data).all():
        raise ValueError("data contain non-finite values.")
    if _data.ndim == 1:
        _data = _data[np.newaxis, :]
    elif _data.ndim > 2:
        raise ValueError(f"data must be a 1D or 2D array, but is a {_data.ndim}D array.")

    number_samples = _data.shape[0]
    model_parameters = np.asarray(model_parameters, dtype=float)
    if model_parameters.ndim != 1:
        raise ValueError(
            "model_parameters must be a 1D array, "
            f"but is a {model_parameters.ndim}D array."
        )
    model_parameters = model_parameters.ravel()
    if not np.isfinite(model_parameters).all():
        raise ValueError("model_parameters contain non-finite values.")

    number_model_parameters = model_parameters.size
    if number_samples != number_model_parameters:
        raise ValueError(
            "There must be as many model parameters as there are samples of data. "
            f"(n_params={number_model_parameters}, n_samples={number_samples})"
        )

    cov = np.asarray(cov, dtype=float)
    if not np.isfinite(cov).all():
        raise ValueError("cov contains non-finite values.")
    cov_dim = cov.ndim
    cov_shape = cov.shape
    is_scalar = cov_dim == 0
    is_valid_vector = cov_dim == 1 and cov_shape[0] == number_model_parameters
    is_valid_matrix = (
            cov_dim == 2
            and cov_shape[0] == cov_shape[1] == number_model_parameters
    )
    if not (is_scalar or is_valid_vector or is_valid_matrix):
        raise ValueError(
            "Input cov is not compatible with input model_parameters."
        )

    sigma = normalize_covariance(
        (cov+cov.T)/2,
        n_parameters=number_model_parameters
    )

    # The data are coordinate vectors, which have to be extended into a
    # coordinate grid (meshgrid). The grids are then combined to give a
    # box of coordinates (dstack), which is then sent to the PDF. The
    # indexing in meshgrid should ensure that the ordering of the grids
    # corresponds to the ordering of the original data.
    coordinate_grids = np.meshgrid(*_data, indexing="ij")
    coordinate_box = np.dstack(coordinate_grids)
    distribution = multivariate_normal(mean=model_parameters, cov=sigma)
    probabilities = distribution.logpdf(coordinate_box) \
        if return_log \
        else distribution.pdf(coordinate_box)
    return coordinate_grids, probabilities
