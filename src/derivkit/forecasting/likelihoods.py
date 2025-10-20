"""Basic likelihood functions for forecasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import multivariate_normal, poisson

from derivkit.utils import normalize_covariance


def build_gaussian_likelihood(
    data: ArrayLike,
    model_parameters: ArrayLike,
    cov: ArrayLike,
    ) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Constructs the Gaussian likelihood function.

    Args:
        data: a 1D or 2D array representing the given data values. It is
            expected that axis 0 represents different samples of data while
            axis 1 represents the data values.
        model_parameters: a 1D array representing the theoretical values
            of the model parameters.
        cov: covariance matrix. May be a scalar, a 1D vector of diagonal variances,
            or a full 2D covariance matrix. It will be normalized internally
            to ensure compatibility with the data and model_parameters.

    Returns:
        A tuple:
          - coordinate_grids: tuple of ndarrays from meshgrid (one per axis).
          - pdf: ndarray of the Gaussian PDF evaluated on the grid.

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
            >>> grid, pdf = build_gaussian_likelihood(data, model_parameters, cov)
            >>> plt.contour(*grid, pdf)  # doctest: +SKIP
    """
    # The data is expected to be 2D. However, 1D is allowed, since it can be
    # embedded in a 2D space.
    _data = np.array(data, dtype=float, copy=True)
    if _data.ndim == 1:
        _data = _data[np.newaxis, :]
    elif _data.ndim > 2:
        raise ValueError(f"data must be a scalar, 1D or 2D array, but is a {_data.ndim}D array.")

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

    sigma = normalize_covariance(cov, n_parameters=number_model_parameters)

    # The data are coordinate vectors, which have to be extended into a
    # coordinate grid (meshgrid). The grids are then combined to give a
    # box of coordinates (dstack), which is then sent to the PDF. The
    # indexing in meshgrid should ensure that the ordering of the grids
    # corresponds to the ordering of the original data.
    coordinate_grids = np.meshgrid(*_data, indexing="ij")
    coordinate_box = np.dstack(coordinate_grids)
    distribution = multivariate_normal(mean=model_parameters, cov=sigma)
    return coordinate_grids, distribution.pdf(coordinate_box)


def build_poissonian_likelihood(
    data: float | np.ndarray[float],
    model_parameters: float | np.ndarray[float],
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Constructs the Poissonian likelihood function.

    The shape of the data products depend on the shape of ``model_parameters``.
    The assumption is that ``model_parameters`` contains the expectation value
    of some quantity which is either uniform for the entire distribution or is
    distributed across a grid of bins. It is uniform for the entire distribution
    if it is a scalar.

    The function will try to reshape ``data`` to align with ``model_parameters``.
    If ``model_parameters`` is a scalar, then ``data`` will be flattened. Otherwise,
    the grid can contain any number of axes, but currently the number of axes
    is hardcoded to 2. Supplying a higher-dimensional array to
    ``model_parameters`` may produce unexpected results.

    This hardcoded limit means that, while it is possible to supply
    ``model_parameters`` along a 1D grid, the output shape will always be a
    2D row-major array. See Examples for more details.

    Args:
        data: an array representing the given data values.
        model_parameters: an array representing the means of the data samples.

    Returns:
        A tuple of arrays containing (in order):

            - the data, reshaped to align with the model parameters.
            - the values of the Poissonian probability mass function computed
              from the data and model parameters.

    Raises:
        ValueError: If any of the model_parameters are negative or non-finite,
            or the data points cannot be reshaped to align with
            model_parameters.

    Examples:
        The Poissonian probability of 2 events, given that the mean is
        1.4 events per unit interval, shows that the output is reshaped
        as a 2D array:

        >>> x, y = build_poissonian_likelihood(2, 1.4)
        >>> print(x, y)
        [2] [0.24166502]

        A Poisson-distributed sample can be computed for a given
        expectation value:

        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> model_parameters = 2.4
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [ 1  2  3  4  5  6  7  8  9 10]
        >>> print(y)
        [2.17723088e-01 2.61267705e-01 2.09014164e-01 1.25408499e-01
         6.01960793e-02 2.40784317e-02 8.25546231e-03 2.47663869e-03
         6.60436985e-04 1.58504876e-04]

        Note that the shape of the results are determined by the shape of
        ``model_parameters``:

        >>> data = np.array([1, 2])
        >>> model_parameters = np.array([3])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [[1]
         [2]]
        >>> print(y)
        [[0.14936121]
         [0.22404181]]

        Probabilities computed from values and parameters distributed along
        a 1D grid of bins:

        >>> model_parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [[1 2 3 4 5 6]]
        >>> print(y)
        [[9.04837418e-02 1.63746151e-02 3.33368199e-03 7.15008049e-04
          1.57950693e-04 3.55629940e-05]]

        Probabilities computed from values and parameters distributed across
        a 2D grid of bins:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> model_parameters = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [[[1 2 3]
          [4 5 6]]]
        >>> print(y)
        [[[9.04837418e-02 1.63746151e-02 3.33368199e-03]
          [7.15008049e-04 1.57950693e-04 3.55629940e-05]]]

        Combining multiple data values on the same grid with the same
        Poissonian means:

        >>> val1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> val2 = np.array([[7, 8, 9], [10, 11, 12]])
        >>> data = np.array([val1, val2])
        >>> model_parameters = np.array([[0.1, 0.2, 0.3,], [0.4, 0.5, 0.6]])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [[[ 1  2  3]
          [ 4  5  6]]
        <BLANKLINE>
         [[ 7  8  9]
          [10 11 12]]]
        >>> print(y)
         [[[9.04837418e-02 1.63746151e-02 3.33368199e-03]
          [7.15008049e-04 1.57950693e-04 3.55629940e-05]]
        <BLANKLINE>
         [[1.79531234e-11 5.19829050e-11 4.01827740e-11]
          [1.93695302e-11 7.41937101e-12 2.49402815e-12]]]

        The same result can be obtained by supplying the data in a flattened
        array:

        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> print(x)
        [[[ 1  2  3]
          [ 4  5  6]]
        <BLANKLINE>
         [[ 7  8  9]
          [10 11 12]]]
        >>> print(y)
         [[[9.04837418e-02 1.63746151e-02 3.33368199e-03]
          [7.15008049e-04 1.57950693e-04 3.55629940e-05]]
        <BLANKLINE>
         [[1.79531234e-11 5.19829050e-11 4.01827740e-11]
          [1.93695302e-11 7.41937101e-12 2.49402815e-12]]]
    """
    values_to_reshape = np.asarray(data)
    parameters = np.asarray(model_parameters)

    if np.any(parameters < 0):
        raise ValueError("values of model_parameters must be non-negative.")
    if np.any(~np.isfinite(parameters)):
        raise ValueError("values of model_parameters must be finite.")

    try:
        counts = values_to_reshape.reshape(-1, *parameters.shape[-2:])
    except ValueError:
        raise ValueError(
            "data cannot be reshaped to align with model_parameters: "
            f"data.shape={values_to_reshape.shape} is incompatible with "
            f"model_parameters.shape={parameters.shape}."
        )

    return counts, poisson.pmf(counts, parameters)


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
