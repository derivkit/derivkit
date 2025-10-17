"""Basic likelihood functions for forecasting."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, poisson


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


def build_poissonian_likelihood(
    data: float | np.ndarray[float],
    model_parameters: float | np.ndarray[float],
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Constructs the Poissonian likelihood function.

    The shape of the data products depend on the shape of ``model_parameters``.
    The assumption is that ``model_parameters`` contains the expectation value
    of some quantity distributed across a grid of bins. The function will try
    to reshape ``data`` to align with this grid. In principle the grid can
    contain any number of axes, but currently the number of axes is hardcoded
    to 2. Supplying a higher-dimensional array to ``model_parameters`` may
    produce unexpected results.

    This hardcoded limit means that, while it is possible to supply
    ``model_parameters`` along a 1D grid, the output shape will always be a
    2D row-major array. Similarly, it is possible to supply a single value to
    ``model_parameters``. The output will still be a 2D array. See Examples for
    more details.

    Args:
        data: an array representing the given data values.
        model_parameters: an array representing the means of the data samples.

    Returns:
        ``np.ndarray``: an array containing the probability intervals
            constructed from the data.
        ``np.ndarray``: an array containing the values of the Poissonian
            probibility mass function on the probability intervals.

    Raises:
        ValueError: If any of the model_parameters are negative, or the data
            points cannot be reshaped to align with model_parameters.

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

    try:
        number_events = values_to_reshape.reshape(-1, *parameters.shape[-2:])
    except ValueError:
        raise ValueError(
            "data cannot be reshaped to align with model_parameters: "
            f"data.shape={values_to_reshape.shape} is incompatible with "
            f"model_parametres.shape={parameters.shape}."
        )

    return number_events, poisson.pmf(number_events, parameters)


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
