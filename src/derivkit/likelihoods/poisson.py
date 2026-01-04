"""Poissonian likelihood function module."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

__all__ = [
    "build_poissonian_likelihood",
]


def build_poissonian_likelihood(
    data: float | np.ndarray[float],
    model_parameters: float | np.ndarray[float],
    return_log: bool = False,
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
        return_log: when set to ``True``, returns the log-likelihood. Defaults
            to ``False``.

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
        Scalar mean + scalar data:

        >>> import numpy as np
        >>> from scipy.stats import poisson
        >>> from derivkit.likelihoods.poisson import build_poissonian_likelihood
        >>> x, y = build_poissonian_likelihood(2, 1.4)
        >>> x.shape, y.shape
        ((1,), (1,))
        >>> x[0].item()
        2
        >>> np.allclose(y[0], poisson.pmf(2, 1.4))
        True

        Vector data + scalar mean (data are flattened):

        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> model_parameters = 2.4
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> x.shape, y.shape
        ((10,), (10,))
        >>> np.array_equal(x, data)
        True
        >>> np.allclose(y, poisson.pmf(data, 2.4))
        True

        Shape follows ``model_parameters``:

        >>> data = np.array([1, 2])
        >>> model_parameters = np.array([3])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> x.shape, y.shape
        ((2, 1), (2, 1))
        >>> np.array_equal(x[:, 0], data)
        True
        >>> np.allclose(y[:, 0], poisson.pmf(data, 3))
        True

        1D grid of bins produces a row-major 2D output:

        >>> model_parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> x.shape, y.shape
        ((1, 6), (1, 6))
        >>> np.array_equal(x[0], data)
        True
        >>> np.allclose(y[0], poisson.pmf(data, model_parameters))
        True

        2D grid:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> model_parameters = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> x.shape, y.shape
        ((1, 2, 3), (1, 2, 3))
        >>> np.array_equal(x[0], data)
        True
        >>> np.allclose(y[0], poisson.pmf(data, model_parameters))
        True

        Stacked data on the same grid:

        >>> val1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> val2 = np.array([[7, 8, 9], [10, 11, 12]])
        >>> data = np.array([val1, val2])
        >>> model_parameters = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> x, y = build_poissonian_likelihood(data, model_parameters)
        >>> x.shape, y.shape
        ((2, 2, 3), (2, 2, 3))
        >>> np.array_equal(x[0], val1) and np.array_equal(x[1], val2)
        True
        >>> np.allclose(y[0], poisson.pmf(val1, model_parameters))
        True
        >>> np.allclose(y[1], poisson.pmf(val2, model_parameters))
        True

        Same result when supplying flattened data:

        >>> data_flat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> x2, y2 = build_poissonian_likelihood(data_flat, model_parameters)
        >>> np.array_equal(x2, x) and np.allclose(y2, y)
        True
    """
    values_to_reshape = np.asarray(data)
    parameters = np.asarray(model_parameters)

    if np.any(values_to_reshape < 0):
        raise ValueError("values of data must be non-negative.")
    if np.any(~np.isfinite(values_to_reshape)):
        raise ValueError("values of data must be finite.")
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

    probabilities = poisson.logpmf(counts, parameters) \
        if return_log \
        else poisson.pmf(counts, parameters)

    return counts, probabilities
