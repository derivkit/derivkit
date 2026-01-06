"""Provides the LikelihoodKit class.

Typical usage examples
----------------------

>>> import numpy as np
>>> from derivkit.likelihood_kit import LikelihoodKit
>>>
>>> # Gaussian example
>>> data = np.linspace(-5.0, 5.0, 200)
>>> theta = np.array([0.0])
>>> cov = np.array([[1.0]])
>>> lkit = LikelihoodKit(data=data, model_parameters=theta)
>>> grid, pdf = lkit.gaussian(cov=cov)
>>>
>>> # Poissonian example
>>> counts = np.array([1, 2, 3, 4])
>>> mu = np.array([0.5, 1.0, 1.5, 2.0])
>>> lkit = LikelihoodKit(data=counts, model_parameters=mu)
>>> reshaped_counts, pmf = lkit.poissonian()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.likelihoods.gaussian import build_gaussian_likelihood
from derivkit.likelihoods.poisson import build_poissonian_likelihood


class LikelihoodKit:
    """High-level interface for Gaussian and Poissonian likelihoods.

    The class stores ``data`` and ``model_parameters`` and provides
    methods to evaluate the corresponding likelihoods.
    """
    def __init__(
        self,
        data: ArrayLike,
        model_parameters: ArrayLike,
    ) -> None:
        """Initialises the likelihoods object.

        Args:
            data: Observed data values. The expected shape depends on the
                particular likelihoods. For the Gaussian likelihoods, ``data``
                is 1D or 2D, where axis 0 represents different samples and
                axis 1 the values. For the Poissonian likelihoods, ``data`` is
                reshaped to align with ``model_parameters``.
            model_parameters: Theoretical model values. For the Gaussian
                likelihoods, this is a 1D array of parameters used as the mean
                of the multivariate normal. For the Poissonian likelihoods,
                this is the expected counts (Poisson means).
        """
        self.data = np.asarray(data)
        self.model_parameters = np.asarray(model_parameters)

    def gaussian(
        self,
        cov: ArrayLike,
        *,
        return_log: bool = True,
    ) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
        """Evaluates a Gaussian likelihoods for the stored data and parameters.

        Args:
            cov: Covariance matrix. May be a scalar, a 1D vector of diagonal
                variances, or a full 2D covariance matrix. It will be
                symmetrized and normalized internally.
            return_log: If ``True``, return the log-likelihoods instead of
                the probability density function.

        Returns:
            A tuple ``(coordinate_grids, probabilities)`` where:

            * ``coordinate_grids`` is a tuple of 1D arrays giving the
              evaluation coordinates for each dimension.
            * ``probabilities`` is an array with the values of the
              multivariate Gaussian probability density (or log-density)
              evaluated on the Cartesian product of those coordinates.
        """
        return build_gaussian_likelihood(
            data=self.data,
            model_parameters=self.model_parameters,
            cov=cov,
            return_log=return_log,
        )

    def poissonian(
        self,
        *,
        return_log: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates a Poissonian likelihoods for the stored data and parameters.

        Args:
            return_log: If ``True``, return the log-likelihoods instead of
                the probability mass function.

        Returns:
            A tuple ``(counts, probabilities)`` where:

            * ``counts`` is the data reshaped to align with the
              model parameters.
            * ``probabilities`` is an array of Poisson probabilities
              (or log-probabilities) computed from ``counts`` and
              ``model_parameters``.
        """
        return build_poissonian_likelihood(
            data=self.data,
            model_parameters=self.model_parameters,
            return_log=return_log,
        )
