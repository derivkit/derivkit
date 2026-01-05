r"""Provides the ForecastKit class.

A light wrapper around the core forecasting utilities
(:func:`derivkit.forecasting.fisher.build_fisher_matrix`,
:func:`derivkit.forecasting.dali.build_dali`,
:func:`derivkit.forecasting.fisher.build_delta_nu`,
and :func:`derivkit.forecasting.fisher.build_fisher_bias`) that exposes a simple
API for Fisher and DALI tensors.

Typical usage example:

>>> import numpy as np
>>> from derivkit.forecast_kit import ForecastKit
>>>
>>> # Toy linear model: 2 params -> 2 observables
>>> def model(theta: np.ndarray) -> np.ndarray:
...     theta = np.asarray(theta, dtype=float)
...     return np.array([theta[0] + 2.0 * theta[1], 3.0 * theta[0] - theta[1]], dtype=float)
>>>
>>> theta0 = np.array([0.1, -0.2])
>>> cov = np.eye(2)
>>>
>>> fk = ForecastKit(function=model, theta0=theta0, cov=cov)
>>> fisher_matrix = fk.fisher(method="finite", n_workers=1)
>>> fisher_matrix.shape
(2, 2)
>>>
>>> data_unbiased = model(theta0)
>>> data_biased = data_unbiased + np.array([1e-3, -2e-3])
>>> dn = fk.delta_nu(data_biased=data_biased, data_unbiased=data_unbiased)
>>> dn.shape
(2,)
>>>
>>> bias_vec, delta_theta = fk.fisher_bias(
...     fisher_matrix=fisher_matrix,
...     delta_nu=dn,
...     method="finite",
...     n_workers=1,
... )
>>> bias_vec.shape, delta_theta.shape
((2,), (2,))
"""

from collections.abc import Callable
from typing import Any, Sequence

import numpy as np

from derivkit.forecasting.dali import build_dali
from derivkit.forecasting.fisher import (
    build_delta_nu,
    build_fisher_bias,
    build_fisher_matrix,
)
from derivkit.forecasting.fisher_gaussian import (
    build_gaussian_fisher_matrix,
)
from derivkit.utils.validate import (
    require_callable,
    resolve_covariance_input,
    validate_covariance_matrix_shape,
)


class ForecastKit:
    """Provides access to Fisher and DALI likelihood-expansion tensors."""

    def __init__(
            self,
            function: Callable[[Sequence[float] | np.ndarray], np.ndarray] | None,
            theta0: Sequence[float] | np.ndarray,
            cov: np.ndarray
                 | Callable[[np.ndarray], np.ndarray]
                 | tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    ):
        r"""Initialises the ForecastKit with model, fiducials, and covariance.

        Args:
            function: Callable returning the model mean vector :math:`\\mu(\\theta)`.
                May be ``None`` if you only plan to use covariance-only workflows
                (e.g. generalized Fisher with ``term="cov"``). Required for
                :meth:`fisher`, :meth:`dali`, and :meth:`fisher_bias`.
            theta0: Fiducial parameter values of shape ``(p,)`` where ``p`` is the
                number of parameters.
            cov: Covariance specification. Supported forms are:

                - ``cov=C0``: fixed covariance matrix :math:`C(\theta_0)` with shape
                  ``(n_obs, n_obs)``, where ``n_obs`` is the number of observables.
                - ``cov=cov_fn``: callable ``cov_fn(theta)`` returning the covariance
                  matrix :math:`C(\theta)` evaluated at the parameter vector ``theta``
                  (shape ``(n_obs, n_obs)``). The covariance at ``theta0`` is evaluated
                  once and cached.
                - ``cov=(C0, cov_fn)``: provide both a fixed covariance
                  ``C0 = C(theta0)`` and a callable ``cov_fn(theta) -> C(theta)``.
                  This avoids recomputing ``cov_fn(theta0)`` internally.
        """
        self.function = function
        self.theta0 = np.atleast_1d(np.asarray(theta0, dtype=np.float64))

        cov0, cov_fn = resolve_covariance_input(
            cov,
            theta0=self.theta0,
            validate=validate_covariance_matrix_shape,
        )

        self.cov0 = np.asarray(cov0, dtype=np.float64)
        self.cov_fn = cov_fn
        self.n_observables = int(self.cov0.shape[0])

    def fisher(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        """Computes the Fisher information matrix for a given model and covariance.

        Args:
            method: Derivative method name or alias (e.g., ``"adaptive"``,
                ``"finite"``). If ``None``, the
                :class:`derivkit.derivative_kit.DerivativeKit` default is used.
            n_workers: Number of workers for per-parameter parallelisation.
                Default is ``1`` (serial).
            **dk_kwargs: Additional keyword arguments forwarded to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            Fisher matrix with shape ``(n_parameters, n_parameters)``.
        """
        function = require_callable(self.function, context="ForecastKit.fisher")

        fisher_matrix = build_fisher_matrix(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )
        return fisher_matrix

    def fisher_bias(
        self,
        *,
        fisher_matrix: np.ndarray,
        delta_nu: np.ndarray,
        method: str | None = None,
        n_workers: int = 1,
        rcond: float = 1e-12,
        **dk_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Estimates parameter bias using the stored model, expansion point, and covariance.

        This function takes a model, an expansion point, a covariance matrix,
        a Fisher matrix, and a data-vector difference ``delta_nu`` and maps that
        difference into parameter space. A common use case is the classic
        "Fisher bias" setup, where one asks how a systematic-induced change in
        the data would shift inferred parameters.

        Internally, the function evaluates the model response at the expansion
        point and uses the covariance and Fisher matrix to compute both the
        parameter-space bias vector and the corresponding shifts. See
        https://arxiv.org/abs/0710.5171 for details.

        Args:
            fisher_matrix: Square matrix describing information about
                the parameters. Its shape must be ``(p, p)``, where ``p``
                is the number of parameters.
            delta_nu: Difference between a biased and an unbiased data vector,
                for example :math:`\\Delta\nu = \nu_{\\mathrm{with\\,sys}} - \nu_{\\mathrm{without\\,sys}}`.
                Accepts a 1D array of length n or a 2D array that will be
                flattened in row-major order ("C") to length n, where n is
                the number of observables. If supplied as a 1D array, it must
                already follow the same row-major ("C") flattening convention
                used throughout the package.
            n_workers: Number of workers used by the internal derivative routine
                when forming the Jacobian.
            method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
                If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit` default is used.
            rcond: Regularization cutoff for pseudoinverse.
                Default is ``1e-12``.
            **dk_kwargs: Additional keyword arguments passed to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            A tuple ``(bias_vec, delta_theta)`` of 1D arrays with length ``p``,
            where ``bias_vec`` is the parameter-space bias vector
            and ``delta_theta`` are the corresponding parameter shifts.

        Raises:
          ValueError: If input shapes are inconsistent with the stored model,
            covariance, or the Fisher matrix dimensions.
          FloatingPointError: If the difference vector contains ``NaN``.
        """
        function = require_callable(self.function, context="ForecastKit.fisher_bias")

        bias = build_fisher_bias(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            fisher_matrix=fisher_matrix,
            delta_nu=delta_nu,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            **dk_kwargs,
        )
        return bias

    def delta_nu(self,
                 data_unbiased: np.ndarray,
                 data_biased: np.ndarray,
                 ):
        """Computes the difference between two data vectors.

        This helper is used in Fisher-bias calculations and any other workflow
        where two data vectors are compared: it takes a pair of vectors (for
        example, a version with a systematic and one without) and returns their
        difference as a 1D array whose length matches the number of observables
        implied by ``cov``. It works with both 1D inputs and 2D arrays (for
        example, correlation-by-ell) and flattens 2D inputs using NumPy's
        row-major ("C") order, which is the standard convention throughout the
        DerivKit package.

        Args:
            data_unbiased: Reference data vector without the systematic.
                Can be 1D or 2D. If 1D, it must follow the NumPy's row-major
                ("C") flattening convention used throughout the package.
            data_biased: Data vector that includes the systematic effect.
                Can be 1D or 2D. If 1D, it must follow the NumPy's row-major
                ("C") flattening convention used throughout the package.

        Returns:
            A 1D NumPy array of length ``n_observables`` representing the
            mismatch between the two input data vectors. This is simply the
            element-wise difference between the input with systematic and the
            input without systematic, flattened if necessary to match the
            expected observable ordering.

        Raises:
          ValueError: If input shapes differ, inputs are not 1D/2D, or the
            flattened length does not match ``n_observables``.
          FloatingPointError: If non-finite values are detected in the result.
        """
        nu = build_delta_nu(
            cov=self.cov,
            data_biased=data_biased,
            data_unbiased=data_unbiased,
        )
        return nu

    def dali(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the doublet-DALI tensors (G, H) for the given model.

        Args:
            method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
                If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
                default is used.
            n_workers: Number of workers for per-parameter
                parallelization/threads. Default ``1`` (serial). Inner batch
                evaluation is kept serial to avoid oversubscription.
            dk_kwargs: Additional keyword arguments passed to
                :class:`derivkit.calculus_kit.CalculusKit`.

        Returns:
            A tuple ``(G, H)`` where ``G`` has shape ``(p, p, p)`` and ``H``
            has shape ``(p, p, p, p)``, with ``p`` being the number of model
            parameters.
        """
        function = require_callable(self.function, context="ForecastKit.dali")

        dali_tensors = build_dali(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
        )
        return dali_tensors

    def gaussian_fisher(
            self,
            *,
            term: str = "both",
            method: str | None = None,
            n_workers: int = 1,
            rcond: float = 1e-12,
            symmetrize_dcov: bool = True,
            **dk_kwargs: Any,
    ) -> np.ndarray:
        r"""Computes the generalized Fisher matrix for parameter-dependent mean and/or covariance.

        This function computes the generalized Fisher matrix for a Gaussian
        likelihood with parameter-dependent mean and/or covariance.
        Uses :func:`derivkit.forecasting.fisher_general.build_generalized_fisher_matrix`.

        Notes:
            ``function`` may be ``None`` if ``term="cov"``. For ``term="mean"`` or
            ``term="both"``, a mean model is required.

        Args:
            term: Which contribution(s) to return: ``"mean"``, ``"cov"``, or ``"both"``.
            method: Derivative method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            n_workers: Number of workers for per-parameter parallelisation.
            rcond: Regularization cutoff for pseudoinverse fallback in linear solves.
            symmetrize_dcov: If ``True``, symmetrize each covariance derivative via
                :math:`\\tfrac{1}{2}(C_{,i} + C_{,i}^{\\mathsf{T}})`.
            **dk_kwargs: Forwarded to the internal derivative calls.

        Returns:
            Fisher matrix with shape ``(p, p)``.
        """
        if self.cov_fn is None and term in ("cov", "both"):
            raise ValueError(
                "ForecastKit.generalized_fisher requires a parameter-dependent covariance callable "
                "for term='cov' or term='both'. Initialize ForecastKit with cov=cov_fn or cov=(cov0, cov_fn)."
            )

        cov_spec = (self.cov0, self.cov_fn) if self.cov_fn is not None else self.cov0

        return build_gaussian_fisher_matrix(
            theta0=self.theta0,
            cov=cov_spec,
            function=self.function,
            term=term,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            symmetrize_dcov=symmetrize_dcov,
            **dk_kwargs,
        )
