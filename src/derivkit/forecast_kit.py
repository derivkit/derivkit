r"""Provides the ForecastKit class.

A light wrapper around the core forecasting utilities
(:func:`derivkit.forecasting.fisher.build_fisher_matrix`,
:func:`derivkit.forecasting.dali.build_dali`,
:func:`derivkit.forecasting.dali_bias.build_dali_bias_tensor`,
:func:`derivkit.forecasting.fisher.build_delta_nu`,
and :func:`derivkit.forecasting.fisher.build_fisher_bias`) that exposes a simple
API for Fisher and DALI forecast tensors.

Typical usage example:

>>> import numpy as np
>>> from derivkit import ForecastKit
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
>>> dali = fk.dali(forecast_order=4, method="finite", n_workers=1)
>>> F = dali[1][0]  # fisher matrix
>>> D1, D2 = dali[2]  # dali doublet
>>> T1, T2, T3 = dali[3]  # dali triplet
>>> Qa1, Qa2, Qa3, Qa4 = dali[4]  # dali quadruplet
>>>
>>> data_unbiased = model(theta0)
>>> data_biased = data_unbiased + np.array([1e-3, -2e-3])
>>> dn = fk.delta_nu(data_unbiased=data_unbiased, data_biased=data_biased)
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

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Mapping, Sequence

import numpy as np

from derivkit.forecasting.dali import build_dali
from derivkit.forecasting.dali_bias import (
    build_dali_bias_tensor,
    build_dali_map_shift,
    build_dali_mean_response_shift,
    build_dali_mean_shift,
)
from derivkit.forecasting.expansions import (
    build_delta_chi2_dali,
    build_delta_chi2_fisher,
    build_logposterior_dali,
    build_logposterior_fisher,
)
from derivkit.forecasting.fisher import (
    build_delta_nu,
    build_fisher_bias,
    build_fisher_matrix,
)
from derivkit.forecasting.fisher_gaussian import (
    build_gaussian_fisher_matrix,
)
from derivkit.forecasting.fisher_xy import (
    build_xy_gaussian_fisher_matrix,
)
from derivkit.forecasting.getdist_dali_samples import (
    dali_to_getdist_emcee,
    dali_to_getdist_importance,
)
from derivkit.forecasting.getdist_fisher_samples import (
    fisher_to_getdist_gaussiannd,
    fisher_to_getdist_samples,
)
from derivkit.forecasting.laplace import (
    build_laplace_approximation,
    build_laplace_covariance,
    build_laplace_hessian,
    build_negative_logposterior,
)
from derivkit.utils.caching import wrap_input_cache
from derivkit.utils.linalg import split_xy_covariance
from derivkit.utils.thread_safety import wrap_with_lock
from derivkit.utils.types import (
    ArrayLike1D,
    ArrayLike2D,
    FloatArray,
)
from derivkit.utils.validate import (
    require_callable,
    resolve_covariance_input,
    validate_covariance_matrix_shape,
)

_RESERVED_KWARGS = {"theta0"}


class ForecastKit:
    """Provides access to forecasting workflows."""

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], np.ndarray] | None,
        theta0: Sequence[float] | np.ndarray,
        cov: np.ndarray | Callable[[np.ndarray], np.ndarray],
        *,
        use_input_cache: bool = True,
        cache_number_decimal_places: int | None = None,
        cache_maxsize: int | None = 4096,
        cache_copy: bool = True,
        thread_safe: bool = False,
        thread_lock: threading.RLock | None = None,
    ):
        r"""Initialises the ForecastKit with model, fiducials, and covariance.

        As an optimization the class can cache function values, which speeds
        up repeated function calls and their associated derivatives. In this
        case the function will truncate the number of decimals of the input
        parameters.

        Args:
            function: Callable returning the model mean vector :math:`\mu(\theta)`.
                May be ``None`` if you only plan to use covariance-only workflows
                (e.g. generalized Fisher with ``term="cov"``). Required for
                :meth:`fisher`, :meth:`dali`, and :meth:`fisher_bias`.
            theta0: Fiducial parameter values of shape ``(p, )`` where ``p`` is the
                number of parameters.
            cov: Covariance specification. Supported forms are:

                - ``cov=C0``: fixed covariance matrix :math:`C(\theta_0)` with shape
                  ``(n_obs, n_obs)``, where ``n_obs`` is the number of observables.
                - ``cov=cov_fn``: callable with ``cov_fn(theta)`` returning the covariance
                  matrix :math:`C(\theta)` evaluated at the parameter vector ``theta``,
                  with shape ``(n_obs, n_obs)``. The covariance at ``theta0`` is evaluated
                  once and cached.

            use_input_cache: If ``True``, wrap the input function with an
                input-based LRU cache. Exact input values are cached by default.
            cache_number_decimal_places: Decimal places used when constructing
                cache keys. If ``None``, exact input values are used.
            cache_maxsize: Maximum size of the input cache.
            cache_copy: If ``True``, return copies of cached NumPy array outputs.
            thread_safe: If ``True``, serialize calls to ``function`` using a lock.
                This prevents concurrent evaluation within a single Python process
                (i.e. thread-based parallelism). It does not synchronize across
                multiple processes.
            thread_lock: Optional lock object to use when ``thread_safe=True``.
                If ``None``, an internal lock is created.
        """
        self.function = function

        if self.function is not None and use_input_cache:
            self.function = wrap_input_cache(
                self.function,
                number_decimal_places=cache_number_decimal_places,
                maxsize=cache_maxsize,
                copy=cache_copy,
            )

        if self.function is not None and thread_safe:
            self.function = wrap_with_lock(self.function, lock=thread_lock)

        self.theta0 = np.atleast_1d(np.asarray(theta0, dtype=np.float64))

        cov0, cov_fn = resolve_covariance_input(
            cov,
            theta0=self.theta0,
            validate=validate_covariance_matrix_shape,
        )

        self.cov0 = cov0
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
                for example :math:`\Delta\nu = \nu_{\mathrm{biased}} - \nu_{\mathrm{unbiased}}`.
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

    def delta_nu(
        self,
        data_unbiased: np.ndarray,
        data_biased: np.ndarray,
    ) -> np.ndarray:
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
        """
        nu = build_delta_nu(
            cov=self.cov0,
            data_biased=data_biased,
            data_unbiased=data_unbiased,
        )
        return nu

    def dali(
        self,
        *,
        method: str | None = None,
        forecast_order: int = 2,
        symmetrize: bool = True,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> dict[int, tuple[FloatArray, ...]]:
        """Builds the DALI expansion for the given model up to the given order.

        Args:
            method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``).
                If ``None``, the :class:`derivkit.derivative_kit.DerivativeKit`
                default is used.
            forecast_order: The requested order of the forecast.
                Currently supported values and their meaning are given in
                :data:`derivkit.forecasting.forecast_core.SUPPORTED_FORECAST_ORDERS`.
            symmetrize: Flag to force symmetrization across all DALI tensor axes.
            n_workers: Number of workers for per-parameter
                parallelization/threads. Default ``1`` (serial). Inner batch
                evaluation is kept serial to avoid oversubscription.
            **dk_kwargs: Additional keyword arguments passed to
                :class:`derivkit.calculus_kit.CalculusKit`.

        Returns:
            A dict mapping ``order -> multiplet`` for all ``order = 1..forecast_order``.

            For each forecast order k, the returned multiplet contains the tensors
            introduced at that order. Concretely:

            - order 1: ``(F_{(1,1)},)`` (Fisher matrix)
            - order 2: ``(D_{(2,1)}, D_{(2,2)})``
            - order 3: ``(T_{(3,1)}, T_{(3,2)}, T_{(3,3)})``
            - order 4: ``(Qa_{(4,1)}, Qa_{(4,2)}, Qa_{(4,3)}, Qa_{(4,4)})``

            Here the tensors at order ``k`` are contractions of the
            ``k``-th order model derivative with derivatives of orders
            ``1`` through ``k`` via the inverse covariance.

            Each tensor axis has length ``p = len(self.theta0)``. The
            additional tensors at
            order ``k`` have parameter-axis ranks from ``k+1`` through ``2*k``.
        """
        function = require_callable(self.function, context="ForecastKit.dali")

        dali_tensors = build_dali(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            method=method,
            forecast_order=forecast_order,
            symmetrize_dali=symmetrize,
            n_workers=n_workers,
            **dk_kwargs,
        )
        return dali_tensors

    def gaussian_fisher(
        self,
        *,
        method: str | None = None,
        n_workers: int = 1,
        rcond: float = 1e-12,
        symmetrize_dcov: bool = True,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        r"""Computes the generalized Fisher matrix for parameter-dependent mean and/or covariance.

        This function computes the generalized Fisher matrix for a Gaussian
        likelihood with parameter-dependent mean and/or covariance.
        Uses :func:`derivkit.forecasting.fisher_gaussian.build_gaussian_fisher_matrix`.

        Args:
            method: Derivative method name or alias (e.g., ``"adaptive"``, ``"finite"``).
            n_workers: Number of workers for per-parameter parallelisation.
            rcond: Regularization cutoff for pseudoinverse fallback in linear solves.
            symmetrize_dcov: If ``True``, symmetrize each covariance derivative via
                :math:`\tfrac{1}{2}(C_{,i} + C_{,i}^{\mathsf{T}})`.
            **dk_kwargs: Forwarded to the internal derivative calls.

        Returns:
            Fisher matrix with shape ``(p, p)`` with ``p`` the number of parameters.
        """
        cov_spec = self.cov_fn if self.cov_fn is not None else self.cov0

        return build_gaussian_fisher_matrix(
            theta0=self.theta0,
            cov=cov_spec,
            function=self.function,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            symmetrize_dcov=symmetrize_dcov,
            **dk_kwargs,
        )

    def xy_fisher(
        self,
        *,
        x0: np.ndarray,
        mu_xy: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cov_xy: np.ndarray,
        method: str | None = None,
        n_workers: int = 1,
        rcond: float = 1e-12,
        symmetrize_dcov: bool = True,
        check_cyy_consistency: bool = True,
        atol: float = 0.0,
        rtol: float = 1e-12,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        r"""Computes the X–Y Gaussian Fisher matrix (noisy inputs and outputs).

        This implements the X–Y Gaussian Fisher formalism where the measured inputs
        and outputs are both noisy and may be correlated. Input uncertainty is
        propagated into an effective output covariance via a local linearization of
        ``mu_xy(x, theta)`` with respect to ``x`` at the measured inputs ``x0``.

        Args:
            x0: Measured inputs with shape ``(n_x,)``.
            mu_xy: Model mean callable ``mu_xy(x, theta) -> y``.
            cov_xy: Full covariance for the stacked vector ``[x, y]`` with shape
                ``(n_x + n_y, n_x + n_y)``. Note that this is the joint covariance of the input
                and output measurements, ordered as ``[x0, y0]`` (inputs first, then outputs),
                so it contains the blocks ``Cxx``, ``Cxy``, and ``Cyy``.
            method: Derivative method name/alias forwarded to the derivative backend.
            n_workers: Number of workers for derivative evaluations.
            rcond: Regularization cutoff used in linear solves / pseudoinverse fallback.
            symmetrize_dcov: If ``True``, symmetrize each covariance derivative.
            check_cyy_consistency: If ``True`` and this :class:`ForecastKit` was
                initialized with a fixed output covariance ``cov=self.cov0`` (i.e.
                :math:`C_{yy}`), verify that it matches the :math:`C_{yy}` block implied
                by ``cov_xy``.
            atol: Absolute tolerance for the :math:`C_{yy}` consistency check.
            rtol: Relative tolerance for the :math:`C_{yy}` consistency check.
            **dk_kwargs: Forwarded to the derivative backend.

        Returns:
            Fisher matrix with shape ``(p, p)``, where ``p`` is the number of parameters.

        Raises:
            ValueError: If ``check_cyy_consistency=True`` and ``self.cov0`` is compatible
                with a :math:`C_{yy}` block but differs from the :math:`C_{yy}` block
                extracted from ``cov_xy`` beyond ``rtol``/``atol``.
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        cov_xy_arr = np.asarray(cov_xy, dtype=np.float64)

        if check_cyy_consistency and getattr(self, "cov0", None) is not None:
            nx = int(x0.size)
            _, _, cyy_from_xy = split_xy_covariance(cov_xy_arr, nx=nx)

            # Only check if self.cov0 looks like a Cyy (ny, ny) matrix.
            if self.cov0.shape == cyy_from_xy.shape:
                if not np.allclose(self.cov0, cyy_from_xy, rtol=rtol,
                                   atol=atol):
                    raise ValueError(
                        "Inconsistent covariances: ForecastKit was initialized with cov=Cyy, "
                        "but the Cyy block extracted from cov_xy differs. "
                        "Either initialize ForecastKit with the same Cyy, or set "
                        "check_cyy_consistency=False if you intentionally want them different."
                    )

        return build_xy_gaussian_fisher_matrix(
            theta0=self.theta0,
            x0=x0,
            mu_xy=mu_xy,
            cov=cov_xy_arr,
            method=method,
            n_workers=n_workers,
            rcond=rcond,
            symmetrize_dcov=symmetrize_dcov,
            **dk_kwargs,
        )

    def delta_chi2_fisher(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
    ) -> float:
        """Computes a displacement chi-squared under the Fisher approximation.

        This evaluates the standard quadratic form

            ``delta_chi2 = (theta - theta0)^T @ F @ (theta - theta0)``

        using the provided Fisher matrix and the stored expansion point :attr:`ForecastKit.theta0`.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            fisher: Fisher matrix with shape ``(p, p)``.

        Returns:
            Scalar delta chi-squared value.
        """
        return build_delta_chi2_fisher(theta=theta, theta0=self.theta0, fisher=fisher)

    def delta_chi2_dali(
        self,
        *,
        theta: np.ndarray,
        dali: dict[int, tuple[np.ndarray, ...]],
        forecast_order: int | None = 2,
    ) -> float:
        """Computes a displacement chi-squared under the DALI approximation.

        This evaluates a scalar ``delta_chi2`` from the displacement
        ``d = theta - theta0`` using the provided
        DALI tensors.

        The expansion point is taken from :attr:`ForecastKit.theta0`.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            dali: DALI tensors as returned by :meth:`ForecastKit.dali`.
            forecast_order: Maximum order of the DALI expansion to include.
                If ``None``, the highest order available in ``dali`` is used.

        Returns:
            Scalar delta chi-squared value.
        """
        return build_delta_chi2_dali(
            theta=theta,
            theta0=self.theta0,
            dali=dali,
            forecast_order=forecast_order,
        )

    def logposterior_fisher(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
        bias: np.ndarray[np.floating] | None = None,
        prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
        prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
        logprior: Callable[[np.ndarray], float] | None = None,
    ) -> float:
        """Computes the log posterior under the Fisher approximation.

        The returned value is defined up to an additive constant in log space.
        This corresponds to an overall multiplicative normalization of the posterior
        density in probability space.

        If no prior is provided, this returns the Fisher log-likelihood expansion
        with a flat prior and no hard cutoffs. Priors may be provided either as a
        pre-built ``logprior(theta)`` callable or as a lightweight prior specification
        via ``prior_terms`` and/or ``prior_bounds``.

        The expansion point is taken from the stored ``self.theta0``.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            fisher: Fisher matrix with shape ``(p, p)``.
            bias: Optional linear systematic-bias vector of shape ``(p,)``.
                If provided, adds ``bias @ (theta - theta0)`` to the
                Fisher log posterior.
            prior_terms: Prior term specification passed to the underlying prior
                builder. Use this only if ``logprior`` is not provided.
            prior_bounds: Global hard bounds passed to the underlying prior builder.
                Use this only if ``logprior`` is not provided.
            logprior: Optional custom log-prior callable. If it returns a non-finite
                value, the posterior is treated as zero at that point and the function
                returns ``-np.inf``. Cannot be used together with ``prior_terms`` or
                ``prior_bounds``.

        Returns:
            Scalar log posterior value, defined up to an additive constant.
        """
        return build_logposterior_fisher(
            theta=theta,
            theta0=self.theta0,
            fisher=fisher,
            bias=bias,
            prior_terms=prior_terms,
            prior_bounds=prior_bounds,
            logprior=logprior,
        )

    def logposterior_dali(
        self,
        *,
        theta: np.ndarray,
        dali: dict[int, tuple[np.ndarray, ...]],
        bias: dict[int, np.ndarray[np.floating]] | None = None,
        forecast_order: int | None = 2,
        prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
        prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
        logprior: Callable[[np.ndarray], float] | None = None,
    ) -> float:
        """Computes the log posterior (up to a constant) under the DALI approximation.

        If no prior is provided, this returns the DALI log-likelihood expansion with
        a flat prior and no hard cutoffs. Priors may be provided either as a pre-built
        ``logprior(theta)`` callable or as a lightweight prior specification via
        ``prior_terms`` and/or ``prior_bounds``.

        The expansion point is taken from the stored ``self.theta0``.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            dali: DALI tensors as returned by :meth:`ForecastKit.dali`.
            bias: Optional systematic-bias tensors mapping expansion order
                to parameter-space tensors. Adds the factorial-weighted
                mismatch expansion through ``forecast_order``.
                If ``None``, evaluates the unbiased DALI posterior.
            forecast_order: Maximum order of the DALI expansion to include.
                If ``None``, the highest order available in ``dali`` is used.
            prior_terms: Prior term specification passed to the underlying prior
                builder. Use this only if ``logprior`` is not provided.
            prior_bounds: Global hard bounds passed to the underlying prior builder.
                Use this only if ``logprior`` is not provided.
            logprior: Optional custom log-prior callable. If it returns a non-finite
                value, the posterior is treated as zero at that point and the function
                returns ``-np.inf``. Cannot be used together with ``prior_terms`` or
                ``prior_bounds``.

        Returns:
            Scalar log posterior value, defined up to an additive constant.
        """
        return build_logposterior_dali(
            theta=theta,
            theta0=self.theta0,
            dali=dali,
            bias=bias,
            forecast_order=forecast_order,
            prior_terms=prior_terms,
            prior_bounds=prior_bounds,
            logprior=logprior,
        )

    def negative_logposterior(
        self,
        theta: Sequence[float] | np.ndarray,
        *,
        logposterior: Callable[[np.ndarray], float],
    ) -> float:
        """Computes the negative log-posterior at ``theta``.

        This converts a log-posterior callable into the objective used by MAP
        estimation and curvature-based methods. It simply returns
        ``-logposterior(theta)`` and validates that the result is finite.

        Args:
            theta: 1D array-like parameter vector.
            logposterior: Callable that accepts a 1D float64 array and returns a scalar float.

        Returns:
            Negative log-posterior value as a float.
        """
        return build_negative_logposterior(theta, logposterior=logposterior)

    def laplace_hessian(
        self,
        *,
        neg_logposterior: Callable[[np.ndarray], float],
        theta_map: Sequence[float] | np.ndarray | None = None,
        method: str | None = None,
        n_workers: int = 1,
        **dk_kwargs: Any,
    ) -> np.ndarray:
        """Computes the Hessian of the negative log-posterior at ``theta_map``.

        The Hessian at ``theta_map`` measures the local curvature of the posterior peak.
        In the Laplace approximation, this Hessian plays the role of a local precision
        matrix, and its inverse provides a fast Gaussian estimate of parameter
        uncertainties and correlations.

        If ``theta_map`` is not provided, this uses the stored expansion point ``self.theta0``.

        Args:
            neg_logposterior: Callable returning the scalar negative log-posterior.
            theta_map: Point where the curvature is evaluated (typically the MAP).
                If ``None``, uses ``self.theta0``.
            method: Derivative method name/alias forwarded to the calculus machinery.
            n_workers: Outer parallelism forwarded to Hessian construction.
            **dk_kwargs: Additional keyword arguments forwarded to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            A symmetric 2D array with shape ``(p, p)`` giving the Hessian of
            ``neg_logposterior`` evaluated at ``theta_map``.
        """
        theta = self.theta0 if theta_map is None else theta_map
        return build_laplace_hessian(
            neg_logposterior=neg_logposterior,
            theta_map=theta,
            method=method,
            n_workers=n_workers,
            dk_kwargs=dk_kwargs,
        )

    def laplace_covariance(
        self,
        hessian: np.ndarray,
        *,
        rcond: float = 1e-12,
    ) -> np.ndarray:
        """Computes the Laplace covariance matrix from a Hessian.

        In the Laplace (Gaussian) approximation, the Hessian of the negative
        log-posterior at the expansion point acts like a local precision matrix.
        The approximate posterior covariance is the matrix inverse of that Hessian.

        Args:
            hessian: 2D square Hessian matrix.
            rcond: Cutoff for small singular values used by the pseudoinverse fallback.

        Returns:
            A 2D symmetric covariance matrix with the same shape as ``hessian``.
        """
        return build_laplace_covariance(hessian, rcond=rcond)

    def laplace_approximation(
        self,
        *,
        neg_logposterior: Callable[[np.ndarray], float],
        theta_map: Sequence[float] | np.ndarray | None = None,
        method: str | None = None,
        n_workers: int = 1,
        ensure_spd: bool = True,
        rcond: float = 1e-12,
        **dk_kwargs: Any,
    ) -> dict[str, Any]:
        """Computes a Laplace (Gaussian) approximation around ``theta_map``.

        The Laplace approximation replaces the posterior near its peak with a Gaussian.
        It does this by measuring the local curvature of the negative log-posterior
        using its Hessian at ``theta_map``. The Hessian acts like a local precision
        matrix, and its inverse is the approximate covariance.

        If ``theta_map`` is not provided, this uses the stored expansion point ``self.theta0``.

        Args:
            neg_logposterior: Callable that accepts a 1D float64 parameter vector and
                returns a scalar negative log-posterior value.
            theta_map: Expansion point for the approximation. This is often the maximum a
                posteriori estimate (MAP). If ``None``, uses ``self.theta0``.
            method: Derivative method name/alias forwarded to the Hessian builder.
            n_workers: Outer parallelism forwarded to Hessian construction.
            ensure_spd: If ``True``, attempt to regularize the Hessian to be symmetric positive definite
                (SPD) by adding diagonal jitter.
            rcond: Cutoff for small singular values used by the pseudoinverse fallback
                when computing the covariance.
            **dk_kwargs: Additional keyword arguments forwarded to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            Dictionary with the Laplace approximation outputs (theta_map, neg_logposterior_at_map,
            hessian, cov, and jitter).
        """
        theta = self.theta0 if theta_map is None else theta_map
        return build_laplace_approximation(
            neg_logposterior=neg_logposterior,
            theta_map=theta,
            method=method,
            n_workers=n_workers,
            ensure_spd=ensure_spd,
            rcond=rcond,
            dk_kwargs=dk_kwargs,
        )

    def dali_bias_tensor(
        self,
        *,
        delta_nu: ArrayLike1D | ArrayLike2D,
        bias_order: int = 3,
        method: str | None = None,
        n_workers: int = 1,
        derivatives: dict[int, FloatArray] | None = None,
        **dk_kwargs: Any,
    ) -> dict[int, FloatArray]:
        """Builds the systematic mismatch tensors entering the DALI bias expansion.

        The tensors describe how the data-model mismatch ``delta_nu``
        couples to successive derivatives of the model.

        Args:
            delta_nu: Difference between two data vectors, which may represent
                a systematic mismatch or a difference between model predictions.
                Accepts a 1D array or a column vector of shape ``(n_observables, 1)``.
                The input is flattened internally.
            bias_order: Highest mismatch tensor order to compute. Supported values
                are given in
                :data:`derivkit.forecasting.forecast_core.SUPPORTED_DERIVATIVE_ORDERS`.
            method: Numerical differentiation method or alias (e.g.,
                ``"adaptive"``, ``"finite"``). If ``None``, uses
                :attr:`derivkit.derivative_kit.DerivativeKit.default_method`.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default ``1`` (serial).
            derivatives: Optional dictionary of model derivatives keyed by order.
                Existing derivatives are reused. Missing orders are computed and
                added to the supplied dictionary in place, modifying the input.
                If ``None``, an internal dictionary is used.
            **dk_kwargs: Additional keyword arguments passed to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            A dictionary mapping derivative orders 1 up to and including
            ``bias_order`` to their corresponding mismatch tensors.
            Zeroth-order terms are omitted because they are independent of
            the parameter displacement.

        Raises:
            TypeError: If ``bias_order`` is not an integer.
            ValueError: If ``bias_order`` is unsupported, ``theta0`` is empty, or
                the data mismatch does not match the number of observables.
        """
        function = require_callable(self.function, context="ForecastKit.dali_bias")

        return build_dali_bias_tensor(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            delta_nu=delta_nu,
            bias_order=bias_order,
            method=method,
            n_workers=n_workers,
            derivatives=derivatives,
            **dk_kwargs
        )

    def dali_map_shift(
        self,
        *,
        delta_nu: ArrayLike1D | ArrayLike2D,
        expansion_order: int = 3,
        return_metadata: bool = False,
        method: str | None = None,
        n_workers: int = 1,
        derivatives: dict[int, FloatArray] | None = None,
        **dk_kwargs: Any,
    ):
        """Predict how the posterior maximum moves when a systematic is introduced.

        Args:
            delta_nu: Actual difference between the systematically affected data
                vector and the fiducial model prediction. Its size is already
                included; no separate amplitude parameter is needed.
            expansion_order: Number of perturbative MAP response orders (1 to 3).
            return_metadata: If set to ``True``, return the shift and a dictionary of
                explanatory results; otherwise return only the shift.
            method: Numerical differentiation method or alias (e.g.,
                ``"adaptive"``, ``"finite"``). If ``None``, uses
                :attr:`derivkit.derivative_kit.DerivativeKit.default_method`.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default ``1`` (serial).
            derivatives: Optional dictionary of model derivatives keyed by order.
                Existing derivatives are reused. Missing orders are computed and
                added to the supplied dictionary in place, modifying the input.
                If ``None``, an internal dictionary is used.
            **dk_kwargs: Additional keyword arguments passed to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            If ``return_metadata=False``, an array giving the MAP systematic
            shift: MAP with the mismatch minus MAP without the mismatch.
            Since the baseline MAP is assumed to be ``theta0``, this is also
            the MAP displacement from the fiducial parameters.

            If ``return_metadata=True``, returns ``(shift, metadata)``.
            The metadata dictionary contains:

            ``estimator``: Predicted MAP parameter values with the mismatch.
                This is ``theta0 + shift``.

            ``baseline``: MAP without the mismatch, assumed to be ``theta0``.

            ``displacement``: MAP with the mismatch minus ``theta0``.
                This equals ``shift`` because the baseline MAP is assumed
                to be ``theta0``.

            ``terms``: Dictionary keyed by response order. Entry 1 is the
                leading Fisher bias; entries 2 and 3 are the successive
                corrections. Their sum is the returned shift.

            ``gaussian_terms``: Terms obtained if the unbiased posterior is
                approximated by its quadratic (Gaussian) Fisher form, while
                retaining the mismatch potential up to and including the
                requested order.

            ``non_gaussian_terms``: Differences between the DALI and Gaussian
                terms at each order. They isolate corrections caused by the
                non-quadratic reference posterior within this expansion.

            ``gaussian_contribution``: Sum of ``gaussian_terms``.

            ``non_gaussian_contribution``: Sum of ``non_gaussian_terms``.
                Adding the two contributions reproduces the MAP shift.

            These Gaussian/non-Gaussian labels refer to the reference posterior,
            not to whether a term is linear or nonlinear in systematic amplitude.
        """
        function = require_callable(self.function, context="ForecastKit.dali_bias")

        return build_dali_map_shift(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            delta_nu=delta_nu,
            expansion_order=expansion_order,
            return_metadata=return_metadata,
            method=method,
            n_workers=n_workers,
            derivatives=derivatives,
            **dk_kwargs,
        )

    def dali_mean_shift(
        self,
        *,
        delta_nu: ArrayLike1D | ArrayLike2D,
        return_metadata: bool = False,
        method: str | None = None,
        n_workers: int = 1,
        derivatives: dict[int, FloatArray] | None = None,
        **dk_kwargs: Any,
    ):
        """Predict the systematic shift of the posterior mean analytically.

        Uses third-order DALI tensors to calculate the perturbative MAP shift
        and the change in the posterior mean relative to the MAP.

        Args:
            delta_nu: Difference between the systematically affected data
                vector and the fiducial model prediction, including its magnitude.
            return_metadata: If ``True``, return the shift and explanatory metadata.
            method: Numerical differentiation method or alias (e.g.,
                ``"adaptive"``, ``"finite"``). If ``None``, uses
                :attr:`derivkit.derivative_kit.DerivativeKit.default_method`.
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default ``1`` (serial).
            derivatives: Optional dictionary of model derivatives keyed by order.
                Existing derivatives are reused. Missing orders are computed and
                added to the supplied dictionary in place, modifying the input.
                If ``None``, an internal dictionary is used.
            **dk_kwargs: Additional keyword arguments passed to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            An array containing the predicted systematic shift of the posterior
            mean, defined as the posterior mean with the mismatch minus the
            posterior mean without it.

            If ``return_metadata=True``, returns ``(shift, metadata)`` instead.
            ``metadata`` is a dictionary containing:

            ``estimator``: Predicted posterior mean with the mismatch.

            ``baseline``: Predicted posterior mean without the mismatch.
                This need not equal ``theta0``.

            ``displacement``: Predicted posterior mean with the mismatch minus
                the fiducial parameters ``theta0``.

            ``map_shift``: Predicted systematic shift of the posterior MAP.
        """
        function = require_callable(self.function, context="ForecastKit.dali_bias")

        return build_dali_mean_shift(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            delta_nu=delta_nu,
            return_metadata=return_metadata,
            method=method,
            n_workers=n_workers,
            derivatives=derivatives,
            **dk_kwargs,
        )

    def dali_mean_response(
        self,
        *,
        delta_nu: ArrayLike1D | ArrayLike2D,
        theta: FloatArray,
        forecast_order: int = 3,
        response_order: int = 4,
        return_metadata: bool = False,
        method: str | None = None,
        n_workers: int = 1,
        derivatives: dict[int, FloatArray] | None = None,
        **dk_kwargs: Any,
    ):
        """Predict the posterior mean shift from its systematic response.

        Expands the mean response around an unbiased reference posterior
        represented by samples or integration nodes.

        Args:
            delta_nu: Full systematic difference in the data vector.
            theta: Samples or integration nodes representing the unbiased
                reference posterior, with shape ``(n_samples, n_parameters)``.
            forecast_order: Highest model derivative order included in the
                systematic mismatch potential. Supported values are given in
                :data:`derivkit.forecasting.forecast_core.SUPPORTED_FORECAST_ORDERS`.
            response_order: Highest order retained in the systematic amplitude
                expansion of the posterior mean. Must be 1, 2, 3, or 4.
            return_metadata: If ``True``, return the shift and explanatory metadata.
            method: Numerical differentiation method or alias (e.g.,
                ``"adaptive"``, ``"finite"``). If ``None``, uses
                :attr:`derivkit.derivative_kit.DerivativeKit.default_method`
            n_workers: Number of workers for per-parameter parallelization/threads.
                Default ``1`` (serial).
            derivatives: Optional dictionary of model derivatives keyed by order.
                Existing derivatives are reused. Missing orders are computed and
                added to the supplied dictionary in place, modifying the input.
                If ``None``, an internal dictionary is used.
            **dk_kwargs: Additional keyword arguments passed to
                :meth:`derivkit.derivative_kit.DerivativeKit.differentiate`.

        Returns:
            An array containing the predicted systematic shift of the posterior
            mean, calculated by summing the response terms through
            ``response_order``. The shift is the posterior mean with the mismatch
            minus the posterior mean without it.

            If ``return_metadata=True``, returns ``(shift, metadata)`` instead.
            The metadata dictionary contains:

            ``estimator``: Predicted posterior mean with the mismatch.
                This is the reference posterior mean plus the predicted shift.

            ``baseline``: Posterior mean without the mismatch, calculated from
                the supplied samples.

            ``displacement``: Predicted posterior mean with the mismatch minus
                the fiducial parameters ``theta0``.

            ``terms``: Dictionary mapping each response order to its contribution
                to the predicted posterior mean shift for the supplied mismatch.
                Their sum equals the returned shift.

        Raises:
            TypeError: If ``forecast_order`` or ``response_order`` is not an integer.
            ValueError: If either order is unsupported, or if the supplied
                reference samples have invalid shapes.
        """
        function = require_callable(self.function, context="ForecastKit.dali_bias")

        return build_dali_mean_response_shift(
            function=function,
            theta0=self.theta0,
            cov=self.cov0,
            delta_nu=delta_nu,
            theta=theta,
            forecast_order=forecast_order,
            response_order=response_order,
            return_metadata=return_metadata,
            method=method,
            n_workers=n_workers,
            derivatives=derivatives,
            **dk_kwargs,
        )

    def getdist_fisher_gaussian(
        self,
        *,
        fisher: np.ndarray,
        bias: np.ndarray | None = None,
        names: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """Converts a Fisher Gaussian into a GetDist :class:`getdist.gaussian_mixtures.GaussianND`.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_fisher_samples.fisher_to_getdist_gaussiannd`
        that fixes the mean to the stored expansion point ``self.theta0``.

        Args:
            fisher: Fisher matrix with shape ``(p, p)`` evaluated at ``self.theta0``.
            bias: Optional Fisher systematic bias vector of shape ``(p,)``.
                Shifts the Gaussian mean by ``covariance @ bias``.
            names: Optional parameter names (length ``p``).
            labels: Optional parameter labels (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_fisher_samples.fisher_to_getdist_gaussiannd`
                (e.g. ``label``, ``rcond``).

        Returns:
            A :class:`getdist.gaussian_mixtures.GaussianND` with mean ``self.theta0`` and
            covariance given by the (pseudo-)inverse Fisher matrix.
        """
        return fisher_to_getdist_gaussiannd(
            self.theta0,
            fisher,
            bias=bias,
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_fisher_samples(
        self,
        *,
        fisher: np.ndarray,
        bias: np.ndarray | None = None,
        names: Sequence[str],
        labels: Sequence[str],
        **kwargs: Any,
    ):
        """Draws GetDist :class:`getdist.MCSamples` from the Fisher Gaussian at ``self.theta0``.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_fisher_samples.fisher_to_getdist_samples`
        that fixes the sampling center to the stored expansion point ``self.theta0``.

        Args:
            fisher: Fisher matrix with shape ``(p, p)`` evaluated at ``self.theta0``.
            bias: Optional Fisher systematic bias vector of shape ``(p,)``.
                Shifts the Gaussian sampling mean by ``pinv(fisher) @ bias``
                and contributes to the stored negative log posterior.
            names: Parameter names for GetDist (length ``p``).
            labels: Parameter labels for GetDist (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_fisher_samples.fisher_to_getdist_samples`
                (e.g. ``n_samples``, ``seed``, ``kernel_scale``, ``prior_terms``,
                ``prior_bounds``, ``logprior``, ``hard_bounds``, ``store_loglikes``, ``label``).

        Returns:
            A :class:`getdist.MCSamples` object containing samples drawn from the Fisher Gaussian.
        """
        return fisher_to_getdist_samples(
            self.theta0,
            fisher,
            bias=bias,
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_dali_importance(
        self,
        *,
        dali: dict[int, tuple[np.ndarray, ...]],
        bias: dict[int, np.ndarray[np.floating]] | None = None,
        names: Sequence[str],
        labels: Sequence[str],
        **kwargs: Any,
    ):
        """Returns GetDist :class:`getdist.MCSamples` for a DALI posterior via importance sampling.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_importance`
        that fixes the expansion point to ``self.theta0``.

        Args:
            dali: DALI tensors as returned by :meth:`ForecastKit.dali`.
            bias: Optional dictionary of systematic bias tensors indexed by expansion
                order, as returned by :func:`derivkit.forecasting.build_dali_bias_tensor`.
                These contribute to the target log posterior.
            names: Parameter names for GetDist (length ``p``).
            labels: Parameter labels for GetDist (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_importance`
                (e.g. ``forecast_order``, ``n_samples``, ``kernel_scale``, ``seed``,
                ``prior_terms``, ``prior_bounds``, ``logprior``,
                ``sampler_bounds``, ``label``).

        Returns:
            A :class:`getdist.MCSamples` with importance weights.
        """
        kwargs = _drop_reserved_kwargs(kwargs, reserved=_RESERVED_KWARGS)

        return dali_to_getdist_importance(
            theta0=self.theta0,
            dali=dali,
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_dali_emcee(
        self,
        *,
        dali: dict[int, tuple[np.ndarray, ...]],
        bias: dict[int, np.ndarray[np.floating]] | None = None,
        names: Sequence[str],
        labels: Sequence[str],
        **kwargs: Any,
    ):
        """Returns GetDist :class:`getdist.MCSamples` from ``emcee`` sampling of a DALI posterior.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_emcee`
        that fixes the expansion point to ``self.theta0``.

        Args:
            dali: DALI tensors as returned by :meth:`ForecastKit.dali`.
            bias: Optional dictionary of systematic bias tensors indexed by expansion
                order, as returned by :func:`derivkit.forecasting.build_dali_bias_tensor`.
                These contribute to the target log posterior.
            names: Parameter names for GetDist (length ``p``).
            labels: Parameter labels for GetDist (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_emcee`
                (e.g. ``forecast_order``, ``n_steps``, ``burn``, ``thin``,
                ``n_walkers``, ``init_scale``, ``seed``, ``prior_terms``,
                ``prior_bounds``, ``logprior``, ``sampler_bounds``, ``label``).

        Returns:
            A :class:`getdist.MCSamples` containing MCMC chains.
        """
        kwargs = _drop_reserved_kwargs(kwargs, reserved=_RESERVED_KWARGS)

        return dali_to_getdist_emcee(
            theta0=self.theta0,
            dali=dali,
            bias=bias,
            names=names,
            labels=labels,
            **kwargs,
        )


def _drop_reserved_kwargs(
    kwargs: Mapping[str, Any],
    *,
    reserved: set[str]
) -> dict[str, Any]:
    """Removes reserved keyword arguments from a dictionary."""
    return {k: v for k, v in kwargs.items() if k not in reserved}
