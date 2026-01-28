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
from typing import Any, Mapping, Sequence

import numpy as np

from derivkit.forecasting.dali import build_dali
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
from derivkit.utils.validate import (
    require_callable,
    resolve_covariance_input,
    validate_covariance_matrix_shape,
)

_RESERVED_KWARGS = {"theta0"}

class ForecastKit:
    """Provides access to Fisher and DALI likelihoods-expansion tensors."""

    def __init__(
            self,
            function: Callable[[Sequence[float] | np.ndarray], np.ndarray] | None,
            theta0: Sequence[float] | np.ndarray,
            cov: np.ndarray
                 | Callable[[np.ndarray], np.ndarray],
    ):
        r"""Initialises the ForecastKit with model, fiducials, and covariance.

        Args:
            function: Callable returning the model mean vector :math:`\mu(\theta)`.
                May be ``None`` if you only plan to use covariance-only workflows
                (e.g. generalized Fisher with ``term="cov"``). Required for
                :meth:`fisher`, :meth:`dali`, and :meth:`fisher_bias`.
            theta0: Fiducial parameter values of shape ``(p,)`` where ``p`` is the
                number of parameters.
            cov: Covariance specification. Supported forms are:

                - ``cov=C0``: fixed covariance matrix :math:`C(\theta_0)` with shape
                  ``(n_obs, n_obs)``, where ``n_obs`` is the number of observables.
                - ``cov=cov_fn``: callable with ``cov_fn(theta)`` returning the covariance
                  matrix :math:`C(\theta)` evaluated at the parameter vector ``theta``,
                  with shape ``(n_obs, n_obs)``. The covariance at ``theta0`` is evaluated
                  once and cached.
        """
        self.function = function
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

        Raises:
          ValueError: If input shapes differ, inputs are not 1D/2D, or the
            flattened length does not match ``n_observables``.
          FloatingPointError: If non-finite values are detected in the result.
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
            **dk_kwargs: Additional keyword arguments passed to
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
            Fisher matrix with shape ``(p, p)``.
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

    def delta_chi2_fisher(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
    ) -> float:
        """Computes a displacement chi-squared under the Fisher approximation.

        This evaluates the standard quadratic form

            ``delta_chi2 = (theta - theta0)^T @ F @ (theta - theta0)``

        using the provided Fisher matrix and the stored expansion point ``self.theta0``.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            fisher: Fisher matrix with shape ``(p, p)``.

        Returns:
            Scalar delta chi-squared value.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        return build_delta_chi2_fisher(theta=theta, theta0=self.theta0, fisher=fisher)

    def delta_chi2_dali(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
        g_tensor: np.ndarray,
        h_tensor: np.ndarray | None,
        convention: str = "delta_chi2",
    ) -> float:
        """Computes a displacement chi-squared under the DALI approximation.

        This evaluates a scalar ``delta_chi2`` from the displacement
        ``d = theta - theta0`` using the Fisher matrix and (optionally) the cubic
        and quartic DALI tensors. The ``convention`` parameter controls the
        numerical prefactors applied to the cubic/quartic contractions.

        The expansion point is taken from the stored ``self.theta0``.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            fisher: Fisher matrix with shape ``(p, p)``.
            g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
            h_tensor: DALI quartic tensor with shape ``(p, p, p, p)`` or ``None``.
            convention: Controls the prefactors used in the cubic/quartic tensor
                contractions. Supported conventions are ``"delta_chi2"`` and
                ``"matplotlib_loglike"``.

        Returns:
            Scalar delta chi-squared value.

        Raises:
            ValueError: If an unknown ``convention`` is provided.
            ValueError: If input shapes are inconsistent.
        """
        return build_delta_chi2_dali(
            theta=theta,
            theta0=self.theta0,
            fisher=fisher,
            g_tensor=g_tensor,
            h_tensor=h_tensor,
            convention=convention,
        )

    def logposterior_fisher(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
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

        Raises:
            ValueError: If shapes are inconsistent or if both prior styles are provided.
        """
        return build_logposterior_fisher(
            theta=theta,
            theta0=self.theta0,
            fisher=fisher,
            prior_terms=prior_terms,
            prior_bounds=prior_bounds,
            logprior=logprior,
        )

    def logposterior_dali(
        self,
        *,
        theta: np.ndarray,
        fisher: np.ndarray,
        g_tensor: np.ndarray,
        h_tensor: np.ndarray | None,
        convention: str = "delta_chi2",
        prior_terms: Sequence[tuple[str, dict[str, Any]] | dict[str, Any]] | None = None,
        prior_bounds: Sequence[tuple[float | None, float | None]] | None = None,
        logprior: Callable[[np.ndarray], float] | None = None,
    ) -> float:
        """Computes the log posterior (up to a constant) under the DALI approximation.

        If no prior is provided, this returns the DALI log-likelihood expansion with
        a flat prior and no hard cutoffs. Priors may be provided either as a pre-built
        ``logprior(theta)`` callable or as a lightweight prior specification via
        ``prior_terms`` and/or ``prior_bounds``.

        The ``convention`` parameter controls the numerical prefactors used in the
        cubic/quartic contractions and matches the underlying expansion helpers.

        The expansion point is taken from the stored ``self.theta0``.

        Args:
            theta: Evaluation point in parameter space with shape ``(p,)``.
            fisher: Fisher matrix with shape ``(p, p)``.
            g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
            h_tensor: DALI quartic tensor with shape ``(p, p, p, p)`` or ``None``.
            convention: The normalization to use (``"delta_chi2"`` or
                ``"matplotlib_loglike"``).
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

        Raises:
            ValueError: If an unknown ``convention`` is provided.
            ValueError: If shapes are inconsistent or if both prior styles are provided.
        """
        return build_logposterior_dali(
            theta=theta,
            theta0=self.theta0,
            fisher=fisher,
            g_tensor=g_tensor,
            h_tensor=h_tensor,
            convention=convention,
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

        Raises:
            ValueError: If ``theta`` is not a finite 1D vector or if the negative log-posterior
                evaluates to a non-finite value.
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

        Raises:
            TypeError: If ``neg_logposterior`` is not scalar-valued (Hessian is not 2D).
            ValueError: If inputs are invalid or the Hessian is not a finite square matrix.
        """
        theta = self.theta0 if theta_map is None else theta_map
        return build_laplace_hessian(
            neg_logposterior=neg_logposterior,
            theta_map=theta,
            method=method,
            n_workers=n_workers,
            **dk_kwargs,
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

        Raises:
            ValueError: If ``hessian`` is not a finite square matrix.
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

        Raises:
            TypeError: If ``neg_logposterior`` is not scalar-valued.
            ValueError: If inputs are invalid or non-finite values are encountered.
            np.linalg.LinAlgError: If ``ensure_spd=True`` and the Hessian cannot be regularized to be SPD.
        """
        theta = self.theta0 if theta_map is None else theta_map
        return build_laplace_approximation(
            neg_logposterior=neg_logposterior,
            theta_map=theta,
            method=method,
            n_workers=n_workers,
            ensure_spd=ensure_spd,
            rcond=rcond,
            **dk_kwargs,
        )

    def getdist_fisher_gaussian(
            self,
            *,
            fisher: np.ndarray,
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
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_fisher_samples(
            self,
            *,
            fisher: np.ndarray,
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
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_dali_importance(
            self,
            *,
            fisher: np.ndarray,
            g_tensor: np.ndarray,
            h_tensor: np.ndarray | None,
            names: Sequence[str],
            labels: Sequence[str],
            **kwargs: Any,
    ):
        """Returns GetDist :class:`getdist.MCSamples` for a DALI posterior via importance sampling.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_importance`
        that fixes the expansion point to ``self.theta0``.

        Args:
            fisher: Fisher matrix with shape ``(p, p)`` at ``self.theta0``.
            g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
            h_tensor: Optional DALI quartic tensor with shape ``(p, p, p, p)``.
            names: Parameter names for GetDist (length ``p``).
            labels: Parameter labels for GetDist (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_importance`
                (e.g. ``n_samples``, ``kernel_scale``, ``convention``, ``seed``,
                ``prior_terms``, ``prior_bounds``, ``logprior``, ``sampler_bounds``, ``label``).

        Returns:
            A :class:`getdist.MCSamples` with importance weights.
        """
        kwargs = _drop_reserved_kwargs(kwargs, reserved=_RESERVED_KWARGS)

        return dali_to_getdist_importance(
            theta0=self.theta0,
            fisher=fisher,
            g_tensor=g_tensor,
            h_tensor=h_tensor,
            names=names,
            labels=labels,
            **kwargs,
        )

    def getdist_dali_emcee(
            self,
            *,
            fisher: np.ndarray,
            g_tensor: np.ndarray,
            h_tensor: np.ndarray | None,
            names: Sequence[str],
            labels: Sequence[str],
            **kwargs: Any,
    ):
        """Returns GetDist :class:`getdist.MCSamples` from ``emcee`` sampling of a DALI posterior.

        This is a thin wrapper around
        :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_emcee`
        that fixes the expansion point to ``self.theta0``.

        Args:
            fisher: Fisher matrix with shape ``(p, p)`` at ``self.theta0``.
            g_tensor: DALI cubic tensor with shape ``(p, p, p)``.
            h_tensor: Optional DALI quartic tensor with shape ``(p, p, p, p)``.
            names: Parameter names for GetDist (length ``p``).
            labels: Parameter labels for GetDist (length ``p``).
            **kwargs: Forwarded to
                :func:`derivkit.forecasting.getdist_dali_samples.dali_to_getdist_emcee`
                (e.g. ``n_steps``, ``burn``, ``thin``, ``n_walkers``, ``init_scale``, ``seed``,
                ``convention``, ``prior_terms``, ``prior_bounds``, ``logprior``,
                ``sampler_bounds``, ``label``).

        Returns:
            A :class:`getdist.MCSamples` containing MCMC chains.

        Raises:
            TypeError: If ``theta0`` is provided (ForecastKit always uses ``self.theta0``).
            ValueError: If shapes are inconsistent or mutually exclusive options are provided.
            RuntimeError: If walker initialization fails.
        """
        kwargs = _drop_reserved_kwargs(kwargs, reserved=_RESERVED_KWARGS)

        return dali_to_getdist_emcee(
            theta0=self.theta0,
            fisher=fisher,
            g_tensor=g_tensor,
            h_tensor=h_tensor,
            names=names,
            labels=labels,
            **kwargs,
        )


def _drop_reserved_kwargs(
    kwargs: Mapping[str, Any],
    *,
    reserved: set[str]
) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in reserved}
