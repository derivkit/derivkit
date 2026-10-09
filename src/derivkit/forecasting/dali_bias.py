"""DALI systematic-bias tensor utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from derivkit.forecasting.forecast_core import (
    SUPPORTED_DERIVATIVE_ORDERS,
    _get_derivatives,
    get_forecast_tensors,
)
from derivkit.utils.linalg import invert_covariance
from derivkit.utils.tensors import (
    contract_tensor_with_vector,
    gaussian_fourth_moment,
)
from derivkit.utils.types import Array, ArrayLike1D, ArrayLike2D, FloatArray
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "build_dali_bias_tensor",
    "build_dali_map_shift_from_bias",
    "build_dali_map_shift",
    "build_dali_mean_map_offset_from_bias",
    "build_dali_mean_shift_from_bias",
    "build_dali_mean_shift",
    "build_dali_response_potential",
    "build_dali_mean_response_coefficients",
    "build_dali_mean_response_shift",
    "build_dali_mean_response_displacement",
    "build_dali_response_posterior_mean",
    "build_dali_mean_response",
    "build_dali_bias_shifts",
]


def build_dali_bias_tensor(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
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
        function: The scalar or vector-valued model function.
        theta0: Fiducial parameter values at which derivatives are evaluated.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between two data vectors, which may represent
            a systematic mismatch or a difference between model predictions.
            Accepts a 1D array or a column vector of shape ``(n_observables, 1)``.
            The input is flattened internally.
        bias_order: Highest mismatch tensor order to compute. Supported values
            are given in
            :data:`derivkit.forecasting.forecast_core.SUPPORTED_DERIVATIVE_ORDERS`.
        method: Numerical differentiation method. If ``None``, the
            :class:`derivkit.derivative_kit.DerivativeKit` default is used.
        n_workers: Number of workers for per-parameter parallelization/threads.
            Default ``1`` (serial).
        derivatives: Optional dictionary of precomputed model derivatives.
            Missing orders are computed and stored for reuse.
        **dk_kwargs: Additional keyword arguments passed to DerivKit's
            differentiation machinery.

    Returns:
        A dictionary mapping derivative orders 1 through ``bias_order`` to
        their corresponding mismatch tensors. Zeroth-order terms are omitted
        because they are independent of the parameter displacement.

    Raises:
        TypeError: If ``bias_order`` is not an integer.
        ValueError: If ``bias_order`` is unsupported, ``theta0`` is empty, or
            the data mismatch does not match the number of observables.
    """
    if not isinstance(bias_order, (int, np.integer)) or isinstance(bias_order, (bool, np.bool_)):
        raise TypeError(f"bias_order must be an int; got {type(bias_order)}.")

    if bias_order not in SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            f"bias_order={bias_order} is not supported. "
            f"Supported values: {SUPPORTED_DERIVATIVE_ORDERS}."
        )

    theta0_arr = np.asarray(theta0, dtype=float).reshape(-1)
    if theta0_arr.size == 0:
        raise ValueError("theta0 must be non-empty 1D.")

    cov_arr = validate_covariance_matrix_shape(cov)
    n_observables = cov_arr.shape[0]

    delta_nu_arr = np.asarray(delta_nu, dtype=float).reshape(-1)
    if delta_nu_arr.size != n_observables:
        raise ValueError(
            f"Expected {n_observables} elements in delta_nu "
            f"(from cov {cov_arr.shape}), but got {delta_nu_arr.size}."
        )

    y0 = np.asarray(function(theta0_arr), dtype=float)
    y0_flat = y0.reshape(-1)

    if y0_flat.size != n_observables:
        raise ValueError(
            f"Expected {n_observables} observables from model "
            f"(from cov {cov_arr.shape}), "
            f"but got {y0_flat.size} (output shape {y0.shape})."
        )

    inv_cov = invert_covariance(cov_arr, warn_prefix="build_dali_bias")

    bias: dict[int, FloatArray] = {}

    if derivatives is None:
        derivatives = {}

    for order in range(1, bias_order + 1):
        if order not in derivatives:
            derivatives[order] = _get_derivatives(
                function,
                theta0_arr,
                cov_arr,
                order=order,
                method=method,
                n_workers=n_workers,
                **dk_kwargs,
            )

        derivative = derivatives[order]

        bias[order] = np.einsum(
            "i...,ij,j->...",
            derivative, inv_cov, delta_nu_arr,
        ).astype(np.float64, copy=False)

    return bias


def build_dali_map_shift_from_bias(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    *,
    expansion_order: int = 3,
) -> FloatArray:
    """Computes the perturbative DALI MAP shift from mismatch tensors.

    Args:
        dali: Precomputed fully symmetrized Fisher and DALI tensors.
        bias: Systematic mismatch tensors indexed by derivative order.
        expansion_order: Highest systematic response order, from 1 to 3.

    Returns:
        MAP parameter displacement from the fiducial point through the
        requested perturbative order.

    Raises:
        TypeError: If ``expansion_order`` is not an integer.
        ValueError: If ``expansion_order`` is not between 1 and 3.
    """
    if isinstance(expansion_order, (bool, np.bool_)) or not isinstance(
        expansion_order, (int, np.integer)
    ):
        raise TypeError("expansion_order must be an integer.")

    if expansion_order not in (1, 2, 3):
        raise ValueError("expansion_order must be 1, 2, or 3.")

    fisher_inv = np.linalg.inv(dali[1][0])
    shift1 = fisher_inv @ bias[1]

    if expansion_order == 1:
        return shift1

    D1, D2 = dali[2]

    shift2 = fisher_inv @ (
        contract_tensor_with_vector(bias[2], shift1)
        - 1.5 * contract_tensor_with_vector(D1, shift1, n_axes=2)
    )

    if expansion_order == 2:
        return shift1 + shift2

    T1, _, _ = dali[3]
    cubic = (2.0 / 3.0) * T1 + 0.5 * D2

    shift3 = fisher_inv @ (
        0.5 * contract_tensor_with_vector(bias[3], shift1, n_axes=2)
        + contract_tensor_with_vector(bias[2], shift2)
        - 3.0 * np.einsum("abc,b,c->a", D1, shift1, shift2)
        - contract_tensor_with_vector(cubic, shift1, n_axes=3)
    )

    return shift1 + shift2 + shift3


def build_dali_map_shift(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
    expansion_order: int = 3,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> FloatArray:
    """Computes the systematic-induced DALI MAP parameter shift.

    Constructs the DALI and systematic mismatch tensors using shared
    model derivatives, avoiding repeated differentiation.

    Args:
        function: The scalar or vector-valued model function.
        theta0: Fiducial parameter values.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between biased and unbiased data vectors.
        expansion_order: Highest systematic response order, from 1 to 3.
        method: Numerical differentiation method.
        n_workers: Number of workers for parallel differentiation.
        **dk_kwargs: Additional keyword arguments passed to DerivKit.

    Returns:
        MAP parameter displacement from the fiducial point through the
        requested perturbative order.
    """
    derivatives: dict[int, FloatArray] = {}

    dali = get_forecast_tensors(
        function,
        theta0,
        cov,
        forecast_order=expansion_order,
        method=method,
        n_workers=n_workers,
        derivatives=derivatives,
        **dk_kwargs,
    )

    bias = build_dali_bias_tensor(
        function,
        theta0,
        cov,
        delta_nu,
        bias_order=expansion_order,
        method=method,
        n_workers=n_workers,
        derivatives=derivatives,
        **dk_kwargs,
    )

    return build_dali_map_shift_from_bias(dali, bias, expansion_order=expansion_order)


def build_dali_mean_map_offset_from_bias(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    map_shift: FloatArray,
) -> FloatArray:
    """Computes the analytic DALI posterior mean offset from the MAP.

    The offset is evaluated using the local DALI expansion around the
    supplied MAP displacement, retaining cubic and quintic contributions.

    Args:
        dali: Precomputed fully symmetrized Fisher and DALI tensors.
        bias: Systematic mismatch tensors through third order.
        map_shift: MAP displacement from the fiducial parameters.

    Returns:
        Analytic posterior mean offset relative to the supplied MAP.
    """
    fisher = dali[1][0]
    D1, D2 = dali[2]
    T1, T2, T3 = dali[3]

    map_shift = np.asarray(map_shift, dtype=float)

    c3 = bias[3] / 6.0 - D1 / 2.0
    c4 = -T1 / 6.0 - D2 / 8.0
    c5 = -T2 / 12.0
    c6 = -T3 / 72.0

    precision = (
        fisher - bias[2]
        - 6.0 * contract_tensor_with_vector(c3, map_shift)
        - 12.0 * contract_tensor_with_vector(c4, map_shift, n_axes=2)
        - 20.0 * contract_tensor_with_vector(c5, map_shift, n_axes=3)
        - 30.0 * contract_tensor_with_vector(c6, map_shift, n_axes=4)
    )

    covariance = np.linalg.inv(precision)

    q3 = (
        c3
        + 4.0 * contract_tensor_with_vector(c4, map_shift)
        + 10.0 * contract_tensor_with_vector(c5, map_shift, n_axes=2)
        + 20.0 * contract_tensor_with_vector(c6, map_shift, n_axes=3)
    )

    q5 = c5 + 6.0 * contract_tensor_with_vector(c6, map_shift)

    fourth_moment = gaussian_fourth_moment(covariance)

    cubic_offset = 3.0 * np.einsum(
        "ijk,ia,jk->a", q3, covariance, covariance
    )

    quintic_offset = 5.0 * np.einsum(
        "ijklm,ia,jklm->a", q5, covariance, fourth_moment
    )

    return cubic_offset + quintic_offset


def build_dali_mean_shift_from_bias(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    map_shift: FloatArray,
) -> FloatArray:
    """Computes the systematic-induced DALI posterior mean shift.

    The shift includes the MAP displacement and the change in the
    mean-MAP offset relative to the unbiased posterior.

    Args:
        dali: Precomputed fully symmetrized Fisher and DALI tensors.
        bias: Systematic mismatch tensors through third order.
        map_shift: MAP displacement from the fiducial parameters.

    Returns:
        Change in the posterior mean relative to the unbiased posterior.
    """
    biased_offset = build_dali_mean_map_offset_from_bias(dali, bias, map_shift)

    zero_bias = {order: np.zeros_like(value) for order, value in bias.items()}
    baseline_offset = build_dali_mean_map_offset_from_bias(
        dali, zero_bias, np.zeros_like(map_shift)
    )

    return map_shift + biased_offset - baseline_offset


def build_dali_mean_shift(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> FloatArray:
    """Computes the systematic-induced DALI posterior mean shift.

    Constructs DALI and mismatch tensors using shared model derivatives,
    avoiding repeated differentiation.

    Args:
        function: The scalar or vector-valued model function.
        theta0: Fiducial parameter values.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between biased and unbiased data vectors.
        method: Numerical differentiation method.
        n_workers: Number of workers for parallel differentiation.
        **dk_kwargs: Additional keyword arguments passed to DerivKit.

    Returns:
        Change in the posterior mean relative to the unbiased posterior.
    """
    derivatives: dict[int, FloatArray] = {}

    dali = get_forecast_tensors(
        function,
        theta0,
        cov,
        forecast_order=3,
        method=method,
        n_workers=n_workers,
        derivatives=derivatives,
        **dk_kwargs,
    )

    bias = build_dali_bias_tensor(
        function,
        theta0,
        cov,
        delta_nu,
        bias_order=3,
        method=method,
        n_workers=n_workers,
        derivatives=derivatives,
        **dk_kwargs,
    )

    map_shift = build_dali_map_shift_from_bias(dali, bias, expansion_order=3)

    return build_dali_mean_shift_from_bias(dali, bias, map_shift)


def build_dali_response_potential(
    theta,
    theta0,
    bias,
):
    """Evaluates the systematic deformation of the DALI log posterior.

    The potential describes how a systematic data mismatch changes the
    log posterior as a function of parameter displacement from the fiducial
    point, retaining terms through cubic order.

    Args:
        theta: Parameter samples with shape (n_samples, n_parameters).
        theta0: Fiducial parameter values.
        bias: Systematic mismatch tensors through third order.

    Returns:
        Systematic potential evaluated at each parameter sample.
    """
    theta = np.asarray(theta, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)

    delta = theta - theta0

    linear = np.einsum(
        "i,...i->...",
        bias[1],
        delta,
    )

    quadratic = 0.5 * np.einsum(
        "ij,...i,...j->...",
        bias[2],
        delta,
        delta,
    )

    cubic = (1.0 / 6.0) * np.einsum(
        "ijk,...i,...j,...k->...",
        bias[3],
        delta,
        delta,
        delta,
    )

    return linear + quadratic + cubic


def build_dali_mean_response_coefficients(
    theta,
    weights,
    theta0,
    bias,
    *,
    bias_second=None,
    bias_third=None,
    bias_fourth=None,
    order=4,
):
    """Computes the perturbative response of the DALI posterior mean.

    Expands the change in the posterior mean in powers of systematic
    amplitude, retaining the full non-Gaussian unbiased DALI posterior
    as the reference distribution.

    Args:
        theta: Samples or integration nodes representing the unbiased
            DALI posterior.
        weights: Corresponding posterior weights, normalized internally.
        theta0: Fiducial parameter values.
        bias: Unit-amplitude systematic mismatch tensors.
        bias_second: Optional second amplitude derivative of the
            systematic mismatch tensors.
        bias_third: Optional third amplitude derivative of the
            systematic mismatch tensors.
        bias_fourth: Optional fourth amplitude derivative of the
            systematic mismatch tensors.
        order: Highest response order, from 1 to 4.

    Returns:
        Dictionary containing the unbiased posterior mean, fiducial
        parameters, and response coefficients through the requested order.

    Raises:
        ValueError: If the response order is outside the supported range.
    """
    if order not in (1, 2, 3, 4):
        raise ValueError(
            "order must be 1, 2, 3, or 4; "
            f"got {order}."
        )

    theta = np.asarray(theta, dtype=float)
    weights = np.asarray(weights, dtype=float)

    weights = weights / np.sum(weights)

    mean0 = np.sum(weights[:, None] * theta, axis=0)

    x = theta - mean0

    u1 = build_dali_response_potential(theta, theta0, bias)
    u2 = np.zeros_like(u1)
    u3 = np.zeros_like(u1)
    u4 = np.zeros_like(u1)

    if bias_second is not None:
        u2 = build_dali_response_potential(theta, theta0, bias_second)

    if bias_third is not None:
        u3 = build_dali_response_potential(theta, theta0, bias_third)

    if bias_fourth is not None:
        u4 = build_dali_response_potential(theta, theta0, bias_fourth)

    e1 = u1
    e2 = (0.5 * u1**2 + 0.5 * u2)
    e3 = (
        (1.0 / 6.0) * u1**3
        + 0.5 * u1 * u2
        + (1.0 / 6.0) * u3
)
    e4 = (
        (1.0 / 24.0) * u1**4
        + 0.25 * u1**2 * u2
        + (1.0 / 6.0) * u1 * u3
        + 0.125 * u2**2
        + (1.0 / 24.0) * u4
    )

    expansion_terms = {
        1: e1,
        2: e2,
        3: e3,
        4: e4,
    }

    names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
    }

    denominator = {}
    numerator = {}

    for response_order in range(1, order + 1):
        term = expansion_terms[response_order]
        denominator[response_order] = np.sum(weights * term)
        numerator[response_order] = np.sum(weights[:, None] * x * term[:, None], axis=0)

    responses = {}

    for response_order in range(1, order + 1):
        response = np.array(numerator[response_order], copy=True)

        for lower_order in range(1, response_order):
            response -= (denominator[lower_order] * responses[response_order - lower_order])

        responses[response_order] = response

    coefficients = {
        "mean0": mean0,
        "theta0": np.asarray(theta0, dtype=float),
    }

    for response_order in range(1, order + 1):
        coefficients[names[response_order]] = responses[response_order]

    return coefficients


def build_dali_mean_response_shift(
    coefficients,
    amplitude,
    *,
    order=4,
):
    """Evaluates the systematic shift in the DALI posterior mean.

    Reconstructs the change in the posterior mean relative to the
    unbiased posterior using the perturbative response coefficients.

    Args:
        coefficients: Posterior mean response coefficients.
        amplitude: Amplitude of the systematic mismatch.
        order: Highest response order to include, from 1 to 4.

    Returns:
        Posterior mean shift relative to the unbiased posterior.

    Raises:
        ValueError: If the response order is invalid or a required
            coefficient is unavailable.
    """
    if order not in (1, 2, 3, 4):
        raise ValueError(
            "order must be 1, 2, 3, or 4; "
            f"got {order}."
        )

    amplitude = float(amplitude)

    names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
    }

    shift = np.zeros_like(coefficients["mean0"], dtype=float)

    for response_order in range(1, order + 1):
        name = names[response_order]

        if name not in coefficients:
            raise ValueError(
                f"Response coefficient {name!r} is not available."
            )

        shift += (amplitude**response_order * coefficients[name])

    return shift


def build_dali_mean_response_displacement(
    coefficients,
    amplitude,
    *,
    order=4,
):
    """Evaluates the DALI posterior mean displacement from the fiducial point.

    Includes both the displacement of the unbiased posterior mean from
    the fiducial parameters and the additional shift induced by the
    systematic mismatch.

    Args:
        coefficients: Posterior mean response coefficients.
        amplitude: Amplitude of the systematic mismatch.
        order: Highest response order to include, from 1 to 4.

    Returns:
        Posterior mean displacement relative to the fiducial parameters.
    """
    baseline_displacement = (coefficients["mean0"] - coefficients["theta0"])

    return (
        baseline_displacement
        + build_dali_mean_response_shift(
            coefficients,
            amplitude,
            order=order,
        )
    )


def build_dali_response_posterior_mean(
    coefficients,
    amplitude,
    *,
    order=4,
):
    """Reconstructs the DALI posterior mean under a systematic mismatch.

    Provides the posterior mean, its displacement from the fiducial
    parameters, and its shift relative to the unbiased posterior.

    Args:
        coefficients: Posterior mean response coefficients.
        amplitude: Amplitude of the systematic mismatch.
        order: Highest response order to include, from 1 to 4.

    Returns:
        Dictionary containing the posterior mean, displacement from
        the fiducial parameters, and systematic mean shift.
    """
    shift = build_dali_mean_response_shift(
        coefficients,
        amplitude,
        order=order,
    )

    mean = (coefficients["mean0"] + shift)
    displacement = (mean - coefficients["theta0"])

    return {
        "mean": mean,
        "displacement": displacement,
        "shift": shift,
    }

def build_dali_mean_response(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    theta: FloatArray,
    weights: FloatArray,
    *,
    amplitude: float = 1.0,
    order: int = 4,
    method: str | None = None,
    n_workers: int = 1,
    derivatives: dict[int, FloatArray] | None = None,
    **dk_kwargs: Any,
) -> FloatArray:
    """Computes the systematic response of the full DALI posterior mean.

    Predicts the change in the posterior mean using a perturbative
    expansion in systematic amplitude, while retaining the non-Gaussian
    structure of the unbiased DALI posterior.

    Args:
        function: Scalar or vector-valued model function.
        theta0: Fiducial parameter values.
        cov: Covariance matrix of the observables.
        delta_nu: Unit-amplitude systematic data mismatch.
        theta: Samples or integration nodes representing the unbiased
            DALI posterior.
        weights: Corresponding posterior weights, normalized internally.
        amplitude: Amplitude of the systematic mismatch.
        order: Highest response order, from 1 to 4.
        method: Numerical differentiation method.
        n_workers: Number of workers for parallel differentiation.
        derivatives: Optional precomputed model derivatives.
        **dk_kwargs: Additional arguments passed to DerivKit.

    Returns:
        Systematic posterior mean shift relative to the unbiased
        DALI posterior.
    """
    bias = build_dali_bias_tensor(
        function, theta0, cov, delta_nu,
        bias_order=3, method=method, n_workers=n_workers,
        derivatives=derivatives, **dk_kwargs,
    )
    coefficients = build_dali_mean_response_coefficients(
        theta, weights, theta0, bias, order=order,
    )
    return build_dali_mean_response_shift(coefficients, amplitude, order=order)


def build_dali_bias_shifts(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    theta: FloatArray,
    weights: FloatArray,
    *,
    amplitude: float = 1.0,
    response_order: int = 4,
    method: str | None = None,
    n_workers: int = 1,
    **dk_kwargs: Any,
) -> dict[str, FloatArray]:
    """Computes three DALI estimates of systematic parameter shifts.

    Returns the perturbative MAP shift, the analytical posterior mean
    shift based on the local expansion around the MAP, and the posterior
    mean response obtained from the full non-Gaussian reference posterior.

    Args:
        function: Scalar or vector-valued model function.
        theta0: Fiducial parameter values.
        cov: Covariance matrix of the observables.
        delta_nu: Unit-amplitude systematic data mismatch.
        theta: Samples or integration nodes representing the unbiased
            DALI posterior.
        weights: Corresponding posterior weights, normalized internally.
        amplitude: Amplitude of the systematic mismatch.
        response_order: Highest posterior mean response order, from 1 to 4.
        method: Numerical differentiation method.
        n_workers: Number of workers for parallel differentiation.
        **dk_kwargs: Additional arguments passed to DerivKit.

    Returns:
        Dictionary with three parameter shifts:
        ``map`` for the MAP shift, ``analytic_mean`` for the analytical
        mean shift, and ``response_mean`` for the posterior mean response.
    """
    derivatives: dict[int, FloatArray] = {}
    dali = get_forecast_tensors(
        function, theta0, cov, forecast_order=3,
        method=method, n_workers=n_workers,
        derivatives=derivatives, **dk_kwargs,
    )
    unit_bias = build_dali_bias_tensor(
        function, theta0, cov, delta_nu, bias_order=3,
        method=method, n_workers=n_workers,
        derivatives=derivatives, **dk_kwargs,
    )
    bias = {order: amplitude * tensor for order, tensor in unit_bias.items()}
    map_shift = build_dali_map_shift_from_bias(dali, bias, expansion_order=3)
    analytic_mean_shift = build_dali_mean_shift_from_bias(dali, bias, map_shift)
    coefficients = build_dali_mean_response_coefficients(
        theta, weights, theta0, unit_bias, order=response_order,
    )
    response_mean_shift = build_dali_mean_response_shift(
        coefficients, amplitude, order=response_order,
    )
    return {
        "map": map_shift,
        "analytic_mean": analytic_mean_shift,
        "response_mean": response_mean_shift,
    }
