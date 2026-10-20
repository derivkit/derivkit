"""DALI systematic-bias forecasts.

The three main entry points return systematic MAP shifts, analytic posterior-mean
shifts, and posterior-mean response shifts.
Optional metadata explains where each estimator ends up and which terms
contribute to its response. All shifts compare the same inference model
with and without an injected data-model mismatch.

The implementation follows https://arxiv.org/abs/2610.02104.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Callable

import numpy as np

from derivkit.forecasting.forecast_core import (
    SUPPORTED_DERIVATIVE_ORDERS,
    _get_derivatives,
    get_forecast_tensors,
)
from derivkit.utils.linalg import invert_covariance, solve_or_pinv
from derivkit.utils.tensors import (
    contract_tensor_with_vector,
    contract_tensor_with_vector_batch,
    gaussian_fourth_moment,
)
from derivkit.utils.types import Array, ArrayLike1D, ArrayLike2D, FloatArray
from derivkit.utils.validate import validate_covariance_matrix_shape

__all__ = [
    "build_dali_bias_tensor",
    "build_dali_map_shift",
    "build_dali_mean_shift",
    "build_dali_mean_response_shift",
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


def _map_terms(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    order: int,
) -> dict[int, FloatArray]:
    """Compute successive terms in the perturbative DALI MAP shift.

    Args:
        dali: Dictionary of Fisher and higher-order DALI tensors, keyed by
            forecast order.
        bias: Dictionary of systematic mismatch tensors, keyed by derivative
            order.
        order: Highest MAP response order to compute, from 1 to 3.

    Returns:
        Dictionary mapping each response order to its contribution to the
        MAP shift. The first term is the Fisher bias, while the second and
        third terms account for nonlinear corrections. Summing the terms
        gives the predicted MAP shift.

    Raises:
        TypeError: If order is not an integer.
        ValueError: If order is not 1, 2, or 3.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError(f"expansion_order must be an integer; got {type(order)}.")
    if order not in (1, 2, 3):
        raise ValueError("expansion_order must be 1, 2, or 3.")

    fisher = dali[1][0]
    s1 = solve_or_pinv(fisher, bias[1], warn_context="DALI MAP")
    terms = {1: s1}
    if order >= 2:
        D1, D2 = dali[2]
        s2 = solve_or_pinv(fisher, (
            contract_tensor_with_vector(bias[2], s1)
            - 1.5 * contract_tensor_with_vector(D1, s1, n_axes=2)
        ), warn_context="DALI MAP")
        terms[2] = s2
    if order >= 3:
        T1 = dali[3][0]
        cubic = (2.0 / 3.0) * T1 + 0.5 * D2
        s3 = solve_or_pinv(fisher, (
            0.5 * contract_tensor_with_vector(bias[3], s1, n_axes=2)
            + contract_tensor_with_vector(bias[2], s2)
            - 3.0 * np.einsum("abc,b,c->a", D1, s1, s2)
            - contract_tensor_with_vector(cubic, s1, n_axes=3)
        ), warn_context="DALI MAP")
        terms[3] = s3
    return terms


def _map_gaussian_terms(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    order: int,
) -> dict[int, FloatArray]:
    """Compute MAP response terms for a Gaussian reference posterior.

    The unbiased posterior is approximated by its quadratic Fisher form,
    while the systematic mismatch retains its higher-order dependence
    on the parameters.

    Args:
        dali: Dictionary of Fisher and higher-order DALI tensors, keyed by
            forecast order. Only the Fisher matrix is used.
        bias: Dictionary of systematic mismatch tensors, keyed by derivative
            order.
        order: Highest MAP response order to compute, from 1 to 3.

    Returns:
        Dictionary mapping each response order to its contribution to the
        MAP shift under the Gaussian reference approximation. Summing the
        terms gives the Gaussian-reference contribution to the MAP shift.
    """
    fisher = dali[1][0]
    s1 = solve_or_pinv(fisher, bias[1], warn_context="DALI MAP")
    terms = {1: s1}
    if order >= 2:
        s2 = solve_or_pinv(fisher, contract_tensor_with_vector(bias[2], s1), warn_context="Gaussian MAP")
        terms[2] = s2
    if order >= 3:
        terms[3] = solve_or_pinv(fisher, (
            0.5 * contract_tensor_with_vector(bias[3], s1, n_axes=2)
            + contract_tensor_with_vector(bias[2], terms[2])
        ), warn_context="Gaussian MAP")
    return terms


def _sum_terms(terms: dict[int, FloatArray]) -> FloatArray:
    """Sum the contributions to a parameter shift.

    Args:
        terms: Non-empty dictionary of parameter-shift vectors, keyed by
            response order.

    Returns:
        Array containing the total parameter shift obtained by summing
        all response-order contributions.
    """
    return sum(terms.values(), np.zeros_like(next(iter(terms.values()))))


def _forecast_tensors(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    order: int,
    method: str | None,
    n_workers: int,
    derivatives: dict[int, FloatArray],
    dk_kwargs: dict[str, Any],
) -> tuple[dict[int, tuple[FloatArray, ...]], dict[int, FloatArray]]:
    """Build the DALI forecast tensors and systematic mismatch tensors.

    The model derivatives are shared between the two calculations to avoid
    recomputing derivatives of the same order.

    Args:
        function: Model predicting observables from the parameters.
        theta0: Fiducial parameter values at which derivatives are evaluated.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between the systematically affected data vector
            and the fiducial model prediction.
        order: Highest derivative order used for both the DALI forecast and
            systematic mismatch tensors.
        method: Numerical differentiation method.
        n_workers: Number of differentiation workers.
        derivatives: Dictionary of precomputed model derivatives. Missing
            orders are computed and added to this dictionary in place.
        dk_kwargs: Additional keyword arguments passed to the numerical
            differentiation method.

    Returns:
        Tuple containing two dictionaries:

        - The DALI forecast tensors, keyed by forecast order. Each entry
          contains the tensors associated with that order.
        - The systematic mismatch tensors, keyed by derivative order.
          Each tensor describes how the data-model mismatch couples to
          derivatives of the model.
    """
    dali = get_forecast_tensors(
        function, theta0, cov, forecast_order=order,
        method=method, n_workers=n_workers,
        derivatives=derivatives, **dk_kwargs,
    )
    bias = build_dali_bias_tensor(
        function, theta0, cov, delta_nu, bias_order=order,
        method=method, n_workers=n_workers,
        derivatives=derivatives, **dk_kwargs,
    )
    return dali, bias


def build_dali_map_shift(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
    expansion_order: int = 3,
    return_metadata: bool = False,
    method: str | None = None,
    n_workers: int = 1,
    derivatives: dict[int, FloatArray] | None = None,
    **dk_kwargs: Any,
):
    """Predict how the posterior maximum moves when a systematic is introduced.

    Args:
        function: Model predicting the observables from the parameters.
        theta0: Fiducial parameter values. Assumed to be the MAP without
            the systematic mismatch.
        cov: Covariance matrix of the observables.
        delta_nu: Actual difference between the systematically affected data
            vector and the fiducial model prediction. Its size is already
            included; no separate amplitude parameter is needed.
        expansion_order: Number of perturbative MAP response orders (1 to 3).
        return_metadata: If set to ``True``, return the shift and a dictionary of
            explanatory results; otherwise return only the shift.
        method: Numerical differentiation method passed to
            :class:`derivkit.derivative_kit.DerivativeKit`.
        n_workers: Number of differentiation workers.
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
    if isinstance(expansion_order, (bool, np.bool_)) or not isinstance(
            expansion_order, (int, np.integer)
    ):
        raise TypeError("expansion_order must be an integer.")

    if expansion_order not in (1, 2, 3):
        raise ValueError("expansion_order must be 1, 2, or 3.")

    if derivatives is None:
        derivatives = {}

    dali, bias = _forecast_tensors(
        function, theta0, cov, delta_nu, expansion_order,
        method, n_workers, derivatives, dk_kwargs,
    )
    terms = _map_terms(dali, bias, expansion_order)
    shift = _sum_terms(terms)
    if not return_metadata:
        return shift

    gaussian_terms = _map_gaussian_terms(dali, bias, expansion_order)
    non_gaussian_terms = {k: terms[k] - gaussian_terms[k] for k in terms}
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    metadata = {
        "estimator": theta0 + shift,
        "baseline": theta0,
        "displacement": shift,
        "terms": terms,
        "gaussian_terms": gaussian_terms,
        "non_gaussian_terms": non_gaussian_terms,
        "gaussian_contribution": _sum_terms(gaussian_terms),
        "non_gaussian_contribution": _sum_terms(non_gaussian_terms),
    }
    return shift, metadata


def _mean_map_offset(
    dali: dict[int, tuple[FloatArray, ...]],
    bias: dict[int, FloatArray],
    map_shift: FloatArray,
) -> FloatArray:
    """Evaluate the analytic mean-minus-MAP correction.

    The offset is evaluated using the local DALI expansion around the
    supplied MAP displacement, retaining cubic and quintic contributions.

    Args:
        dali: Precomputed fully symmetrized Fisher and DALI tensors.
        bias: Systematic mismatch tensors through third order.
        map_shift: MAP displacement from the fiducial parameters.

    Returns:
        Analytic posterior mean offset relative to the supplied MAP.
    """
    if bias.keys() != dali.keys():
        raise ValueError("bias and dali must contain the same tensor orders.")

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

    covariance = solve_or_pinv(
        precision, np.eye(precision.shape[0]), warn_context="DALI analytic mean"
    )

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


def _response_potential(
    theta: FloatArray,
    theta0: ArrayLike1D,
    bias: dict[int, FloatArray],
) -> FloatArray:
    """Evaluate the systematic contribution to the log posterior.

    Includes every available mismatch tensor, with the appropriate
    Taylor factorial for its order.

    Args:
        theta: Parameter samples with shape (n_samples, n_parameters).
        theta0: Fiducial parameter values.
        bias: Dictionary of mismatch tensors keyed by derivative order.

    Returns:
        An array containing the systematic log-posterior contribution
        at each parameter sample.
    """
    delta = np.asarray(theta, dtype=float) - np.asarray(theta0, dtype=float)
    potential = np.zeros(delta.shape[:-1], dtype=float)

    for order, tensor in bias.items():
        potential += (
            contract_tensor_with_vector_batch(tensor, delta, n_axes=order)
            / factorial(order)
        )

    return potential


def _mean_response_coefficients(
    theta: FloatArray,
    weights: FloatArray,
    theta0: ArrayLike1D,
    bias: dict[int, FloatArray],
    *,
    bias_second: dict[int, FloatArray] | None = None,
    bias_third: dict[int, FloatArray] | None = None,
    bias_fourth: dict[int, FloatArray] | None = None,
    order: int = 4,
) -> dict[str, FloatArray]:
    """Compute the systematic response coefficients of the posterior mean.

    Expands the change in the posterior mean in powers of systematic
    amplitude, using the supplied unbiased posterior as the reference
    distribution.

    Args:
        theta: Samples or integration nodes representing the unbiased
            reference posterior.
        weights: Nonnegative weights representing the reference posterior
            at the supplied points. Normalized internally.
        theta0: Fiducial parameter values.
        bias: Tensors for the full supplied mismatch. For a hypothetical
            scaled mismatch t * delta_nu, these are the coefficients of t.
        bias_second: Optional second amplitude derivative of the
            systematic mismatch tensors.
        bias_third: Optional third amplitude derivative of the
            systematic mismatch tensors.
        bias_fourth: Optional fourth amplitude derivative of the
            systematic mismatch tensors.
        order: Highest response order to include (1 to 4, inclusive).

    Returns:
        Dictionary containing the unbiased posterior mean, fiducial
        parameters, and response coefficients through the requested order.

    Raises:
        ValueError: If the response order is outside the supported range.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order not in (1, 2, 3, 4):
        raise ValueError(
            "order must be 1, 2, 3, or 4; "
            f"got {order}."
        )

    theta = np.asarray(theta, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if theta.ndim != 2 or weights.ndim != 1 or len(weights) != len(theta):
        raise ValueError("theta must be (n_samples, n_parameters) and weights (n_samples,).")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("weights must be finite, nonnegative, and have a positive sum.")
    weights = weights / np.sum(weights)

    mean0 = np.sum(weights[:, None] * theta, axis=0)

    x = theta - mean0

    u1 = _response_potential(theta, theta0, bias)
    u2 = np.zeros_like(u1)
    u3 = np.zeros_like(u1)
    u4 = np.zeros_like(u1)

    if bias_second is not None:
        u2 = _response_potential(theta, theta0, bias_second)

    if bias_third is not None:
        u3 = _response_potential(theta, theta0, bias_third)

    if bias_fourth is not None:
        u4 = _response_potential(theta, theta0, bias_fourth)

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

    # Series division: R(t) = N(t) / D(t), where
    # N(t) = sum(n_k * t**k) and D(t) = 1 + sum(d_k * t**k).
    # Matching coefficients in N(t) = D(t) * R(t) gives
    # r_k = n_k - sum(d_j * r_(k-j), j=1,...,k-1).
    # Here n_k = numerator[k], d_k = denominator[k],
    # and r_k = responses[k].
    #
    # For example:
    # r_1 = n_1
    # r_2 = n_2 - d_1*n_1
    # r_3 = n_3 - d_1*n_2 + (d_1**2 - d_2)*n_1
    # r_4 = n_4 - d_1*n_3 + (d_1**2 - d_2)*n_2
    #       + (-d_1**3 + 2*d_1*d_2 - d_3)*n_1
    #
    # Thus the higher powers of the denominator are already included
    # through the lower-order responses, rather than written explicitly.
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


def build_dali_mean_shift(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
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
        function: Model predicting observables from parameters.
        theta0: Fiducial parameter values and assumed baseline MAP.
        cov: Covariance matrix of the observables.
        delta_nu: Difference between the systematically affected data
            vector and the fiducial model prediction, including its magnitude.
        return_metadata: If ``True``, return the shift and explanatory metadata.
        method: Numerical differentiation method.
        n_workers: Number of differentiation workers.
        derivatives: Optional dictionary of model derivatives keyed by order.
            Existing derivatives are reused and missing derivatives are added
            to the supplied dictionary.
        **dk_kwargs: Additional arguments passed to the derivative method.

    Returns:
        An array containing the predicted systematic shift of the posterior
        mean, defined as the posterior mean with the mismatch minus the
        posterior mean without it.

        If ``return_metadata=True``, returns ``(shift, metadata)`` instead.
        The metadata dictionary contains:

        ``estimator``: Predicted posterior mean with the mismatch.

        ``baseline``: Predicted posterior mean without the mismatch.
            This need not equal ``theta0``.

        ``displacement``: Predicted posterior mean with the mismatch minus
            the fiducial parameters ``theta0``.

        ``map_shift``: Predicted systematic shift of the posterior MAP.
    """
    if derivatives is None:
        derivatives = {}

    theta0_arr = np.asarray(theta0, dtype=float).reshape(-1)

    dali, bias = _forecast_tensors(
        function, theta0, cov, delta_nu, 3,
        method, n_workers, derivatives, dk_kwargs,
    )

    map_shift = _sum_terms(_map_terms(dali, bias, 3))
    offset = _mean_map_offset(dali, bias, map_shift)

    zero_bias = {k: np.zeros_like(v) for k, v in bias.items()}
    baseline_offset = _mean_map_offset(
        dali, zero_bias, np.zeros_like(map_shift),
    )

    shift = map_shift + offset - baseline_offset

    if not return_metadata:
        return shift

    baseline = theta0_arr + baseline_offset

    return shift, {
        "estimator": baseline + shift,
        "baseline": baseline,
        "displacement": baseline_offset + shift,
        "map_shift": map_shift,
    }


def build_dali_mean_response_shift(
    function: Callable[[ArrayLike1D], np.floating | Array],
    theta0: ArrayLike1D,
    cov: ArrayLike2D,
    delta_nu: ArrayLike1D | ArrayLike2D,
    *,
    theta: FloatArray,
    weights: FloatArray | None = None,
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
        function: Model predicting observables from parameters.
        theta0: Fiducial parameter values.
        cov: Covariance matrix of the observables.
        delta_nu: Full systematic difference in the data vector.
        theta: Samples or integration nodes representing the unbiased
            reference posterior, with shape (n_samples, n_parameters).
        weights: Weights representing the reference posterior at the supplied
            points. If ``None``, equal weights are used, appropriate for
            equally weighted posterior samples. For integration nodes, the
            weights must include the reference posterior density and
            integration measure. Weights are normalized internally.
        forecast_order: Highest model derivative order included in the
            systematic mismatch potential. Supported values are 1 through 4.
        response_order: Highest order retained in the systematic amplitude
            expansion of the posterior mean. Supported values are 1 through 4.
        return_metadata: If True, return the shift and explanatory metadata.
        method: Numerical differentiation method.
        n_workers: Number of differentiation workers.
        derivatives: Optional dictionary of cached model derivatives.
            Missing orders are computed and added to the dictionary.
        **dk_kwargs: Additional arguments passed to the derivative method.

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
            the supplied samples and weights.

        ``displacement``: Predicted posterior mean with the mismatch minus
            the fiducial parameters ``theta0``.

        ``terms``: Dictionary mapping each response order to its contribution
            to the predicted posterior mean shift for the supplied mismatch.
            Their sum equals the returned shift.

    Raises:
        TypeError: If ``forecast_order`` or ``response_order`` is not an integer.
        ValueError: If either order is unsupported, or if the supplied
            reference samples or weights have invalid shapes or weights
            are not finite, nonnegative, or have zero total weight.
    """
    if derivatives is None:
        derivatives = {}

    theta0_arr = np.asarray(theta0, dtype=float).reshape(-1)

    if weights is None:
        weights = np.ones(len(theta))

    if isinstance(response_order, (bool, np.bool_)) or not isinstance(
        response_order, (int, np.integer)
    ):
        raise TypeError(f"response_order must be an integer; got {type(response_order)}.")

    if response_order not in (1, 2, 3, 4):
        raise ValueError("response_order must be 1, 2, 3, or 4.")

    if isinstance(forecast_order, (bool, np.bool_)) or not isinstance(
        forecast_order, (int, np.integer)
    ):
        raise TypeError(f"forecast_order must be an integer; got {type(forecast_order)}.")

    if forecast_order not in (1, 2, 3, 4):
        raise ValueError("forecast_order must be 1, 2, 3, or 4.")

    bias = build_dali_bias_tensor(
        function, theta0, cov, delta_nu,
        bias_order=forecast_order,
        method=method,
        n_workers=n_workers,
        derivatives=derivatives,
        **dk_kwargs,
    )

    raw = _mean_response_coefficients(
        theta, weights, theta0, bias, order=response_order,
    )

    names = ("first", "second", "third", "fourth")
    terms = {
        n: raw[names[n - 1]]
        for n in range(1, response_order + 1)
    }

    shift = _sum_terms(terms)

    if not return_metadata:
        return shift

    baseline = raw["mean0"]

    return shift, {
        "estimator": baseline + shift,
        "baseline": baseline,
        "displacement": baseline + shift - theta0_arr,
        "terms": terms,
    }
