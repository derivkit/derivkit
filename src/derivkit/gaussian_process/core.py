"""Gaussian Process core functionality: fitting, prediction, hyperparameter selection, derivatives."""

from __future__ import annotations

import warnings
from typing import Any, Tuple

import numpy as np
from numpy.linalg import LinAlgError, cholesky, solve
from scipy.optimize import minimize

from derivkit.gaussian_process.gp_utils import (
    chol_solve,
    neg_log_marginal_likelihood,
    standardize_targets,
    to_psd,
    try_cholesky,
)
from derivkit.gaussian_process.kernels import get_kernel
from derivkit.gaussian_process.kernels.base import Kernel

__all__ = ["gp_fit", "gp_predict", "gp_choose_hyperparams", "gp_derivative"]


def gp_fit(
    training_inputs: np.ndarray,
    targets: np.ndarray,
    kernel: str | Kernel,
    kernel_params: dict,
    noise_variance: float,
    *,
    normalize: bool = True,
    jitter: float = 1e-12,
    variance_floor: float = 1e-18,
) -> dict[str, Any]:
    """Fits a Gaussian Process (GP) model to training data.

    This function prepares everything needed for later predictions with a GP.
    It builds the kernel matrix on the training inputs, adds observation noise
    and a small diagonal jitter for numerical stability, and computes a
    Cholesky factorization. It also solves for the weight vector used during
    prediction and optionally centers the targets if normalization is enabled.

    In summary, the function:
      1. Validates and reshapes inputs as needed.
      2. Optionally standardizes targets to have zero mean.
      3. Forms the kernel matrix on the training inputs and adds noise and jitter.
      4. Computes a stable Cholesky factorization of this matrix.
      5. Solves for the weights needed for fast prediction.
      6. Returns a state dictionary with everything required by `gp_predict`.

    Args:
        training_inputs: Array with shape (n_samples, n_features) containing the
            training input points. A 1D array is treated as (n_samples, 1).
        targets: Array with shape (n_samples,) containing the target values.
        kernel: Kernel name or a kernel instance. If a name is provided, it is
            resolved to a kernel object.
        kernel_params: Dictionary of kernel hyperparameters.
        noise_variance: Nonnegative observation noise variance.
        normalize: If True, center targets to zero mean during fitting.
        jitter: Nonnegative value added to the kernel matrix diagonal to improve
            numerical stability during factorization.
        variance_floor: Nonnegative minimum eigenvalue used later during prediction
            to keep covariance matrices positive semidefinite.

    Returns:
        A dictionary with the fitted GP state, including:
            - "training_inputs": Training inputs as a 2D array.
            - "kernel": The resolved kernel object.
            - "kernel_params": Kernel hyperparameters.
            - "noise_variance": Observation noise variance used.
            - "chol_factor": Cholesky factor of the stabilized kernel matrix.
            - "alpha": Solution to the linear system used for prediction weights.
            - "target_mean": Mean of targets if normalization was applied, else 0.0.
            - "normalize": Whether normalization was enabled.
            - "jitter": The actual jitter used during factorization.
            - "variance_floor": The variance floor to use during prediction.

    Raises:
        TypeError: If inputs are not array-like or `kernel_params` is not a dict.
        ValueError: If shapes do not match, arrays contain NaN or Inf, or any of
            `noise_variance`, `jitter`, or `variance_floor` is negative.
        RuntimeError: If the kernel matrix cannot be factorized even after
            increasing jitter for stability.
    """
    x = np.asarray(training_inputs)
    y = np.asarray(targets)

    if x.ndim == 1:
        # allow (n,) -> (n,1)
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"`training_inputs` must be 2D (n_samples, n_features); got shape {x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"`targets` must be 1D (n_samples,); got shape {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: x has {x.shape[0]} rows but y has {y.shape[0]} elements.")

    if not isinstance(kernel_params, dict):
        raise TypeError("`kernel_params` must be a dict of kernel hyperparameters.")

    # numeric sanity
    if not np.isfinite(x).all():
        raise ValueError("`training_inputs` contains NaN or Inf.")
    if not np.isfinite(y).all():
        raise ValueError("`targets` contains NaN or Inf.")

    if noise_variance < 0:
        raise ValueError("`noise_variance` must be ≥ 0.")
    if jitter < 0:
        raise ValueError("`jitter` must be ≥ 0.")
    if variance_floor < 0:
        raise ValueError("`variance_floor` must be ≥ 0.")

    ker: Any = get_kernel(kernel) if isinstance(kernel, str) else kernel
    if ker is None:
        raise ValueError(f"Unknown kernel identifier: {kernel!r}")

    if normalize:
        y_centered, y_mean = standardize_targets(y)
    else:
        y_centered, y_mean = y, 0.0

    k_xx = ker.cov_value_value(x, x, kernel_params)  # expects (n, n)  # expects (n, n)
    if k_xx.shape != (x.shape[0], x.shape[0]):
        raise ValueError(
            f"Kernel.k returned shape {k_xx.shape}, expected {(x.shape[0], x.shape[0])}."
        )
    if not np.isfinite(k_xx).all():
        raise ValueError("Kernel matrix contains NaN/Inf.")

    # base diagonal (noise) + initial jitter
    k_mat = k_xx + (float(noise_variance) + float(jitter)) * np.eye(x.shape[0])

    # robust factorization with adaptive jitter (×10 backoff)
    try:
        chol_lower, used_jitter = try_cholesky(k_mat, jitter0=jitter, max_tries=6)
    except LinAlgError as err:
        cond_note = ""
        with np.errstate(invalid="ignore"):
            eigvals = np.linalg.eigvalsh(k_mat)
            if np.isfinite(eigvals).all():
                neg = (eigvals < 0).sum()
                cond_note = f" (min eig={eigvals.min():.3e}, neg_eigs={int(neg)})"
        raise RuntimeError(
            "Cholesky factorization failed after jitter escalation; "
            "kernel matrix appears non-PSD or ill-conditioned" + cond_note
        ) from err

    # heads up if we had to crank jitter a lot (debuggable but non-fatal)
    if used_jitter > max(1e-6, 100 * max(jitter, 1e-18)):
        warnings.warn(
            f"High jitter used for GP fit: {used_jitter:.3e} "
            f"(initial={jitter:.3e}). Consider reviewing kernel hyperparameters.",
            RuntimeWarning,
            stacklevel=2,
        )

    alpha = chol_solve(chol_lower, y_centered)

    return {
        "training_inputs": np.atleast_2d(x),
        "kernel": ker,
        "kernel_params": dict(kernel_params),
        "noise_variance": float(noise_variance),
        "chol_factor": chol_lower,
        "alpha": alpha,
        "target_mean": float(y_mean),
        "normalize": bool(normalize),
        "jitter": float(used_jitter),  # record actual jitter used
        "variance_floor": float(variance_floor),  # for downstream PSD projection
    }


def gp_predict(state: dict[str, Any], test_locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the Gaussian Process (GP) mean and variance at new input locations.

    This function uses the trained GP model produced by ``gp_fit`` to compute
    predictions at test points. It combines the correlations between the
    training and test inputs to estimate both the expected function value
    (mean) and the model's uncertainty (variance) at each test location.

    The computation follows the standard GP prediction rule: it uses the
    kernel function to measure similarity between training and test points,
    applies the precomputed Cholesky factor and weights from training,
    and produces the posterior mean and covariance. Numerical safeguards
    such as jitter addition, symmetrization, and positive-semidefinite
    projection are applied to ensure stability and valid output.

    In summary, this function takes the trained GP state and:
      1. Calculates the covariance between training and test inputs.
      2. Computes the predicted mean using the stored weights.
      3. Computes the predictive covariance and extracts its diagonal
         (marginal variances) for uncertainty estimates.
      4. Ensures the covariance matrix is symmetric and numerically stable.

    Args:
        state: Dictionary returned by `gp_fit` containing all GP model data.
        test_locations: Array of new input locations where predictions are made.
            If 1D, it is reshaped to two dimensions automatically.

    Returns:
        mean: Array of predictive mean values at the test locations.
        var: Array of predictive marginal variances (uncertainties).

    Raises:
        ValueError: If input arrays have incompatible shapes, invalid values,
            or the state dictionary is incomplete.
        TypeError: If the provided state or kernel object is invalid.
        RuntimeError: If matrix operations fail due to numerical issues.
    """
    if not isinstance(state, dict):
        raise TypeError("`state` must be a dict returned by `gp_fit`.")

    required_keys = {
        "training_inputs", "kernel", "kernel_params", "chol_factor", "alpha",
        "normalize", "jitter", "variance_floor",
    }
    missing = required_keys - set(state.keys())
    if missing:
        raise ValueError(f"`state` is missing required keys: {sorted(missing)}")

    x_train = np.asarray(state["training_inputs"])
    ker: Any = state["kernel"]
    theta = state["kernel_params"]
    chol_lower = np.asarray(state["chol_factor"])
    alpha = np.asarray(state["alpha"])

    if x_train.ndim != 2:
        raise ValueError(f"`training_inputs` must be 2D; got shape {x_train.shape}.")
    n_train = x_train.shape[0]

    if chol_lower.shape != (n_train, n_train):
        raise ValueError(
            f"`chol_factor` must be square {(n_train, n_train)}; got {chol_lower.shape}."
        )
    if alpha.shape != (n_train,):
        raise ValueError(f"`alpha` must have shape ({n_train},); got {alpha.shape}.")

    if not np.isfinite(x_train).all():
        raise ValueError("`training_inputs` contains NaN or Inf.")
    if not np.isfinite(alpha).all():
        raise ValueError("`alpha` contains NaN or Inf.")

    x_test = np.asarray(test_locations)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    if x_test.ndim != 2:
        raise ValueError(f"`test_locations` must be 2D after reshape; got shape {x_test.shape}.")
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train has {x_train.shape[1]} features, "
            f"but test has {x_test.shape[1]}."
        )
    if not np.isfinite(x_test).all():
        raise ValueError("`test_locations` contains NaN or Inf.")

    if not hasattr(ker, "k"):
        raise TypeError("`kernel` object in state must implement .k(x, y, params).")

    k_xt = ker.cov_value_value(x_train, x_test, theta)  # (n_train, m)
    if k_xt.shape[0] != n_train or k_xt.shape[1] != x_test.shape[0]:
        raise ValueError(
            f"kernel.k(train, test) returned shape {k_xt.shape}, expected {(n_train, x_test.shape[0])}."
        )
    if not np.isfinite(k_xt).all():
        raise ValueError("Cross-kernel matrix contains NaN/Inf.")

    mean = k_xt.T @ alpha  # (m,)

    try:
        v = solve(chol_lower, k_xt)  # (n_train, m)
    except LinAlgError as err:
        raise RuntimeError("Failed to solve L * v = K(X, x_*) with provided Cholesky factor.") from err

    k_tt = ker.cov_value_value(x_test, x_test, theta)  # (m, m)
    if k_tt.shape != (x_test.shape[0], x_test.shape[0]):
        raise ValueError(
            f"kernel.k(test, test) returned shape {k_tt.shape}, expected {(x_test.shape[0], x_test.shape[0])}."
        )
    if not np.isfinite(k_tt).all():
        raise ValueError("Test-test kernel matrix contains NaN/Inf.")

    raw_cov = k_tt - v.T @ v
    raw_cov = 0.5 * (raw_cov + raw_cov.T)

    # jitter / floor controls
    jitter = float(state.get("jitter", 1e-12))
    floor = float(state.get("variance_floor", 1e-18))
    if jitter < 0:
        raise ValueError("`state['jitter']` must be ≥ 0.")
    if floor < 0:
        raise ValueError("`state['variance_floor']` must be ≥ 0.")

    raw_cov = raw_cov + jitter * np.eye(raw_cov.shape[0])
    cov = to_psd(raw_cov, floor=floor)  # assumes helper exists
    var = np.diag(cov)
    if not np.isfinite(var).all():
        raise RuntimeError("Predicted variance contains NaN/Inf after PSD projection.")

    # de-standardize mean if needed
    if state.get("normalize", False):
        mean = mean + float(state.get("target_mean", 0.0))

    return mean.squeeze(), var


def gp_choose_hyperparams(
    training_inputs: np.ndarray,
    targets: np.ndarray,
    *,
    kernel: str | Kernel = "rbf",
    init_params: dict | None = None,
    init_noise: float = 1e-6,
    normalize: bool = True,
    n_restarts: int = 6,
    random_state: int = 123,
) -> Tuple[dict, float]:
    """Choose Gaussian Process (GP) hyperparameters by minimizing the negative log marginal likelihood (NLML).

    This function estimates kernel hyperparameters and the observation noise
    level directly from data. It sets sensible, data-driven bounds, performs
    optimization in log space using L-BFGS-B (Limited-memory BFGS) with multiple random restarts,
    and falls back to a small deterministic grid search if all optimizations
    fail. It supports both single length scale and ARD (Automatic Relevance Determination; array of per-dimension
    length scales). In the ARD case, length scales are treated as fixed inputs
    and only the output scale and noise level are optimized.

    In summary, the function:
      1. Validates and reshapes inputs as needed, and resolves the kernel.
      2. Derives initial values and bounds from the span of the first feature
         and the standard deviation of the targets.
      3. Optimizes the negative log marginal likelihood in log space using
         multiple restarts for robustness.
      4. If all restarts fail, runs a tiny grid search as a last resort.
      5. Returns the selected kernel parameters and noise variance.

    Args:
        training_inputs: Array with shape (n_samples, n_features) containing the
            training inputs. A 1D array is treated as (n_samples, 1).
        targets: Array with shape (n_samples,) containing the target values.
        kernel: Kernel name or a kernel instance. If a name is provided, it is
            resolved to a kernel object. Defaults to "rbf".
        init_params: Optional dictionary of initial kernel parameters. If it
            includes a length_scale array, ARD mode is assumed and length scales
            are held fixed during optimization.
        init_noise: Initial observation noise variance used to seed the search.
            Defaults to 1e-6.
        normalize: If True, the negative log marginal likelihood is evaluated
            with centered targets. Defaults to True.
        n_restarts: Number of random restarts for the optimizer, including the
            seeded start. Defaults to 6.
        random_state: Integer seed for reproducible random restarts. Defaults to 123.

    Returns:
        kernel_params: Dictionary of selected kernel hyperparameters. This always
            includes "output_scale" and "length_scale" (a float or an array for ARD).
        noise_variance: Float with the selected observation noise variance.

    Raises:
        ValueError: If input shapes do not match or contain invalid values.
        TypeError: If the kernel cannot be resolved from the provided identifier.
        RuntimeError: If the objective could not be evaluated for any candidate
            during both optimization and fallback grid search.

    Notes:
        - Bounds are derived from the data: the length scale range is based on
          the span of the first feature, and the output scale and noise ranges
          are based on the standard deviation of the targets.
        - Optimization is performed in log space with L-BFGS-B to enforce
          positivity and improve numerical stability.
        - In ARD mode, only output_scale and noise_variance are optimized; the
          provided length_scale array is used as-is.
    """
    rng = np.random.default_rng(random_state)
    x = np.atleast_2d(training_inputs)
    y = np.asarray(targets).reshape(-1)
    ker: Any = get_kernel(kernel) if isinstance(kernel, str) else kernel

    span = float(np.ptp(x[:, 0])) or 1.0
    amp = float(np.std(y) or 1.0)

    init_params = dict(init_params or {})
    length_scale_init = init_params.get("length_scale", 0.5 * span)
    output_scale_init = init_params.get("output_scale", amp)
    noise_init = float(init_noise)

    # tighter, safer bounds in log space
    def lb(v: float) -> float:
        return np.log(v)

    def ub(v: float) -> float:
        return np.log(v)

    # length_scale in [span/20, span*2] because we focus on local behavior
    length_scale_bounds = (lb(max(span / 20.0, 1e-6)), ub(max(span * 2.0, 1e-6)))
    # output_scale in [amp/100, 100*amp] because targets' stddev is a good guide
    output_scale_bounds = (lb(max(amp / 100.0, 1e-8)), ub(max(100.0 * amp, 1e-6)))
    # noise in [amp/1e6, amp/10] because noise is usually much smaller than signal
    noise_bounds = (lb(max(amp / 1e6, 1e-12)), ub(max(amp / 10.0, 1e-9)))

    is_ard = np.ndim(length_scale_init) > 0
    if is_ard:
        length_scale_fixed = np.asarray(length_scale_init, float).copy()

        def objective(u: np.ndarray) -> float:
            output_scale_val, noise_val = np.exp(u)
            params = {"length_scale": length_scale_fixed, "output_scale": float(output_scale_val)}
            return neg_log_marginal_likelihood(
                np.log([1.0, output_scale_val, noise_val]), x, y, ker, params, normalize
            )

        seed = np.array(
            [np.log(max(output_scale_init, 1e-8)), np.log(max(noise_init, 1e-12))]
        )
        bounds = [output_scale_bounds, noise_bounds]
        starts = [seed] + [
            np.array([rng.uniform(*output_scale_bounds), rng.uniform(*noise_bounds)])
            for _ in range(max(0, n_restarts - 1))
        ]
    else:
        def objective(u: np.ndarray) -> float:
            # u = [log(length_scale), log(output_scale), log(noise)]
            params = dict(init_params)
            return neg_log_marginal_likelihood(u, x, y, ker, params, normalize)

        seed = np.log(
            [
                max(float(length_scale_init), 1e-6),
                max(float(output_scale_init), 1e-8),
                max(noise_init, 1e-12),
            ]
        )
        bounds = [length_scale_bounds, output_scale_bounds, noise_bounds]
        starts = [seed] + [
            np.array(
                [
                    rng.uniform(*length_scale_bounds),
                    rng.uniform(*output_scale_bounds),
                    rng.uniform(*noise_bounds),
                ]
            )
            for _ in range(max(0, n_restarts - 1))
        ]

    best_val = np.inf
    best_u: np.ndarray | None = None

    for u0 in starts:
        res = minimize(
            objective, u0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "ftol": 1e-9}
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_u = res.x

    if best_u is None:
        # tiny deterministic grid fallback
        if is_ard:
            output_scale_grid = np.geomspace(max(amp / 20.0, 1e-6), max(20.0 * amp, 1e-3), 7)
            noise_grid = np.geomspace(max(amp / 1e6, 1e-12), max(amp / 10.0, 1e-9), 5)
            candidates = []
            for output_scale_val in output_scale_grid:
                for noise_val in noise_grid:
                    candidates.append(
                        (
                            neg_log_marginal_likelihood(
                                np.log([1.0, output_scale_val, noise_val]),
                                x,
                                y,
                                ker,
                                {"length_scale": length_scale_fixed, "output_scale": float(output_scale_val)},
                                normalize,
                            ),
                            output_scale_val,
                            noise_val,
                        )
                    )
            _, output_scale_out, noise_out = min(candidates, key=lambda t: t[0])
            return (
                {"length_scale": length_scale_fixed, "output_scale": float(output_scale_out)},
                float(noise_out),
            )
        length_scale_grid = np.geomspace(max(span / 20.0, 1e-6), max(span * 2.0, 1e-6), 9)
        output_scale_grid = np.geomspace(max(amp / 20.0, 1e-6), max(20.0 * amp, 1e-6), 9)
        noise_grid = np.geomspace(max(amp / 1e6, 1e-12), max(amp / 10.0, 1e-9), 5)
        candidates = []
        for length_scale_val in length_scale_grid:
            for output_scale_val in output_scale_grid:
                for noise_val in noise_grid:
                    candidates.append(
                        (
                            neg_log_marginal_likelihood(
                                np.log([length_scale_val, output_scale_val, noise_val]),
                                x,
                                y,
                                ker,
                                init_params,
                                normalize,
                            ),
                            length_scale_val,
                            output_scale_val,
                            noise_val,
                        )
                    )
        _, length_scale_out, output_scale_out, noise_out = min(candidates, key=lambda t: t[0])
        return (
            {"length_scale": float(length_scale_out), "output_scale": float(output_scale_out)},
            float(noise_out),
        )

    if is_ard:
        output_scale_out, noise_out = np.exp(best_u)
        return (
            {"length_scale": length_scale_fixed, "output_scale": float(output_scale_out)},
            float(noise_out),
        )

    length_scale_out, output_scale_out, noise_out = np.exp(best_u)
    return (
        {"length_scale": float(length_scale_out), "output_scale": float(output_scale_out)},
        float(noise_out),
    )


def gp_derivative(
    state: dict[str, Any],
    test_locations: np.ndarray,
    *,
    order: int = 1,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predicts the mean and variance of GP derivatives at new input locations.

    This function uses the trained GP state produced by `gp_fit`` to compute
    derivative predictions at test points. It supports first derivatives
    (order=1) and second derivatives along a single axis using the Hessian
    diagonal at the same point (order=2). The kernel must provide the
    corresponding cross-covariance routines for values versus derivatives.

    In summary, the function:
      1. Builds the cross-covariance between training values and test
         derivatives (order=1) or test second derivatives (order=2).
      2. Forms the derivative mean using the stored weights from training.
      3. Forms the derivative covariance by subtracting the projected term
         based on the Cholesky factor from training.
      4. Symmetrizes the covariance, adds a small jitter, and projects it to
         be positive semidefinite. The diagonal is returned as the marginal
         derivative variances.

    Args:
        state: Dictionary returned by `gp_fit` containing the GP model state.
            Must include:
              - "training_inputs": training inputs as a 2D array.
              - "kernel": kernel object with derivative methods
                (.cov_value_grad, .cov_grad_grad, .cov_value_hessdiag,
                 .cov_hessdiag_samepoint).
              - "kernel_params": kernel hyperparameters.
              - "chol_factor": Cholesky factor from training.
              - "alpha": weights from training solves.
              - "jitter": nonnegative jitter used for stabilization.
              - "variance_floor": nonnegative eigenvalue floor for PSD projection.
        test_locations: Array of new input locations where derivatives are
            predicted. A 1D array is treated as (n_points, 1).
        order: Derivative order. Use 1 for first derivatives, 2 for second
            derivatives using the same-point Hessian diagonal. Only 1 or 2
            is supported.
        axis: Input dimension along which the derivative is taken.

    Returns:
        mean: Array of derivative means at the test locations.
        var: Array of marginal variances of the derivative predictions.

    Raises:
        NotImplementedError: If `order` is not 1 or 2.
        TypeError: If the kernel object does not implement the required
            derivative methods.
        ValueError: If input shapes are inconsistent or contain invalid values.
        RuntimeError: If linear solves or stabilization steps fail due to
            numerical issues.

    Notes:
        - Normalization applied during fitting affects only the target mean and
          does not change derivative scaling. No de-normalization is applied
          to derivative outputs.
        - The second-derivative case uses the Hessian diagonal along a single
          axis at each test point.
    """
    x_train = state["training_inputs"]
    ker: Any = state["kernel"]
    theta = state["kernel_params"]
    chol_lower = state["chol_factor"]
    alpha = state["alpha"]
    var_floor = float(state.get("variance_floor", 1e-18))
    jitter = float(state.get("jitter", 1e-12))

    x_test = np.asarray(test_locations)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    if x_test.ndim != 2:
        raise ValueError(f"`test_locations` must be 2D after reshape; got shape {x_test.shape}.")
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train has {x_train.shape[1]} features, "
            f"but test has {x_test.shape[1]}."
        )

    if order == 1:
        # mean: k_{f, f'_*} K^{-1} y
        k_cross = ker.cov_value_grad(x_train, x_test, theta, axis=axis)  # (n, m)
        mean = k_cross.T @ alpha  # (m,)
        # cov: k_{f'_* f'_*} - k_{f'_* f} K^{-1} k_{f f'_*}
        v = solve(chol_lower, k_cross)  # (n, m)
        k_dd = ker.cov_grad_grad(x_test, x_test, theta, axis=axis)  # (m, m)
        raw_cov = k_dd - v.T @ v   # (m, m)
    elif order == 2:
        # mean: k_{f, f''_*} K^{-1} y
        k_cross = ker.cov_value_hessdiag(x_train, x_test, theta, axis=axis)  # (n, m)
        mean = k_cross.T @ alpha
        # cov: k_{f''_* f''_*} - k_{f''_* f} K^{-1} k_{f f''_*}
        v = solve(chol_lower, k_cross)
        k_dd = ker.cov_hessdiag_samepoint(x_test, theta, axis=axis)  # (m, m)
        raw_cov = k_dd - v.T @ v
    else:
        raise NotImplementedError("only order=1 or 2 supported")

    # robustify covariance: symmetrize + escalate jitter if needed, then PSD-project
    raw_cov = 0.5 * (raw_cov + raw_cov.T)
    eye_mat = np.eye(raw_cov.shape[0], dtype=raw_cov.dtype)
    try:
        _ = cholesky(raw_cov + jitter * eye_mat)
    except LinAlgError:
        jitter = max(jitter, 1e-8)
        try:
            _ = cholesky(raw_cov + jitter * eye_mat)
        except LinAlgError:
            jitter = max(jitter, 1e-6)
            _ = cholesky(raw_cov + jitter * eye_mat)
    raw_cov = raw_cov + jitter * eye_mat

    cov = to_psd(raw_cov, floor=var_floor)
    var = np.clip(np.diag(cov), var_floor, np.inf)

    return mean.squeeze(), var
