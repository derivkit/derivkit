"""Gaussian Process derivative estimation API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple

import numpy as np

from derivkit.gaussian_process.core import (
    gp_choose_hyperparams,
    gp_derivative,
    gp_fit,
)
from derivkit.gaussian_process.gp_diagnostics import make_gp_diag
from derivkit.gaussian_process.gp_utils import (
    TeacherKind,
    clamp_params_for_span,
    default_grid,
    ensure_min_noise,
    resolve_half_width,
    span_width,
    standardize_xy,
    swap_axis_first,
    take_local_window,
    teacher_derivative,
)
from derivkit.gaussian_process.kernels import (
    get_kernel,
    validate_kernel_params,
)


class GaussianProcess:
    """One-dimensional Gaussian Process surrogate centered at a point.

    This class builds a lightweight GP model around a scalar function near a
    chosen expansion point. You can select a built-in kernel by name (e.g.,
    `"rbf"`) or provide a custom kernel object. Sensible defaults are applied
    so you can get started with minimal configuration, while parameters like
    noise handling and small-number stabilizers are still accessible.

    Args:
      func: Callable that maps a single float to a float. This is the function
        the GP will model locally around ``x0``.
      x0: The expansion point at which the GP is anchored. Useful for methods
        that sample symmetrically around a reference location.
      kernel: Kernel choice. Either a string name of a built-in kernel
        (e.g., ``"rbf"``), or a kernel object that exposes a ``.k(x, x', **params)``
        method. If a string is given, it is resolved via ``get_kernel``.
      kernel_params: Optional dictionary of kernel hyperparameters. By default
        uses ``{"length_scale": 1.0, "output_scale": 1.0}``. Any provided keys
        override these defaults. Parameters are validated with
        ``validate_kernel_params``.
      noise_variance: Observation noise variance assumed when fitting the GP.
        Set close to zero for noiseless functions, or larger if evaluations are
        noisy. Must be non-negative.
      normalize: If ``True``, the target values are standardized internally
        (zero mean, unit variance) during fitting and de-standardized for
        predictions. This often improves numerical stability.
      optimize: If ``True``, hyperparameters may be tuned by an internal
        optimizer (if implemented by the caller). When ``False``, supplied
        ``kernel_params`` are used as-is.
      jitter: Small positive number added to the diagonal of covariance
        matrices for numerical stability (especially during Cholesky
        decomposition). Has no modeling meaning; purely a stabilizer.
      variance_floor: Lower bound applied to predicted variances to avoid
        returning exactly zero uncertainty due to round-off. Also a numerical
        stabilizer.

    Attributes:
      func: The target function.
      x0: The expansion point as a float.
      kernel: The resolved kernel object (has a ``.k`` method).
      kernel_params: Dictionary of kernel hyperparameters in effect.
      noise_var: Observation noise variance as a float.
      normalize: Whether standardization is enabled.
      optimize: Whether hyperparameter optimization is requested.
      jitter: Diagonal jitter used for linear algebra.
      variance_floor: Minimum variance returned by the model.
      _state: Internal cache for fitted quantities (e.g., training inputs,
        factorized covariance). Managed by methods that perform fitting or
        prediction.

    Raises:
      ValueError: If kernel parameters fail validation or an unknown kernel
        name is provided.
      TypeError: If ``func`` is not callable or the provided kernel object
        lacks a ``.k`` method.

    Examples:
      Basic construction with a built-in kernel name:

      >>> import numpy as np
      >>> gp = GaussianProcess(
      ...     func=np.sin,
      ...     x0=0.0,
      ...     kernel="rbf",
      ...     kernel_params={"length_scale": 0.4, "output_scale": 0.25},
      ...     noise_variance=1e-8,
      ...     normalize=True,
      ... )

      Using a custom kernel object:

      >>> class MyKernel:
      ...     def k(self, x, xp, *, length_scale=1.0, output_scale=1.0):
      ...         r2 = ((x - xp) / length_scale) ** 2
      ...         return output_scale * np.exp(-0.5 * r2)
      >>> gp = GaussianProcess(func=np.cos, x0=0.5, kernel=MyKernel())

    Notes:
      - This class focuses on clarity and stability for local modeling. Small
        stabilizers like ``jitter`` and ``variance_floor`` are deliberate.
      - If ``optimize=True`` is used, you are expected to provide or call a
        method that actually performs the optimization; this constructor only
        records your preference.
    """
    def __init__(
        self,
        func: Callable[..., float],
        x0: float,
        *,
        kernel: str | object = "rbf",
        kernel_params: dict | None = None,
        noise_variance: float = 1e-6,
        normalize: bool = True,
        optimize: bool = False,
        jitter: float = 1e-12,
        variance_floor: float = 1e-18,
    ):
        """Initializes the GaussianProcess instance.

        Args:
            func: Callable mapping a float (or array-like) to a float.
            x0: Expansion point as a float.
            kernel: Kernel name or object with a ``.k`` method.
            kernel_params: Optional dictionary of kernel hyperparameters.
            noise_variance: Non-negative observation noise variance.
            normalize: Whether to standardize targets during fitting.
            optimize: Whether to enable hyperparameter optimization.
            jitter: Small positive stabilizer added to covariance diagonals.
            variance_floor: Minimum predictive variance returned by the model.

        Raises:
            ValueError: If kernel parameters are invalid or noise_variance is negative.
            TypeError: If func is not callable or kernel lacks a .k method.
        """
        if not callable(func):
            raise TypeError("func must be callable (float -> float or array-like -> float).")

        self.func = func
        self.x0 = float(x0)
        self.kernel = kernel if hasattr(kernel, "k") else get_kernel(kernel)

        default_params = {"length_scale": 1.0, "output_scale": 1.0}
        self.kernel_params = {**default_params, **(kernel_params or {})}
        validate_kernel_params(self.kernel, self.kernel_params)

        if noise_variance < 0:
            raise ValueError("noise_variance must be non-negative.")
        self.noise_var = float(noise_variance)

        self.normalize = bool(normalize)
        self.optimize = bool(optimize)
        self.jitter = float(jitter)
        self.variance_floor = float(variance_floor)
        self._state: Optional[dict] = None

        self._debug: Dict[str, Any] = {
            "rescued": False,
            "jitter_used": None,
            "hp_before": dict(self.kernel_params),
            "hp_after_opt": None,
            "hp_after_clamp": None,
            "noise_opt": None,
            "dvar_ratio": None,
        }

    def differentiate(
            self,
            order: int,
            *,
            samples: np.ndarray | None = None,
            n_points: int = 13,
            spacing: float | str = "auto",
            base_abs: float = 0.5,
            axis: int = 0,
            return_variance: bool = False,
            local_frac_span: float = 0.35,
            teacher: TeacherKind = "adaptive",
            teacher_kwargs: dict | None = None,
            return_diag: bool = False,
    ):
        """Estimates the first or second derivative at the expansion point.

        This method fits a local Gaussian Process to function evaluations near
        ``x0`` and returns the derivative of the requested order at ``x0``.
        You can either pass your own sample locations via ``samples`` or let the
        method build a small symmetric grid around ``x0``.

        The fit is done on a local window (a fraction of the total sampled span).
        As a safety check, the GP derivative is compared to a simple central
        finite-difference estimate at ``x0``. If they disagree strongly, the
        method retries once with a wider window and enables basic hyperparameter
        optimization if available.

        Args:
          order: Derivative order. Must be ``1`` (first) or ``2`` (second).
          samples: Optional sample locations. If provided, must be an array with
            shape ``(n, d)`` where the first column is the coordinate being
            differentiated. If omitted, a default 1D grid is generated.
          n_points: Number of points to generate when ``samples`` is not given.
            Must be at least 3.
          spacing: Controls the automatically generated grid width around ``x0``.
            Use ``"auto"`` for a reasonable choice based on ``x0`` and
            ``base_abs``; or pass a positive float to set the half-width directly.
          base_abs: Baseline absolute half-width used when ``spacing="auto"``.
          axis: Index of the input dimension to differentiate with respect to
            (for multi-dimensional inputs). Default is the first dimension.
          return_variance: If ``True``, also return the GP's variance estimate for
            the requested derivative at ``x0``.
          local_frac_span: Fraction of the full sampled span used to define the
            local window for fitting. Larger values use a wider neighborhood.
          teacher: Finite-difference teacher method used for consistency checking.
          teacher_kwargs: Optional dictionary of extra keyword arguments to pass
            to the teacher derivative function.
          return_diag: If ``True``, return a diagnostics dictionary along with the
            derivative estimate.

        Returns:
          float | tuple[float, float]: If ``return_variance`` is ``False``, returns
            the derivative estimate as a float. If ``True``, returns a tuple
            ``(mean, variance)`` where ``variance`` is the GP uncertainty for the
            requested derivative at ``x0``.

        Raises:
          NotImplementedError: If ``order`` is not 1 or 2.
          ValueError: If ``n_points < 3`` when auto-building a grid, if
            ``samples`` cannot be coerced to a 2D array of shape ``(n, d)``,
            or if invalid settings are detected.

        Notes:
          - When ``samples`` is provided for multi-dimensional inputs, only the
            column indicated by ``axis`` is differentiated; other columns are
            treated as fixed coordinates.
          - The internal finite-difference check is meant to catch poor local fits
            and has no effect when the GP estimate is already consistent.
        """
        if order not in (1, 2):
            raise NotImplementedError("Only order=1 or 2 supported.")
        if samples is None and n_points < 3:
            raise ValueError("n_points must be >= 3 when auto-building a grid.")

            # 1) Build training inputs x (original scale)
        if samples is None:
            half_width = resolve_half_width(self.x0, spacing, base_abs)
            x_mat = default_grid(self.x0, half_width, n_points)  # shape (n, 1)
        else:
            x_mat = np.atleast_2d(np.asarray(samples, dtype=float))
            if x_mat.ndim != 2:
                raise ValueError("samples must be a 2D array of shape (n, d).")

        n, d = x_mat.shape
        if not (0 <= axis < d):
            raise ValueError(f"axis={axis} is out of bounds for input dimension d={d}.")

        # 2) Evaluate training targets y (original scale)
        if d == 1:
            y_vec = np.array([float(self.func(float(row[0]))) for row in x_mat], dtype=float)
        else:
            y_vec = np.array([float(self.func(row)) for row in x_mat], dtype=float)

        # 3) Quick finite-difference teacher (central difference along `axis`)
        x_col = x_mat[:, axis]
        span_full = float(x_col.max() - x_col.min()) or 1.0
        h = 0.05 * span_full  # base step for teacher methods that need it

        # Get the finite-difference estimate at x0
        fd_estimate = teacher_derivative(
            self.func,
            x0=self.x0,
            x_mat=x_mat,
            axis=axis,
            teacher=teacher,
            h=h,
            teacher_kwargs=teacher_kwargs
        )

        # 4) First-pass GP fit/predict on a local window
        dmean, dvar, diag1 = gp_fit_predict_window(
            x_mat=x_mat,
            y_vec=y_vec,
            x0=self.x0,
            kernel=self.kernel,
            base_kernel_params=self.kernel_params,
            noise_variance=self.noise_var,
            jitter=self.jitter,
            variance_floor=self.variance_floor,
            order=order,
            axis=axis,
            frac=local_frac_span,
            optimize=self.optimize,
            want_diag=return_diag,
            diag_extras={
                "spacing": spacing,
                "n_points": int(x_mat.shape[0]),
                "base_abs": base_abs,
                "fd_estimate": fd_estimate,
            },
        )

        # 5) Sanity check vs finite-difference; optional wider retry with tuning
        sigma = float(np.sqrt(max(dvar, 0.0)))
        suspicious = abs(dmean - fd_estimate) > max(5.0 * sigma, 0.05 * max(1.0, abs(fd_estimate)))

        if suspicious:
            wider_frac = max(0.45, local_frac_span * 1.75)
            dmean_w, dvar_w, diag2 = gp_fit_predict_window(
                x_mat=x_mat,
                y_vec=y_vec,
                x0=self.x0,
                kernel=self.kernel,
                base_kernel_params=self.kernel_params,
                noise_variance=self.noise_var,
                jitter=self.jitter,
                variance_floor=self.variance_floor,
                order=order,
                axis=axis,
                frac=wider_frac,
                optimize=True,  # allow local hyperparam tuning on retry
                want_diag=return_diag,
                diag_extras={
                    "spacing": spacing,
                    "n_points": int(x_mat.shape[0]),
                    "base_abs": base_abs,
                    "fd_estimate": fd_estimate,
                },
            )
            if abs(dmean_w - fd_estimate) < abs(dmean - fd_estimate):
                dmean, dvar, diag1 = dmean_w, dvar_w, diag2  # prefer improved fit

        if return_diag:
            return (dmean, dvar, diag1) if return_variance else (dmean, diag1)
        return (dmean, dvar) if return_variance else dmean


    @property
    def fitted_state(self):
        """Summarized, read-only snapshot of the most recent GP fit (or None)."""
        if self._state is None:
            return None
        out = {
            "kernel_params": dict(self._state.get("kernel_params", {})),
            "noise_variance": float(self._state.get("noise_variance", np.nan)),
        }
        out.update(self._debug)
        return out


def gp_fit_predict_window(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    *,
    x0: float,
    kernel: str | Any,
    base_kernel_params: dict,
    noise_variance: float,
    jitter: float,
    variance_floor: float,
    order: int,
    axis: int,
    frac: float,
    optimize: bool,
    want_diag: bool = False,
    diag_extras: Optional[dict] = None,
) -> Tuple[float, float, Optional[dict]]:
    """"Fits a local Gaussian Process around ``x0`` and returns the derivative at ``x0``.

    The routine extracts a window of training points around ``x0``, standardizes
    inputs/targets, maps kernel hyperparameters to standardized space, optionally
    optimizes them, clamps the length scale to the window span, fits the GP, and
    predicts the requested derivative at ``x0``. The result is then un-standardized
    back to original units. Optionally, a diagnostics dictionary is returned.

    Args:
      x_mat: Training inputs of shape ``(n, d)``.
      y_vec: Training targets of shape ``(n,)``.
      x0: Expansion point at which the derivative is predicted (original units).
      kernel: Kernel identifier (name or kernel object).
      base_kernel_params: Kernel hyperparameters on the original scale
        (e.g., ``{"length_scale": ..., "output_scale": ...}``).
      noise_variance: Observation noise variance on the original target scale.
      jitter: Diagonal jitter added to the kernel matrix for numerical stability.
      variance_floor: Minimum predictive variance to guard against zero-variance
        due to round-off.
      order: Derivative order; must be ``1`` (first) or ``2`` (second).
      axis: Column index in ``x_mat`` to differentiate with respect to.
      frac: Fraction of the full sampled span used to form the local window.
      optimize: If ``True``, perform a local hyperparameter (and noise) optimization
        in standardized space via marginal likelihood.
      want_diag: If ``True``, also assemble and return a diagnostics dictionary.
      diag_extras: Optional fields to forward into the diagnostics payload
        (e.g., ``{"spacing": ..., "n_points": ..., "base_abs": ..., "fd_estimate": ...}``).

    Returns:
      Tuple[float, float, Optional[dict]]:
        - ``dmean``: Derivative mean at ``x0`` (original units).
        - ``dvar``: Derivative variance at ``x0`` (original units).
        - ``diag``: Diagnostics dictionary if ``want_diag=True``; otherwise ``None``.

    Raises:
      NotImplementedError: If ``order`` is not ``1`` or ``2``.
      ValueError: If inputs are malformed (e.g., axis out of bounds, shapes invalid).

    Notes:
      - Internally, the target differentiation axis is temporarily moved to column 0
        to satisfy helpers that assume the target dimension is first.
      - Standardization uses the window statistics; hyperparameters (especially
        ``length_scale``) and noise are mapped to this space before fitting.
      - A length-scale clamp relative to the standardized window span improves
        robustness on tiny or highly concentrated windows.
      - When ``want_diag=True``, the diagnostics include window contents, standardization
        stats, pre/post-optimization hyperparameters, kernel condition estimates,
        and the standardized vs. un-standardized derivative numbers.
    """
    # 0) make target dimension be column 0 for helpers that assume that
    x_swap, inv = swap_axis_first(x_mat, axis)

    # 1) local window (helpers expect target dim = col 0; no `dim=` argument)
    x_win_swap, y_win = take_local_window(
        x_swap, y_vec, x0, frac_span=frac, min_pts=max(9, min(31, x_mat.shape[0]))
    )

    # 2) standardize (same assumption: target dim = col 0)
    x_std_swap, y_std, mu_x, sigma_x, mu_y, sigma_y = standardize_xy(x_win_swap, y_win)

    # 3) map kernel params to standardized space
    kernel_params_std = dict(base_kernel_params)
    ls = kernel_params_std.get("length_scale", 1.0)
    kernel_params_std["length_scale"] = (float(ls) / sigma_x) if np.ndim(ls) == 0 else (np.asarray(ls, float) / sigma_x)
    kernel_params_std["output_scale"] = 1.0
    noise_var_std = ensure_min_noise(noise_variance / (sigma_y**2))

    hp_opt = None
    if optimize:
        hp_opt, noise_opt = gp_choose_hyperparams(
            x_std_swap,
            y_std,
            kernel=kernel,
            init_params=kernel_params_std,
            init_noise=noise_var_std,
            normalize=False,
        )
        kernel_params_std, noise_var_std = dict(hp_opt), float(noise_opt)

    # 4) clamp length_scale by standardized span along target dim (col 0)
    span_std = span_width(x_std_swap[:, 0])
    kernel_params_std = clamp_params_for_span(kernel_params_std, span_std)

    # 5) fit and derivative prediction (standardized)
    state = gp_fit(
        x_std_swap,
        y_std,
        kernel,
        kernel_params_std,
        noise_var_std,
        normalize=False,
        jitter=jitter,
        variance_floor=variance_floor,
    )

    x_query_std = np.zeros((1, x_std_swap.shape[1]), dtype=float)
    x_query_std[0, 0] = (x0 - mu_x) / sigma_x  # target dim is col 0 after swap

    dmean_std, dvar_std = gp_derivative(state, x_query_std, order=order, axis=0)

    # 6) un-standardize back to original units
    if order == 1:
        scale = sigma_y / sigma_x
    elif order == 2:
        scale = sigma_y / (sigma_x**2)
    else:
        raise NotImplementedError("Only order=1 or 2 supported.")

    dmean = float(np.squeeze(dmean_std)) * scale
    dvar = float(np.squeeze(dvar_std)) * (scale**2)

    diag: Optional[dict] = None
    if want_diag:
        try:
            ker = state["kernel"]
            theta = state["kernel_params"]
            xs = state["training_inputs"]  # standardized inputs (swapped layout)
            cov = ker.cov_value_value(xs, xs, theta) + (state["noise_variance"] + state["jitter"]) * np.eye(len(xs))
            eig = np.linalg.eigvalsh(cov)
            cov_min, cov_max = float(eig.min()), float(eig.max())
            cov_cond = cov_max / max(cov_min, 1e-300)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            cov_min = cov_max = cov_cond = float("nan")

        # prepare originals for the diag (map swapped arrays back to original layout)
        x_win = x_win_swap[:, inv]
        extras = diag_extras or {}

        diag = make_gp_diag(
            x0=x0,
            order=order,
            axis=axis,
            x_full=x_mat[:, axis],
            y_full=y_vec,
            x_win=x_win[:, axis],
            y_win=y_win,
            mu_x=mu_x,
            sigma_x=sigma_x,
            mu_y=mu_y,
            sigma_y=sigma_y,
            kernel_name=(getattr(kernel, "name", None) or str(kernel)),
            kernel_params_before=base_kernel_params,
            kernel_params_after_opt=(hp_opt if optimize else None),
            kernel_params_after_clamp=kernel_params_std,
            noise_variance_before=noise_variance,
            noise_variance_after_opt=(state.get("noise_variance") if optimize else None),
            jitter_used=state["jitter"],
            variance_floor=state["variance_floor"],
            optimize=optimize,
            frac_window=frac,
            spacing=extras.get("spacing"),
            n_points=int(extras.get("n_points", x_mat.shape[0])),
            base_abs=extras.get("base_abs"),
            state=state,
            fd_estimate=extras.get("fd_estimate"),
            dmean_std=float(np.squeeze(dmean_std)),
            dvar_std=float(np.squeeze(dvar_std)),
            dmean=dmean,
            dvar=dvar,
            suspicious=None,
            kernel_cond=cov_cond,
            kernel_min_eig=cov_min,
            kernel_max_eig=cov_max,
        )

    return dmean, dvar, diag
