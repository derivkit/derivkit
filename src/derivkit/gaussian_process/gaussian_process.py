"""Gaussian Process derivative estimation API."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from derivkit.gaussian_process.core import (
    gp_choose_hyperparams,
    gp_derivative,
    gp_fit,
)
from derivkit.gaussian_process.gp_utils import (
    clamp_params_for_span,
    default_grid,
    ensure_min_noise,
    resolve_half_width,
    span_width,
    standardize_xy,
    take_local_window,
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
        func,
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
        self.func = func
        self.x0 = float(x0)
        self.kernel = kernel if hasattr(kernel, "k") else get_kernel(kernel)

        default_params = {"length_scale": 1.0, "output_scale": 1.0}
        self.kernel_params = {**default_params, **(kernel_params or {})}
        validate_kernel_params(self.kernel, self.kernel_params)

        self.noise_var = float(noise_variance)
        self.normalize = bool(normalize)
        self.optimize = bool(optimize)
        self.jitter = float(jitter)
        self.variance_floor = float(variance_floor)
        self._state = None

        self._debug = {
            "rescued": False,
            "jitter_used": None,
            "hp_before": None,
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
            x_mat = default_grid(self.x0, half_width, n_points)
        else:
            x_mat = np.atleast_2d(
                samples.astype(float) if isinstance(samples, np.ndarray) else np.array(samples, dtype=float)
            )
            if x_mat.ndim != 2:
                raise ValueError("samples must be shape (n, d).")

        # 2) Evaluate training targets y (original scale)
        if x_mat.shape[1] == 1:
            y_vec = np.array([self.func(float(row[0])) for row in x_mat], dtype=float)
        else:
            y_vec = np.array([self.func(row) for row in x_mat], dtype=float)

        # quick finite-difference teacher (cheap sanity check)
        x_col = x_mat[:, 0]
        span_full = float(x_col.max() - x_col.min()) or 1.0
        h = 0.05 * span_full

        def _finite_diff1(x_center: float) -> float:
            return (self.func(x_center + h) - self.func(x_center - h)) / (2.0 * h)

        # 1st pass
        dmean, dvar = gp_fit_predict_window(
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
        )

        # sanity check and optional wider retry
        fd_estimate = _finite_diff1(self.x0)
        sigma = float(np.sqrt(max(dvar, 0.0)))
        suspicious = abs(dmean - fd_estimate) > max(5.0 * sigma, 0.05 * max(1.0, abs(fd_estimate)))

        if suspicious:
            wider_frac = max(0.45, local_frac_span * 1.75)
            dmean_wide, dvar_wide = gp_fit_predict_window(
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
                optimize=True,
            )
            if abs(dmean_wide - fd_estimate) < abs(dmean - fd_estimate):
                dmean, dvar = dmean_wide, dvar_wide

        return (dmean, dvar) if return_variance else dmean

    @property
    def fitted_state(self):
        """Returns a summarized, read-only snapshot of the most recent GP fit (or ``None`` if no fit has run).

        The snapshot includes a copy of the kernel hyperparameters used, the
        observation noise variance, and any available debug fields. Values are
        copied or cast to basic Python types so they’re safe to log, serialize,
        or compare across runs. This is intended for inspection and debugging—not
        for reconstructing internal factorized matrices or solver state.
        """
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
) -> Tuple[float, float]:
    """Fits a local GP on a window around x0 and predict the derivative at x0.

    The steps are:
      1) Take a local window of (x, y) around x0 with the given fraction of span.
      2) Standardize x and y in that window.
      3) Map base kernel params to standardized space (length_scale / sigma_x, output_scale = 1).
      4) Optionally optimize hyperparameters and noise via negative log-likelihood.
      5) Fit GP (Cholesky) and predict derivative at standardized x0.
      6) Un-standardize derivative to original units.

    Args:
        x_mat: Training inputs, shape (n, d).
        y_vec: Training targets, shape (n,).
        x0: Expansion point for derivative prediction.
        kernel: Kernel choice (string name or kernel object).
        base_kernel_params: Base kernel hyperparameters (original scale).
        noise_variance: Observation noise variance (original scale).
        jitter: Diagonal jitter for numerical stability.
        variance_floor: Minimum variance returned by the model.
        order: Derivative order (1 or 2).
        axis: Input dimension index for differentiation.
        frac: Fraction of total span to define the local window around x0.
        optimize: Whether to optimize hyperparameters in the local window.

    Returns:
      A tuple (dmean, dvar): derivative mean and variance at x0 (original units).
    """
    # 1) local window
    x_win, y_win = take_local_window(
        x_mat, y_vec, x0, frac_span=frac, min_pts=max(9, min(31, x_mat.shape[0]))
    )

    # 2) standardize
    x_std, y_std, mu_x, sigma_x, mu_y, sigma_y = standardize_xy(x_win, y_win)

    # 3) params -> standardized space
    kernel_params_std = dict(base_kernel_params)
    ls = kernel_params_std.get("length_scale", 1.0)
    kernel_params_std["length_scale"] = (
        np.asarray(ls, float) / sigma_x if np.ndim(ls) > 0 else float(ls) / sigma_x
    )
    kernel_params_std["output_scale"] = 1.0
    noise_var_std = ensure_min_noise(noise_variance / (sigma_y**2))

    # 4) optional tuning
    if optimize:
        hp_opt, noise_opt = gp_choose_hyperparams(
            x_std,
            y_std,
            kernel=kernel,
            init_params=kernel_params_std,
            init_noise=noise_var_std,
            normalize=False,
        )
        kernel_params_std, noise_var_std = dict(hp_opt), float(noise_opt)

    # 4b) clamp length_scale to a sane range relative to *standardized* window span
    span_std = span_width(x_std[:, 0])
    kernel_params_std = clamp_params_for_span(kernel_params_std, span_std)

    # 5) fit and derivative prediction (standardized)
    state = gp_fit(
        x_std,
        y_std,
        kernel,
        kernel_params_std,
        noise_var_std,
        normalize=False,
        jitter=jitter,
        variance_floor=variance_floor,
    )
    x_query_std = np.array([[(x0 - mu_x) / sigma_x]], dtype=float)
    dmean_std, dvar_std = gp_derivative(state, x_query_std, order=order, axis=axis)

    # 6) un-standardize: dy/dx = (sigma_y / sigma_x) * d(y')/d(x')
    scale = sigma_y / sigma_x
    dmean = float(np.squeeze(dmean_std)) * scale
    dvar = float(np.squeeze(dvar_std)) * (scale**2)
    return dmean, dvar
