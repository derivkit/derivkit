"""Gaussian Process derivative estimation API."""

from __future__ import annotations
import numpy as np
from .core import gp_fit, gp_derivative, gp_choose_hyperparams

class GaussianProcess:
    """
    API aligned with your AdaptiveFitDerivative/FiniteDifferenceDerivative:
      gp = GaussianProcess(func, x0, ell=..., amp=..., ...)
      d  = gp.differentiate(order=1, samples=..., return_variance=False)

    Minimal defaults:
      - If `samples` is not provided, we build a tiny symmetric grid around x0.
      - Optional normalization and hyperparam optimization.
    """

    def __init__(
        self,
        func,
        x0: float,
        *,
        ell: float = 1.0,
        amp: float = 1.0,
        noise_var: float = 1e-6,
        normalize: bool = True,
        optimize: bool = False,
    ):
        self.func = func
        self.x0 = float(x0)
        self.hp = {"ell": float(ell), "amp": float(amp)}
        self.noise_var = float(noise_var)
        self.normalize = bool(normalize)
        self.optimize = bool(optimize)
        self._state = None  # set after first fit

    @staticmethod
    def _default_grid(x0: float, half_width: float, n_points: int) -> np.ndarray:
        # symmetric 1D absolute grid as (n,1)
        return (np.linspace(-half_width, half_width, n_points) + x0)[:, None]

    @staticmethod
    def _resolve_half_width(x0: float, spacing: float | str, base_abs: float) -> float:
        if isinstance(spacing, (int, float)):
            return float(spacing)
        # very simple heuristic for "auto"
        scale = max(1.0, abs(x0))
        return max(base_abs, 0.25 * scale)

    def differentiate(
        self,
        order: int,
        *,
        samples: np.ndarray | None = None,
        n_points: int = 13,
        spacing: float | str = "auto",  # half-width if float; else simple auto heuristic
        base_abs: float = 0.5,
        axis: int = 0,
        return_variance: bool = False,
    ):
        """
        Estimate ∂^order f / ∂x_axis^order at self.x0 using a GP surrogate.

        Args:
            order: 1 or 2.
            samples: optional (n, d) training inputs. If None, we make a 1D grid around x0.
            n_points, spacing, base_abs: used only when samples=None.
            axis: derivative component (for 1D this is 0).
            return_variance: if True, returns (mean, var); else mean.

        Returns: float or (float, float)
        """
        if order not in (1, 2):
            raise NotImplementedError("Only order=1 or 2 supported.")

        # 1) training inputs X
        if samples is None:
            half_width = self._resolve_half_width(self.x0, spacing, base_abs)
            X = self._default_grid(self.x0, half_width, n_points)
        else:
            X = np.atleast_2d(samples)

        # 2) training targets y (evaluate once)
        # support 1D callable f(x) with scalar x
        if X.shape[1] == 1:
            y = np.array([self.func(float(xx[0])) for xx in X], dtype=float)
        else:
            y = np.array([self.func(xx) for xx in X], dtype=float)

        # 3) (optional) quick hyperparam tuning
        if self.optimize:
            hp_opt, noise_opt = gp_choose_hyperparams(
                X, y, init_hp=self.hp, init_noise=self.noise_var, normalize=self.normalize
            )
            self.hp, self.noise_var = hp_opt, noise_opt

        # 4) fit state & predict derivative at x0
        self._state = gp_fit(X, y, self.hp, self.noise_var, normalize=self.normalize)
        dmean, dvar = gp_derivative(self._state, np.atleast_2d([self.x0]), order=order, axis=axis)

        if return_variance:
            # dvar is a length-1 array from core; return scalar
            return float(np.squeeze(dmean)), float(np.squeeze(dvar))
        return float(np.squeeze(dmean))
