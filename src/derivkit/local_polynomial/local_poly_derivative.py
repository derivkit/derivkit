#!/usr/bin/env python3
"""Local polynomial-regression derivative estimator.

This module provides a robust, trimmed local polynomial fit around x0
to estimate derivatives of arbitrary order (up to the fit degree).

It is designed as:
- a simple, self-contained baseline/fallback engine,
- vector-output safe,
- no finite-difference stencils,
- no magic global state.
"""

from __future__ import annotations

from typing import Callable, Any, Dict

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from derivkit.local_polynomial.local_poly_config import LocalPolyConfig



class LocalPolynomialDerivative:
    """Estimates derivatives via trimmed local polynomial regression around x0."""

    def __init__(
        self,
        func: Callable[[float], Any],
        x0: float,
        config: LocalPolyConfig | None = None,
    ):
        """Initializes the estimator."""
        self.func = func
        self.x0 = float(x0)
        self.config = config or LocalPolyConfig()

    def differentiate(
        self,
        order: int = 1,
        degree: int | None = None,
        n_workers: int = 1,
        diagnostics: bool = False,
    ):
        """Returns the derivative of given order at x0.

        Args:
            order:
                Derivative order (>=1).
            degree:
                Polynomial degree to fit in (x-x0).
                If None, uses max(order + 2, 3) but capped by config.max_degree.
            n_workers:
                number of workers for function evaluation.
            diagnostics:
                If True, returns (derivative, diag_dict).

        Returns:
            derivative or (derivative, diagnostics)

            derivative:
                - scalar if func(x) is scalar,
                - np.ndarray (n_comp,) if func(x) is vector-valued.

        Raises:
            ValueError: if order < 1 or feasible fit is impossible.
        """
        if order < 1:
            raise ValueError("order must be >= 1.")

        # Choose polynomial degree with a bit of headroom.
        if degree is None:
            degree = max(order + 2, 3)
        degree = int(min(degree, self.config.max_degree))
        if degree < order:
            raise ValueError("degree must be >= order.")

        xs, ys = self._build_samples(n_workers=n_workers)
        coeffs, used_mask, ok = self._trimmed_polyfit(xs, ys, degree)

        # If nothing worked, coeffs is best-effort; ok=False tells caller to treat with caution.
        # coeffs shape: (degree+1, n_comp); basis in powers of (x - x0) if center=True.
        n_comp = coeffs.shape[1]

        # k-th derivative at x0 from polynomial in (x-x0):
        #   p(z) = sum_{j=0}^deg a_j z^j  =>  p^{(k)}(0) = k! * a_k
        # (this assumes centering, which we enforce in _design_matrix if center=True)
        if order > degree:
            # should not happen with checks above
            raise ValueError("Internal error: order > degree in derivative extraction.")

        factorial = 1
        for k in range(2, order + 1):
            factorial *= k

        a_k = coeffs[order]  # (n_comp,)
        deriv = factorial * a_k
        deriv_out = float(deriv[0]) if n_comp == 1 else deriv

        if not diagnostics:
            return deriv_out

        diag = self._make_diag(xs, ys, used_mask, coeffs, degree, order, ok)
        diag["n_workers"] = int(n_workers)
        return deriv_out, diag

    # ---------- internal helpers ----------

    def _build_samples(self, n_workers: int = 1):
        """Builds symmetric sample points and evaluates func (vector-safe)."""
        rel_steps = np.asarray(self.config.rel_steps, float)
        if rel_steps.ndim != 1 or rel_steps.size == 0:
            raise ValueError("rel_steps must be a 1D non-empty sequence of floats.")

        if self.x0 == 0.0:
            xs = np.concatenate([-rel_steps, rel_steps])
        else:
            xs = self.x0 * (1.0 + np.concatenate([-rel_steps, rel_steps]))

        xs = np.unique(xs)
        xs.sort()

        # Evaluate function; ensure 2D (n_samples, n_comp)
        if n_workers == 1:
            ys_list = [np.atleast_1d(self.func(float(x))) for x in xs]
        else:
            # Thread-based parallelism to avoid pickling issues with closures.
            def _eval_one(x):
                return np.atleast_1d(self.func(float(x)))

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                ys_list = list(ex.map(_eval_one, xs))

        ys = np.stack(ys_list, axis=0)
        if ys.ndim != 2:
            ys = ys.reshape(ys.shape[0], -1)

        return xs, ys

    def _design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Builds Vandermonde-like matrix for polynomial in (x - x0) or x."""
        if self.config.center:
            z = x - self.x0
        else:
            z = x
        # Shape (n_samples, degree+1), columns z^0, z^1, ..., z^degree
        return np.vander(z, N=degree + 1, increasing=True)

    def _trimmed_polyfit(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        degree: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Runs trimmed polynomial regression.

        Returns:
            coeffs : (degree+1, n_comp)
            used_mask : (n_samples,) bool
            ok : bool (True if residuals within tolerances on final mask)
        """
        n_samples, n_comp = ys.shape
        keep = np.ones(n_samples, dtype=bool)
        n_trim = 0

        last_coeffs = None
        last_keep = keep.copy()
        last_ok = False

        needed = max(self.config.min_samples, degree + 1)

        while keep.sum() >= needed and n_trim <= self.config.max_trim:
            idx = np.where(keep)[0]
            x_use = xs[idx]
            y_use = ys[idx]

            matrix = self._design_matrix(x_use, degree)

            coeffs, *_ = np.linalg.lstsq(matrix, y_use, rcond=None)

            y_fit = matrix @ coeffs
            denom = np.maximum(np.abs(y_use), self.config.tol_abs)
            err = np.abs(y_fit - y_use) / denom

            bad_rows = (err > self.config.tol_rel).any(axis=1)

            if not bad_rows.any():
                last_coeffs = coeffs
                last_keep = keep.copy()
                last_ok = True
                break

            bad_idx_all = idx[bad_rows]
            leftmost, rightmost = idx[0], idx[-1]
            trimmed = False

            # shave edges only if we'll still have enough points for this degree
            if bad_idx_all[0] == leftmost and keep.sum() - 1 >= needed:
                keep[leftmost] = False
                trimmed = True
            if bad_idx_all[-1] == rightmost and keep.sum() - 1 >= needed:
                keep[rightmost] = False
                trimmed = True

            if not trimmed:
                # Interior badness; cannot fix by shaving edges only.
                last_coeffs = coeffs
                last_keep = keep.copy()
                last_ok = False
                break

            last_coeffs = coeffs
            last_keep = keep.copy()
            last_ok = False
            n_trim += 1

        if last_coeffs is None:
            last_coeffs = np.zeros((degree + 1, n_comp), dtype=float)
            last_keep = keep.copy()
            last_ok = False

        return last_coeffs, last_keep, last_ok


    def _make_diag(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        keep: np.ndarray,
        coeffs: np.ndarray,
        degree: int,
        order: int,
        ok: bool,
    ) -> Dict[str, Any]:
        """Build diagnostics dictionary."""
        used_x = xs[keep]
        used_y = ys[keep]
        matrix = self._design_matrix(used_x, degree)
        y_fit = matrix @ coeffs

        denom = np.maximum(np.abs(used_y), self.config.tol_abs)
        err = np.abs(y_fit - used_y) / denom
        max_err = float(err.max()) if err.size else float("nan")

        diag = {
            "ok": bool(ok),
            "x0": float(self.x0),
            "degree": int(degree),
            "order": int(order),
            "n_all": int(xs.size),
            "n_used": int(keep.sum()),
            "x_used": used_x.tolist(),
            "max_rel_err_used": max_err,
            "tol_rel": float(self.config.tol_rel),
            "tol_abs": float(self.config.tol_abs),
            "min_samples": int(self.config.min_samples),
            "max_trim": int(self.config.max_trim),
            "center": bool(self.config.center),
            "coeffs": coeffs.tolist(),  # small; fine for debugging
        }

        if not ok:
            diag["note"] = (
                "No interval fully satisfied residual tolerances; derivative is best-effort "
                "from the last polynomial fit and should be treated with caution."
            )

        return diag
