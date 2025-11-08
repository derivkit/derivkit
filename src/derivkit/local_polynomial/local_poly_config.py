"""Local polynomial-regression derivative estimator."""

from __future__ import annotations

from typing import Sequence


class LocalPolyConfig:
    """Configuration for LocalPolynomialDerivative."""

    def __init__(
        self,
        rel_steps=(0.01, 0.02, 0.04, 0.08),
        tol_rel: float = 0.01,
        tol_abs: float = 1e-10,
        min_samples: int = 9,
        max_trim: int = 10,
        max_degree: int = 7,
        center: bool = True,
    ):
        """
        Args:
            rel_steps:
                Symmetric relative offsets used to build the candidate sample set.
            tol_rel:
                Maximum allowed relative deviation for residuals in the accepted window.
            tol_abs:
                Absolute floor when computing relative errors near y ≈ 0.
            min_samples:
                Minimum samples allowed after trimming.
            max_trim:
                Maximum number of trimming iterations.
            max_degree:
                Maximum polynomial degree (in (x-x0)) to consider.
            center:
                If True, fit in powers of (x - x0). Recommended.
        """
        self.rel_steps = tuple(float(s) for s in rel_steps)
        self.tol_rel = float(tol_rel)
        self.tol_abs = float(tol_abs)
        self.min_samples = int(min_samples)
        self.max_trim = int(max_trim)
        self.max_degree = int(max_degree)
        self.center = bool(center)