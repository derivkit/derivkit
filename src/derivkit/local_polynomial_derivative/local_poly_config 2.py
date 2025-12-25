"""Configuration for the local polynomial regression baseline.

This config controls how :class:`LocalPolynomialDerivative` chooses
sample locations, fits the local polynomial, and decides whether the
fit is trustworthy enough to mark ``ok=True`` in diagnostics.
"""

from __future__ import annotations


class LocalPolyConfig:
    """Configuration for the local polynomial regression baseline.

    This config controls how :class:`LocalPolynomialDerivative` chooses
    sample locations, fits the local polynomial, and decides whether the
    fit is trustworthy enough to mark ``ok=True`` in diagnostics.
    """

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
        """Initialize configuration.

        Args:
            rel_steps:
                Symmetric relative offsets around ``x0`` used to build the
                default sample grid.

                - For ``x0 != 0`` the grid is
                  ``x = x0 * (1 ± rel_steps[i])``.
                - For ``x0 == 0`` the grid is
                  ``x = ± rel_steps[i]``.

                Values are deduplicated and sorted; must be a non-empty
                1D sequence.

            tol_rel:
                Relative residual tolerance used when deciding whether a
                sample row is consistent with the current polynomial fit.
                A row is flagged as "bad" if any component satisfies
                ``|y_fit - y| / max(|y|, tol_abs) > tol_rel``.
                Lower values make trimming more aggressive.

            tol_abs:
                Absolute floor in the residual normalization. Prevents
                division by very small ``|y|`` when computing relative
                errors. Used as
                ``denom = max(|y|, tol_abs)``.

            min_samples:
                Minimum number of sample points that must remain after
                trimming for a fit to be considered. Also used (together
                with ``max_degree``) to ensure the system is not
                underdetermined. If trimming would reduce the usable
                samples below this threshold, trimming stops.

            max_trim:
                Maximum number of trimming iterations. Each iteration may
                remove at most one point from each edge of the grid.
                Acts as a safety bound to avoid pathological loops on
                extremely noisy or adversarial data.

            max_degree:
                Maximum polynomial degree allowed for the local fit.
                The actual degree used in :meth:`differentiate` is
                ``min(max_degree, chosen_degree)`` where
                ``chosen_degree`` is usually ``max(order + 2, 3)`` or an
                explicit ``degree=`` passed by the caller.

            center:
                If ``True``, the polynomial is expressed in powers of
                ``(x - x0)``; derivatives at ``x0`` are then read off as
                ``k! * a_k``. If ``False``, the polynomial is in powers
                of ``x`` directly. Centering generally improves numerical
                stability and is recommended.

        """
        # Normalize to a sorted, deduplicated tuple of floats
        self.rel_steps = tuple(float(s) for s in rel_steps)

        self.tol_rel = tol_rel
        self.tol_abs = tol_abs
        self.min_samples = min_samples
        self.max_trim = max_trim
        self.max_degree = max_degree
        self.center = center
