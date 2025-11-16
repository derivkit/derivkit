"""unit tests for local polynomial derivative fitting utilities."""

import numpy as np
from numpy.testing import assert_allclose

from derivkit.local_polynomial_derivative.fit import (
    centered_polyfit_least_squares,
    design_matrix,
    trimmed_polyfit,
)


class DummyConfig:
    """Minimal config with just the attributes used by the fit utilities."""

    def __init__(
        self,
        *,
        center: bool = True,
        min_samples: int = 3,
        max_trim: int = 5,
        tol_abs: float = 1e-12,
        tol_rel: float = 1e-8,
    ):
        """Initializes DummyConfig with given attributes."""
        self.center = center
        self.min_samples = min_samples
        self.max_trim = max_trim
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel


def test_design_matrix_centered_matches_manual_vandermonde():
    """Tests that design_matrix with centering produces expected results."""
    x0 = 1.0
    xs = np.array([0.5, 1.0, 1.5])
    degree = 2
    cfg = DummyConfig(center=True)

    mat = design_matrix(x0, cfg, xs, degree)

    z = xs - x0
    expected = np.vstack([[1.0, zi, zi**2] for zi in z])

    assert_allclose(mat, expected, atol=0.0, rtol=0.0)


def test_design_matrix_uncentered_uses_raw_coordinates():
    """Tests that design_matrix without centering produces expected results."""
    x0 = 1.0
    xs = np.array([0.5, 1.0, 1.5])
    degree = 2
    cfg = DummyConfig(center=False)

    mat = design_matrix(x0, cfg, xs, degree)

    z = xs
    expected = np.vstack([[1.0, zi, zi**2] for zi in z])

    assert_allclose(mat, expected, atol=0.0, rtol=0.0)


def test_centered_polyfit_least_squares_recovers_scalar_polynomial():
    """Tests that centered_polyfit_least_squares recovers known polynomial coefficients."""
    # y = 2 + 3 (x - x0) - (x - x0)^2
    x0 = 0.5
    xs = np.linspace(0.0, 1.0, 10)
    t = xs - x0

    coeff_true = np.array([2.0, 3.0, -1.0])
    ys = coeff_true[0] + coeff_true[1] * t + coeff_true[2] * t**2

    coeffs, used_mask = centered_polyfit_least_squares(x0, xs, ys, degree=2)

    # All samples should be used
    assert used_mask.shape == xs.shape
    assert used_mask.all()

    # One output component -> coeffs[:, 0] should match coeff_true
    assert_allclose(coeffs[:, 0], coeff_true, rtol=1e-12, atol=1e-12)


def test_centered_polyfit_least_squares_handles_vector_valued_output():
    """Tests that centered_polyfit_least_squares works with multi-component outputs."""
    # Component 0: y = x
    # Component 1: y = x^2
    x0 = 0.0
    xs = np.linspace(-1.0, 1.0, 9)
    ys = np.column_stack([xs, xs**2])

    coeffs, used_mask = centered_polyfit_least_squares(x0, xs, ys, degree=2)

    assert used_mask.all()
    assert coeffs.shape == (3, 2)

    # In powers of (x - x0) with x0 = 0:
    # y = x         -> [0, 1, 0]
    # y = x^2       -> [0, 0, 1]
    assert_allclose(coeffs[:, 0], [0.0, 1.0, 0.0], atol=1e-12, rtol=0.0)
    assert_allclose(coeffs[:, 1], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0)


def _eval_poly(coeffs: np.ndarray, x0: float, xs: np.ndarray, center: bool) -> np.ndarray:
    """Helper: evaluate polynomial with coefficients in powers of (x - x0) or x."""
    xs = np.asarray(xs, dtype=float)
    t = xs - x0 if center else xs
    vander = np.vander(t, N=coeffs.shape[0], increasing=True)
    return vander @ coeffs


def test_trimmed_polyfit_perfect_data_no_trimming():
    """Tests that trimmed_polyfit recovers polynomial on perfect data without trimming."""
    x0 = 0.0
    cfg = DummyConfig(
        center=True,
        min_samples=3,
        max_trim=5,
        tol_abs=1e-12,
        tol_rel=1e-6,
    )

    xs = np.linspace(-1.0, 1.0, 7)
    # y = 1 + 0.5 t + 2 t^2
    true_coeffs = np.array([[1.0], [0.5], [2.0]])
    ys = _eval_poly(true_coeffs, x0, xs, center=True)

    coeffs, used_mask, ok = trimmed_polyfit(x0, cfg, xs, ys, degree=2)

    assert ok
    assert used_mask.all()
    assert coeffs.shape == true_coeffs.shape
    assert_allclose(coeffs, true_coeffs, rtol=1e-10, atol=1e-10)


def test_trimmed_polyfit_trims_noisy_edges_and_recovers_polynomial():
    """Tests that trimmed_polyfit removes outliers at edges and recovers polynomial."""
    x0 = 0.0
    cfg = DummyConfig(
        center=True,
        min_samples=3,
        max_trim=10,
        tol_abs=1e-6,
        tol_rel=1e-3,
    )

    xs = np.linspace(-1.0, 1.0, 9)
    # y = t^2 with t = x - x0
    true_coeffs = np.array([[0.0], [0.0], [1.0]])
    ys = _eval_poly(true_coeffs, x0, xs, center=True)

    # Introduce large outliers at both ends
    ys[0, 0] = 10.0
    ys[-1, 0] = -10.0

    coeffs, used_mask, ok = trimmed_polyfit(x0, cfg, xs, ys, degree=2)

    # Expect edges to be dropped, interior kept
    assert not used_mask[0]
    assert not used_mask[-1]
    assert used_mask[1:-1].all()
    assert ok

    # Recovered polynomial should be close to y = t^2
    assert_allclose(coeffs[:, 0], [0.0, 0.0, 1.0], atol=1e-2, rtol=0.0)


def test_trimmed_polyfit_stops_when_min_samples_prevent_further_trimming():
    """Tests that trimmed_polyfit stops trimming when min_samples reached."""
    x0 = 0.0
    cfg = DummyConfig(
        center=True,
        min_samples=4,
        max_trim=1,  # only one trimming step allowed
        tol_abs=1e-6,
        tol_rel=1e-3,
    )

    xs = np.linspace(-1.0, 1.0, 5)
    # y = t with t = x - x0
    true_coeffs = np.array([[0.0], [1.0]])
    ys = _eval_poly(true_coeffs, x0, xs, center=True)

    # Add a big interior outlier (cannot be shaved away by edge trimming)
    ys[2, 0] = 10.0

    coeffs, used_mask, ok = trimmed_polyfit(x0, cfg, xs, ys, degree=1)

    # We should still have at least min_samples points, but tolerances not satisfied
    assert used_mask.sum() >= cfg.min_samples
    assert not ok
    # Coeffs are not checked here – behaviour is “best effort” under constraints.


def test_trimmed_polyfit_returns_zero_coeffs_if_never_fits():
    """Tests that trimmed_polyfit returns zero coeffs if fitting never succeeds."""
    x0 = 0.0
    cfg = DummyConfig(
        center=True,
        min_samples=10,
        max_trim=0,
        tol_abs=1e-6,
        tol_rel=1e-3,
    )

    xs = np.linspace(-1.0, 1.0, 5)  # fewer than min_samples
    ys = np.ones((xs.size, 1))

    coeffs, used_mask, ok = trimmed_polyfit(x0, cfg, xs, ys, degree=2)

    assert not ok
    assert coeffs.shape == (3, 1)    # degree + 1 rows
    assert np.all(coeffs == 0.0)
    # Mask should still reflect “all points considered”
    assert used_mask.shape == xs.shape
    assert used_mask.all()
