"""Tests for LikelihoodExpansion."""

import numpy as np
import pytest

from derivkit.forecasting.expansions import LikelihoodExpansion


def test_order():
    """High derivative orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        like._get_derivatives(order=3)
    with pytest.raises(ValueError):
        like._get_derivatives(order=np.random.randint(low=4, high=30))


def test_forecast_order():
    """High forecast orders (>2) should raise ValueError."""
    like = LikelihoodExpansion(lambda x: x, np.array([1]), np.array([1]))

    with pytest.raises(ValueError):
        like.get_forecast_tensors(forecast_order=3)

    with pytest.raises(ValueError):
        like.get_forecast_tensors(forecast_order=np.random.randint(low=4, high=30))


def test_pseudoinverse_path_no_nan():
    """If inversion fails, pseudoinverse path should still return finite numbers."""
    # singular covariance -> forces pinv path
    forecaster = LikelihoodExpansion(lambda x: x,
        np.array([1.0]),
        np.array([[0.0]]),
    )
    fisher = forecaster.get_forecast_tensors(forecast_order=1)
    assert np.isfinite(fisher).all()
    tensor_g, tensor_h = forecaster.get_forecast_tensors(forecast_order=2)
    assert np.isfinite(tensor_g).all()
    assert np.isfinite(tensor_h).all()


@pytest.mark.parametrize(
    (
        "model, "
        "fiducials, "
        "covariance_matrix, "
        "expected_fisher, "
        "expected_dali_g, "
        "expected_dali_h"
    ),
    [
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[1.03612509]]),
            np.array([[[0.49105455]]]),
            np.array([[[[0.23272727]]]])
        ),
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([1.1, 0.4]),
            np.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            np.array([
                [-0.00890115,  0.08901149],
                [ 0.10357701, -0.01177011]
            ]),
            np.array([
                [
                    [-8.09195402e-03,  8.09195402e-02],
                    [-1.46382975e-16, -2.08644092e-16]
                ],
                [
                    [-1.42668169e-16, -3.75546474e-16],
                    [ 2.58942529e-01, -2.94252874e-02]
                ]
            ]),
            np.array([
                [
                    [
                        [-7.35632184e-03, -1.11448615e-16],
                        [-1.06393661e-16,  2.02298851e-01]
                    ],
                    [
                        [-1.33075432e-16,  7.15511183e-31],
                        [ 1.01887953e-30, -5.21610229e-16]
                    ]
                ],
                [
                    [
                        [-1.29698336e-16,  9.78598871e-31],
                        [ 1.29608820e-30, -9.38866185e-16]
                    ],
                    [
                        [ 2.35402299e-01, -6.14828927e-16],
                        [-1.10097326e-15, -7.35632184e-02]
                    ]
                ]
            ])
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[0.01890366]]),
            np.array([[[-0.03087509]]]),
            np.array([[[[0.05042786]]]])
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([1.1, 0.4]),
            np.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            np.array([
                [-0.00414466, 0.07008167],
                [0.08154958, -0.01566948]
            ]),
            np.array([
                [
                    [ 7.89639690e-04, -1.33519460e-02],
                    [ 6.87814470e-15, -1.84147323e-15]
                ],
                [
                    [ 6.59290723e-17, -1.11478872e-15],
                    [ 1.71255860e-01, -3.29062482e-02]
                ]
            ]),
            np.array([
                [
                    [
                        [-1.50441995e-04, -1.12697772e-15],
                        [-1.25607936e-17, -2.80393711e-02]
                    ],
                    [
                        [-1.31042274e-15, -2.06221337e-28],
                        [-1.09410605e-28, -3.86713302e-15]
                    ]
                ],
                [
                    [
                        [-1.25607936e-17, -9.40943024e-29],
                        [-1.04873334e-30, -2.34108006e-15]
                    ],
                    [
                        [-3.26276318e-02, -4.04783117e-15],
                        [-2.72416588e-15, -6.91038222e-02]
                    ]
                ]
            ])
        ),
   ]
)
def test_forecast(
    model,
    fiducials,
    covariance_matrix,
    expected_fisher,
    expected_dali_g,
    expected_dali_h
):
    """Validate Fisher and DALI tensors against reference values.

    This test is parametrized over simple scalar/vector models and covariance
    matrices.

    - Fisher is compared with `atol=0` (after casting to float64).
    - DALI tensors (G, H) use a mixed tolerance: tight relative tolerance and a
      tiny absolute floor only where the expected entries are near zero. This
      avoids false failures from floating-point noise in ~0 entries while
      keeping real values strict.

    Args:
      model (Callable[[np.ndarray], np.ndarray | float]): Observable
        model evaluated at `fiducials`.
      fiducials (np.ndarray): Parameter point(s) at which derivatives are
        evaluated.
      covariance_matrix (np.ndarray): Observables' covariance used to weight
        derivatives.
      expected_fisher (np.ndarray): Reference Fisher matrix.
      expected_dali_g (np.ndarray): Reference third-order DALI tensor G.
      expected_dali_h (np.ndarray): Reference fourth-order DALI tensor H.

    Raises:
      AssertionError: If any tensor deviates beyond the specified tolerances.
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    le = LikelihoodExpansion(observables, fiducial_values, covmat)

    # It is possible for the computed (and expected) values of the tensors
    # to be much smaller than 0. The default value of the parameter atol of
    # np.isclose is then not appropriate: see the numpy documentation at
    # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    # The value has been set to 0 instead, so the tolerance is quantified
    # by only the relative difference.

    # Fisher stays strict
    fisher_matrix = le.get_forecast_tensors(forecast_order=1)
    assert np.allclose(np.asarray(fisher_matrix, float),
                          np.asarray(expected_fisher, float),
                          atol=0)

    # DALI with tight relative tol + tiny absolute floor only near zero
    dali_g, dali_h = le.get_forecast_tensors(forecast_order=2)
    _assert_close_mixed(dali_g, expected_dali_g, rtol=1e-7, label="dali_g")
    _assert_close_mixed(dali_h, expected_dali_h, rtol=1e-7, label="dali_h")


def _assert_close_mixed(actual, expected, *, rtol=1e-8, zero_band=1e-10, floor=5e-13, label=""):
    """Assert numerical closeness with a near-zero absolute floor.

    Compares arrays using `np.isclose`, but applies a small absolute
    tolerance only where the reference values are already near zero. This
    pattern preserves strict relative comparisons for meaningful entries while
    preventing spurious failures from sub-epsilon noise in ~0 slots.

    Args:
      actual (array_like): Actual values.
      expected (array_like): Reference values (tolerances are defined relative to this).
      rtol (float, optional): Relative tolerance for non-near-zero entries.
      zero_band (float, optional): Threshold below which ``abs(b)`` is treated
        as “near zero” and the absolute floor applies.
      floor (float, optional): Absolute tolerance to allow only where
        ``abs(b) < zero_band``.
      label (str, optional): Short label included in the failure message.

    Raises:
      AssertionError: If any element of ``a`` is not close to ``b`` under the
        mixed tolerance rule. The message includes the first failing index,
        values, absolute difference, and effective tolerances.
    """
    actual = np.asarray(actual, dtype=float)  # ensure float64
    expected = np.asarray(expected, dtype=float)

    # absolute floor only where the *expected* value is near zero
    atol = np.where(np.abs(expected) < zero_band, floor, 0.0)

    ok = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if not ok.all():
        idx = tuple(np.argwhere(~ok)[0])
        diff = actual - expected
        raise AssertionError(
            f"{label} not close at {idx}: "
            f"actual={actual[idx]} expected={expected[idx]} "
            f"|diff|={abs(diff[idx])} rtol={rtol} atol[idx]={atol[idx]}"
        )


def test_raises_on_mismatched_obs_cov_dims_runtime():
    """If model and covariance dimensions mismatch, raise ValueError."""
    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0]**2])

    le = LikelihoodExpansion(model, np.array([0.1]), np.eye(2))  # construct ok
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=1)  # shape check triggers here


def test_le_init_exposes_core_attrs():
    """Test that core attributes are exposed on the instance."""
    L = LikelihoodExpansion(lambda x: x, np.array([1.0, -2.0]), np.eye(2))
    # choose whatever names your class actually sets; adjust if needed
    for name in ("function", "theta0", "cov"):
        assert hasattr(L, name)

def test_inv_cov_behaves_like_inverse():
    """Test that _inv_cov() returns a matrix behaving like the inverse."""
    cov = np.array([[2.0, 0.0],
                    [0.0, 0.5]])
    L = LikelihoodExpansion(lambda x: x, np.array([0.0, 0.0]), cov)
    inv = L._inv_cov()
    np.testing.assert_allclose(inv @ cov, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(cov @ inv, np.eye(2), atol=1e-12)

def test_raises_on_mismatched_obs_cov_dims():
    """If model output length != cov size, computing tensors should raise."""
    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0] ** 2])

    le = LikelihoodExpansion(model, np.array([0.1]), np.eye(2))  # constructs fine

    # Fisher path
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=1)

    # DALI path (optional, but good to check both)
    with pytest.raises(ValueError):
        le.get_forecast_tensors(forecast_order=2)
