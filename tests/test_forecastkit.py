"""Test functions for the ForecastKit class."""
import numpy
import pytest

from derivkit.forecast_kit import ForecastKit
from derivkit.forecasting.expansions import LikelihoodExpansion


def test_order():
    """High derivative orders (>2) should raise ValueError on LikelihoodExpansion."""
    like = LikelihoodExpansion(lambda x: x, numpy.array([1]), numpy.array([1]))
    with pytest.raises(ValueError):
        like._get_derivatives(order=3)
    with pytest.raises(ValueError):
        like._get_derivatives(order=numpy.random.randint(low=4, high=30))


def test_forecast_order():
    """Tests if high forecast orders raise a ValueError.

    The forecast is an expansions of the likelihood function so really this
    tests if high order expansions raise a ValueError.

    High order means higher than 2. The function tests an order of 3
    and a random number between 4 and 30 inclusive.
    """
    like = LikelihoodExpansion(lambda x: x, numpy.array([1]), numpy.array([1]),)

    with pytest.raises(ValueError):
                like.get_forecast_tensors(forecast_order=3)

    with pytest.raises(ValueError):
                like.get_forecast_tensors(forecast_order = numpy.random.randint(low=4, high=30))

def test_pseudoinverse_path_no_nan():
    """If inversion fails, pseudoinverse path should still return finite numbers."""
    # singular covariance -> forces pinv path
    forecaster = ForecastKit(lambda x: x,
        numpy.array([1.0]),
        numpy.array([[0.0]]),
    )
    fisher = forecaster.fisher()
    assert numpy.isfinite(fisher).all()
    tensor_g, tensor_h = forecaster.dali()
    assert numpy.isfinite(tensor_g).all()
    assert numpy.isfinite(tensor_h).all()


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
            numpy.array([2.11]),
            numpy.array([[2.75]]),
            numpy.array([[1.03612509]]),
            numpy.array([[[0.49105455]]]),
            numpy.array([[[[0.23272727]]]])
        ),
        pytest.param(
            lambda x: 0.4 * x**2,
            numpy.array([1.1, 0.4]),
            numpy.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            numpy.array([
                [-0.00890115,  0.08901149],
                [ 0.10357701, -0.01177011]
            ]),
            numpy.array([
                [
                    [-8.09195402e-03,  8.09195402e-02],
                    [-1.46382975e-16, -2.08644092e-16]
                ],
                [
                    [-1.42668169e-16, -3.75546474e-16],
                    [ 2.58942529e-01, -2.94252874e-02]
                ]
            ]),
            numpy.array([
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
            lambda x: numpy.exp(-0.5 * x**2),
            numpy.array([2.11]),
            numpy.array([[2.75]]),
            numpy.array([[0.01890366]]),
            numpy.array([[[-0.03087509]]]),
            numpy.array([[[[0.05042786]]]])
        ),
        pytest.param(
            lambda x: numpy.exp(-0.5 * x**2),
            numpy.array([1.1, 0.4]),
            numpy.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            numpy.array([
                [-0.00414466, 0.07008167],
                [0.08154958, -0.01566948]
            ]),
            numpy.array([
                [
                    [ 7.89639690e-04, -1.33519460e-02],
                    [ 6.87814470e-15, -1.84147323e-15]
                ],
                [
                    [ 6.59290723e-17, -1.11478872e-15],
                    [ 1.71255860e-01, -3.29062482e-02]
                ]
            ]),
            numpy.array([
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
      model (Callable[[numpy.ndarray], numpy.ndarray | float]): Observable
        model evaluated at `fiducials`.
      fiducials (numpy.ndarray): Parameter point(s) at which derivatives are
        evaluated.
      covariance_matrix (numpy.ndarray): Observables' covariance used to weight
        derivatives.
      expected_fisher (numpy.ndarray): Reference Fisher matrix.
      expected_dali_g (numpy.ndarray): Reference third-order DALI tensor G.
      expected_dali_h (numpy.ndarray): Reference fourth-order DALI tensor H.

    Raises:
      AssertionError: If any tensor deviates beyond the specified tolerances.
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    forecaster = ForecastKit(observables, fiducial_values, covmat)

    # It is possible for the computed (and expected) values of the tensors
    # to be much smaller than 0. The default value of the parameter atol of
    # numpy.isclose is then not appropriate: see the numpy documentation at
    # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html.
    # The value has been set to 0 instead, so the tolerance is quantified
    # by only the relative difference.

    # Fisher stays strict
    fisher_matrix = forecaster.fisher()
    assert numpy.allclose(numpy.asarray(fisher_matrix, float),
                          numpy.asarray(expected_fisher, float),
                          atol=0)

    # DALI with tight relative tol + tiny absolute floor only near zero
    dali_g, dali_h = forecaster.dali()
    _assert_close_mixed(dali_g, expected_dali_g, rtol=1e-7, label="dali_g")
    _assert_close_mixed(dali_h, expected_dali_h, rtol=1e-7, label="dali_h")


def _assert_close_mixed(actual, expected, *, rtol=1e-8, zero_band=1e-10, floor=5e-13, label=""):
    """Assert numerical closeness with a near-zero absolute floor.

    Compares arrays using `numpy.isclose`, but applies a small absolute
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
    actual = numpy.asarray(actual, dtype=float)  # ensure float64
    expected = numpy.asarray(expected, dtype=float)

    # absolute floor only where the *expected* value is near zero
    atol = numpy.where(numpy.abs(expected) < zero_band, floor, 0.0)

    ok = numpy.isclose(actual, expected, rtol=rtol, atol=atol)
    if not ok.all():
        idx = tuple(numpy.argwhere(~ok)[0])
        diff = actual - expected
        raise AssertionError(
            f"{label} not close at {idx}: "
            f"actual={actual[idx]} expected={expected[idx]} "
            f"|diff|={abs(diff[idx])} rtol={rtol} atol[idx]={atol[idx]}"
        )
