"""Tests for forecast_core methods."""

import numpy as np
import pytest

import derivkit.forecasting.forecast_core as fc
from derivkit.forecasting.forecast_core import (
    SUPPORTED_FORECAST_ORDERS,
    _get_derivatives,
    get_forecast_tensors,
)


def _identity(x):
    """A test function that returns its input unchanged."""
    return x

def _two_obs(theta):
    """A test function that returns two identical observables."""
    t = float(np.asarray(theta)[0])
    return np.array([t, t], dtype=float)


def get_fisher_matrix(forecast):
    """Returns fisher matrix."""
    return forecast[1][0]


def get_dali_doublet(forecast):
    """Returns dali doublet (d1, d2)."""
    d1, d2 = forecast[2]
    return d1, d2


def get_dali_triplet(forecast):
    """Returns dali triplet (t1, t2, t3)."""
    t1, t2, t3 = forecast[3]
    return t1, t2, t3


def test_derivative_order():
    """Tests that unsupported derivative orders raise ValueError."""
    func = _identity
    theta0 = np.array([1.0])
    cov = np.array([[1.0]])

    with pytest.raises(ValueError):
        _get_derivatives(func, theta0, cov, order=np.random.randint(low=4, high=30))


def test_forecast_order():
    """Tests that unsupported forecast orders raise ValueError."""
    func = _identity
    theta0 = np.array([1.0])
    cov = np.array([[1.0]])

    with pytest.raises(ValueError):
        get_forecast_tensors(
            func, theta0, cov, forecast_order=np.random.randint(low=4, high=30)
        )


def test_pseudoinverse_path_no_nan(caplog):
    """Tests that pseudoinverse path yields finite tensors."""
    func = _two_obs
    theta0 = np.array([1.0])
    cov = np.array([[1.0, 1.0],
                    [1.0, 1.0]], dtype=float)

    out_fisher = get_forecast_tensors(func, theta0, cov, forecast_order=1)
    out_doublet = get_forecast_tensors(func, theta0, cov, forecast_order=2)

    fisher_matrix = get_fisher_matrix(out_fisher)
    dali_g, dali_h = get_dali_doublet(out_doublet)

    assert any("pseudoinverse" in r.message.lower() for r in caplog.records)

    assert np.isfinite(fisher_matrix).all()
    assert np.isfinite(dali_g).all()
    assert np.isfinite(dali_h).all()


@pytest.mark.parametrize(
    (
        "model, "
        "fiducials, "
        "covariance_matrix, "
        "expected_fisher,"
        "expected_d1, "
        "expected_d2"
    ),
    [
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[1.03612509]]),
            np.array([[[0.49105455]]]),
            np.array([[[[0.23272727]]]]),
        ),
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([1.1, 0.4]),
            np.array([[1.0, 2.75], [3.2, 0.1]]),
            np.array([[-0.00890115, 0.08901149], [0.10357701, -0.01177011]]),
            np.array(
                [
                    [
                        [-8.09195402e-03, 8.09195402e-02],
                        [-1.46382975e-16, -2.08644092e-16],
                    ],
                    [
                        [-1.42668169e-16, -3.75546474e-16],
                        [2.58942529e-01, -2.94252874e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-7.35632184e-03, -1.11448615e-16],
                            [-1.06393661e-16, 2.02298851e-01],
                        ],
                        [
                            [-1.33075432e-16, 7.15511183e-31],
                            [1.01887953e-30, -5.21610229e-16],
                        ],
                    ],
                    [
                        [
                            [-1.29698336e-16, 9.78598871e-31],
                            [1.29608820e-30, -9.38866185e-16],
                        ],
                        [
                            [2.35402299e-01, -6.14828927e-16],
                            [-1.10097326e-15, -7.35632184e-02],
                        ],
                    ],
                ]
            ),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[0.01890366]]),
            np.array([[[-0.03087509]]]),
            np.array([[[[0.05042786]]]]),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([1.1, 0.4]),
            np.array([[1.0, 2.75], [3.2, 0.1]]),
            np.array([[-0.00414466, 0.07008167], [0.08154958, -0.01566948]]),
            np.array(
                [
                    [
                        [7.89639690e-04, -1.33519460e-02],
                        [6.87814470e-15, -1.84147323e-15],
                    ],
                    [
                        [6.59290723e-17, -1.11478872e-15],
                        [1.71255860e-01, -3.29062482e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-1.50441995e-04, -1.12697772e-15],
                            [-1.25607936e-17, -2.80393711e-02],
                        ],
                        [
                            [-1.31042274e-15, -2.06221337e-28],
                            [-1.09410605e-28, -3.86713302e-15],
                        ],
                    ],
                    [
                        [
                            [-1.25607936e-17, -9.40943024e-29],
                            [-1.04873334e-30, -2.34108006e-15],
                        ],
                        [
                            [-3.26276318e-02, -4.04783117e-15],
                            [-2.72416588e-15, -6.91038222e-02],
                        ],
                    ],
                ]
            ),
        ),
    ],
)
def test_forecast(
    model,
    fiducials,
    covariance_matrix,
    expected_fisher,
    expected_d1,
    expected_d2,
    caplog,
):
    """Validates Fisher and DALI tensors against reference values.

    This test is parametrized over simple scalar/vector models and covariance
    matrices.

    - Fisher is compared with `atol=0` (after casting to float64).
    - DALI tensors (D1, D2) use a mixed tolerance: tight relative tolerance
    and a
      tiny absolute floor only where the expected entries are near zero. This
      avoids false failures from floating-point noise in ~0 entries while
      keeping real values strict.
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    func = observables
    theta0 = fiducial_values

    # Fisher (order=1): set tolerances up front
    fisher_rtol = 3e-3
    fisher_atol = 1e-12

    out_fisher = get_forecast_tensors(func, theta0, covmat, forecast_order=1)
    out_doublet = get_forecast_tensors(func, theta0, covmat, forecast_order=2)

    fisher = get_fisher_matrix(out_fisher)
    d1, d2 = get_dali_doublet(out_doublet)

    # Helper: warn only if cov is non-symmetric
    want_sym_warn = not np.allclose(covmat, covmat.T)
    if want_sym_warn:
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert r"`cov` is not symmetric; proceeding as-is" in caplog.text

    assert np.allclose(
        np.asarray(fisher, float),
        np.asarray(expected_fisher, float),
        rtol=fisher_rtol,
        atol=fisher_atol,
    )

    is_multi_param = fiducials.size > 1
    rtol_d1 = 3e-3 if is_multi_param else 5e-4
    rtol_d2 = 5e-3 if is_multi_param else 2e-3

    if not np.allclose(covmat, covmat.T):
        rtol_d1 = max(rtol_d1, 1.5e-2)
        rtol_d2 = max(rtol_d2, 2.5e-2)

    _assert_close_mixed(d1, expected_d1, rtol=rtol_d1, label="d1")
    _assert_close_mixed(d2, expected_d2, rtol=rtol_d2, label="d2")


def _assert_close_mixed(
    actual, expected, *, rtol=1e-8, zero_band=1e-10, floor=5e-13, label=""
):
    """Asserts numerical closeness with a near-zero absolute floor.

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
    """Tests that get_forecast_tensors raises on mismatched observable/cov dims."""

    def model(theta):  # returns length 3
        t = np.asarray(theta)
        return np.array([t[0], t[0] + 1.0, t[0] ** 2])

    theta0 = np.array([0.1])
    cov = np.eye(2)  # expects 2 observables

    # Now the shape check happens immediately, for any forecast_order
    with pytest.raises(ValueError, match=r"Expected 2 observables"):
        get_forecast_tensors(model, theta0, cov, forecast_order=1)

    with pytest.raises(ValueError, match=r"Expected 2 observables"):
        get_forecast_tensors(model, theta0, cov, forecast_order=2)


def model_quadratic(theta: np.ndarray) -> np.ndarray:
    """A simple 2D→2D quadratic model for basic multi-parameter tests."""
    t0, t1 = np.asarray(theta, float)
    return np.array([t0**2, 2.0 * t0 * t1], float)


def model_cubic(theta: np.ndarray) -> np.ndarray:
    """Returns a linear combination of cubes."""
    return np.asarray([np.sum(np.asarray(theta)**4)])


@pytest.mark.parametrize(
    (
        "model, "
        "theta, "
        "expected,"
    ),
    [
        pytest.param(
            lambda x: x,
            np.array([2]),
            (
                np.array([[[0]]]),
                np.array([[[[[0]]]]]),
                np.array([[[[[[0]]]]]]),
            ),
        ),
        pytest.param(
            model_cubic,
            np.array([1]),
            (
                np.array([[[96]]]),
                np.array([[[[[288]]]]]),
                np.array([[[[[[576]]]]]]),
            ),
        ),
    ]
)
def test_scalar_dali_triplet(model, theta, expected):
    """Tests the DALI triplet for scalar models.

    The models have a single parameter and produce a single observable.
    """
    forecast = get_forecast_tensors(
        model,
        theta,
        np.array([1]),
        forecast_order=3,
    )

    # forecast is a dict {1: (F,), 2: (D1, D2), 3: (T1, T2, T3)}
    triplet = forecast[3]  # (T1, T2, T3)

    assert len(triplet) == len(expected)

    # Keep parametrization unchanged:
    # expected[0] is (p,p,p) in this test file, but code returns T1 as (p,p,p,p).
    expected_fixed = (expected[0][..., np.newaxis], expected[1], expected[2])

    for i in range(len(triplet)):
        assert np.allclose(triplet[i], expected_fixed[i], atol=1e-6)


def get_all_indices(max_indices):
    """Returns all combinations of numbers up to the given input.

    Args:
        max_indices: a tuple of numbers. Each tuple indicates the number
            of possible values corresponding to the idex of that axis.

    Returns:
        A list containing all combinations of possible indices.
        Each configuration of indices is stored in a tuple.
    """
    tmp = max_indices
    tmp_tuple = ()
    for i in tmp:
        tmp_tuple += (np.arange(0, i),)

    grid = np.meshgrid(*tmp_tuple)
    combinations = np.stack(grid, axis=-1).reshape(-1, len(tmp_tuple))

    return list(map(tuple, combinations))


def test_vector_dali_triplet():
    """Tests the DALI triplet for a vector model.

    In this case the model takes 2 parameters and returns two observables.
    The observables are assumed to be uncorrelated; the correlation matrix
    is unity.

    The reference tensors are called T1, T2, T3. This notation follows the
    convention established in
    derivkit.forecasting.forecast_core.py:get_forecast_tensors.
    """
    model = lambda x: np.array([np.cos(x[0]) + np.sin(x[1]), x[0]**2 * x[1]]) #noqa

    x = np.pi/3
    y = np.e
    theta = np.array([x, y])

    cov = np.eye(2)

    forecast = get_forecast_tensors(model, theta, cov, forecast_order=3)
    triplet = forecast[3]  # (T1, T2, T3)

    # The DALI triplet tensors are fully symmetric in the first three indices.
    # Additionally, the T2 tensor is fully symmetric in its last two indices,
    # while the T3 is symmetric in the last three indices. Below are only the
    # non-zero independent components; all the other non-zero components can be
    # obtained from these through permutations of the axes.
    #
    # The convention is that the independent non-zero components have the indices
    # of each group sorted with the zeroes first and the ones second.
    T1 = np.zeros((2, 2, 2, 2))
    T1[0,0,0,0] = -np.sin(x)**2
    T1[0,0,0,1] = np.sin(x) * np.cos(y)
    T1[0,0,1,0] = 4*x*y
    T1[0,0,1,1] = 2*x**2
    T1[1,1,1,0] = np.sin(x) * np.cos(y)
    T1[1,1,1,1] = -np.cos(y)**2

    T2 = np.zeros((2, 2, 2, 2, 2))
    T2[0,0,0,0,0] = - np.sin(x) * np.cos(x)
    T2[0,0,0,1,1] = - np.sin(x) * np.sin(y)
    T2[0,0,1,0,0] = 4*y
    T2[0,0,1,0,1] = 4*x
    T2[1,1,1,0,0] = np.cos(x) * np.cos(y)
    T2[1,1,1,1,1] = np.cos(y) * np.sin(y)

    T3 = np.zeros((2,2,2,2,2,2))
    T3[0,0,0,0,0,0] = np.sin(x)**2
    T3[0,0,1,0,0,1] = 4
    T3[0,0,0,1,1,1] = - np.sin(x) * np.cos(y)
    T3[1,1,1,0,0,0] = T3[0,0,0,1,1,1]
    T3[1,1,1,1,1,1] = np.cos(y)**2

    for ind, expected_tensor in enumerate((T1,T2,T3)):
        assert expected_tensor.shape == triplet[ind].shape

        dimension = expected_tensor.ndim
        max_index = len(expected_tensor[0])
        max_indices = dimension * (max_index,)

        index_combinations = get_all_indices(max_indices)
        for i in index_combinations:
            # The tensor components cannot be compared directly, because
            # expected_tensor contains only the components for the sorted
            # index groups. The indices therefore have to be grouped and sorted
            # before they're passed to expected_tensor.
            # The first index group is a triple, whereas the
            # second group comprises what remains when the triple is stripped.
            expected_indices = (*sorted(i[:3]), *sorted(i[3:]))
            assert np.allclose(
                expected_tensor[expected_indices],
                triplet[ind][i],
                rtol=1e-8,
                atol=1e-3,
            )


def test_get_forecast_tensors_output_type():
    """Tests that a full forecast returns a dictionary of the right type."""
    max_order = np.random.randint(low=1, high=SUPPORTED_FORECAST_ORDERS[-1])
    forecast = get_forecast_tensors(
        model_cubic,
        [1.2],
        [1],
        forecast_order=max_order,
    )

    assert isinstance(forecast, dict)

    for key in forecast.keys():
        assert isinstance(key, int)
        assert isinstance(forecast[key], tuple)
        for element in forecast[key]:
            assert isinstance(element, np.ndarray)


def test_forecast_dict_keys_and_multiplet_lengths():
    """Tests that the forecast dictionary has the right keys and multiplets."""
    theta0 = np.array([0.2, -0.1])
    cov = np.eye(2)

    def model(th):
        """Test model with non-trivial derivatives."""
        t0, t1 = np.asarray(th, float)
        return np.array([t0 + t1, t0 - 2 * t1], float)

    out = get_forecast_tensors(model, theta0, cov, forecast_order=3)

    assert set(out.keys()) == {1, 2, 3}
    assert len(out[1]) == 1  # (F,)
    assert len(out[2]) == 2  # (D1, D2)
    assert len(out[3]) == 3  # (T1, T2, T3)


@pytest.mark.parametrize("p,nobs", [(1, 1), (2, 2), (3, 2)])
def test_tensor_shapes_all_orders(p, nobs):
    """Tests that the forecast tensors have the right shapes for all orders."""
    rng = np.random.default_rng(0)
    theta0 = rng.normal(size=p)
    cov = np.eye(nobs)

    def model(th):
        """Test model with non-trivial derivatives."""
        th = np.asarray(th, dtype=float)
        y = np.zeros(nobs, dtype=float)
        y[0] = float(np.sum(th ** 2))
        if nobs > 1:
            y[1] = float(np.sum(th) + th[0] ** 3)
        return y

    out = get_forecast_tensors(model, theta0, cov, forecast_order=3)

    F = out[1][0]
    assert F.shape == (p, p)

    D1, D2 = out[2]
    assert D1.shape == (p, p, p)
    assert D2.shape == (p, p, p, p)

    T1, T2, T3 = out[3]
    assert T1.shape == (p, p, p, p)
    assert T2.shape == (p, p, p, p, p)
    assert T3.shape == (p, p, p, p, p, p)


def _assert_symmetric_under_permutation(x, axes, rtol=0, atol=0):
    """Tests that a tensor is symmetric under a given axis permutation."""
    perm = list(range(x.ndim))
    a, b = axes
    perm[a], perm[b] = perm[b], perm[a]
    np.testing.assert_allclose(x, np.transpose(x, perm), rtol=rtol, atol=atol)

def test_expected_symmetries():
    """Tests that the forecast tensors are symmetric under permutations."""
    theta0 = np.array([0.3, -0.2])
    cov = np.eye(2)

    def model(th):
        """Test model with non-trivial derivatives."""
        t0, t1 = np.asarray(th, float)
        return np.array([t0**2 + t1, t0 * t1], float)

    out = get_forecast_tensors(model, theta0, cov, forecast_order=3)
    F = out[1][0]
    np.testing.assert_allclose(F, F.T)

    D1, D2 = out[2]
    # D1 is symmetric in first two indices (a,b)
    _assert_symmetric_under_permutation(D1, (0, 1))
    # D2 symmetric under (a<->b) and (c<->d)
    _assert_symmetric_under_permutation(D2, (0, 1))
    _assert_symmetric_under_permutation(D2, (2, 3))

    T1, T2, T3 = out[3]
    # T1 symmetric in first 3 (a,b,c) -> check a<->b and b<->c
    _assert_symmetric_under_permutation(T1, (0, 1))
    _assert_symmetric_under_permutation(T1, (1, 2))
    # T2 symmetric in first 3 and in last 2
    _assert_symmetric_under_permutation(T2, (0, 1))
    _assert_symmetric_under_permutation(T2, (1, 2))
    _assert_symmetric_under_permutation(T2, (3, 4))
    # T3 symmetric in first 3 and last 3 (check a couple swaps)
    _assert_symmetric_under_permutation(T3, (0, 1))
    _assert_symmetric_under_permutation(T3, (1, 2))
    _assert_symmetric_under_permutation(T3, (3, 4))
    _assert_symmetric_under_permutation(T3, (4, 5))


def test_get_derivatives_rejects_bad_jacobian_shape(monkeypatch):
    """Tests that get_derivatives rejects bad jacobian shapes."""

    class FakeCK:
        """Mock CalculusKit class that returns nonsense jacobian shapes."""
        def __init__(self, function, theta0):
            """Initializes the fake CalculusKit class."""
            _, _ = function, theta0
            pass
        def jacobian(self, **kwargs):
            """Bad Jacobian: 3x3 instead of 2x2."""
            _ = kwargs
            return np.zeros((3, 3))  # nonsense

    monkeypatch.setattr(fc, "CalculusKit", FakeCK)

    with pytest.raises(ValueError, match=r"jacobian returned unexpected shape"):
        fc._get_derivatives(lambda th: np.array([1.0]), np.array([0.1]), np.eye(1), order=1)

def test_get_derivatives_rejects_bad_hessian_shape(monkeypatch):
    """Tests that get_derivatives rejects bad hessian shapes."""
    class FakeCK:
        def __init__(self, function, theta0):
            """Initializes the fake CalculusKit class."""
            _, _ = function, theta0
            pass
        def hessian(self, **kwargs):
            """Bad hessian: 4x4x4 instead of 4x4x4x4."""
            _ = kwargs
            return np.zeros((4, 4, 4))  # nonsense

    monkeypatch.setattr(fc, "CalculusKit", FakeCK)

    with pytest.raises(ValueError, match=r"hessian returned unexpected shape"):
        fc._get_derivatives(lambda th: np.array([1.0]), np.array([0.1, 0.2]), np.eye(1), order=2)

def test_get_derivatives_rejects_bad_hyper_hessian_shape(monkeypatch):
    """Tests that get_derivatives rejects bad hyper_hessian shapes."""

    class FakeCK:
        """Fake CalculusKit class that returns nonsense hyper_hessian shapes."""
        def __init__(self, function, theta0):
            """Initializes the fake CalculusKit class."""
            _, _ = function, theta0
            pass
        def hyper_hessian(self, **kwargs):
            """Bad hyper_hessian: 2x2x2 instead of 2x2x2x2."""
            _ = kwargs
            return np.zeros((2, 2, 2, 2, 2))  # nonsense

    monkeypatch.setattr(fc, "CalculusKit", FakeCK)

    with pytest.raises(ValueError, match=r"hyper_hessian returned unexpected shape"):
        fc._get_derivatives(lambda th: np.array([1.0]), np.array([0.1]), np.eye(1), order=3)


def test_forecast_order_type_error():
    """Tests that forecast_order must be an int."""
    with pytest.raises(TypeError, match=r"forecast_order must be an int"):
        get_forecast_tensors(lambda x: x, np.array([1.0]), np.eye(1), forecast_order="two")


def test_theta0_empty_raises():
    """Tests that theta0 must be non-empty 1D."""
    with pytest.raises(ValueError, match=r"theta0 must be non-empty 1D"):
        get_forecast_tensors(lambda th: np.array([1.0]), np.array([]), np.eye(1), forecast_order=1)


def test_scalar_model_with_1x1_cov_works():
    """Tests that a scalar model with a 1x1 covariance matrix works."""
    def model(th):
        """Test model with non-trivial derivatives."""
        t = float(np.asarray(th)[0])
        return 2.0 * t

    out = get_forecast_tensors(model, np.array([0.1]), np.array([[1.0]]), forecast_order=2)
    assert out[1][0].shape == (1, 1)
    assert out[2][0].shape == (1, 1, 1)
    assert out[2][1].shape == (1, 1, 1, 1)


def test_method_and_workers_are_forwarded(monkeypatch):
    """Tests that method and workers are forwarded to CalculusKit."""
    seen = {}

    class FakeCK:
        """Mock CalculusKit class that records the method and n_workers."""
        def __init__(self, function, theta0):
            """Initializes the fake CalculusKit class."""
            _, _ = function, theta0
            pass
        def jacobian(self, **kwargs):
            """Returns a nonsense jacobian."""
            seen.update(kwargs)
            return np.zeros((1, 1))
        def hessian(self, **kwargs):
            """Returns a nonsense hessian."""
            _ = kwargs
            return np.zeros((1, 1, 1))
        def hyper_hessian(self, **kwargs):
            """Returns a nonsense hyper_hessian."""
            _ = kwargs
            return np.zeros((1, 1, 1, 1))

    monkeypatch.setattr(fc, "CalculusKit", FakeCK)

    # just call order=1 derivative path
    fc._get_derivatives(lambda th: np.array([1.0]), np.array([0.1]), np.eye(1), order=1,
                        method="finite", n_workers=3)

    assert seen.get("method") == "finite"
    assert seen.get("n_workers") == 3
