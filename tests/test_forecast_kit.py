"""Tests for ForecastKit class."""

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit


def test_forecastkit_delegates(monkeypatch):
    """Tests that ForecastKit delegates to fisher/dali helpers correctly."""
    calls = {
        "fisher": None,
        "dali": None,
    }

    def fake_build_fisher_matrix(function, theta0, cov, *, method=None, n_workers=1, **dk_kwargs):
        """Returns a mock Fisher matrix."""
        calls["fisher"] = {
            "function": function,
            "theta0": np.asarray(theta0),
            "cov": np.asarray(cov),
            "method": method,
            "n_workers": n_workers,
            "dk_kwargs": dk_kwargs,
        }
        return np.full((2, 2), 42.0)

    def fake_build_dali(function, theta0, cov, *, method=None, n_workers=1, **dk_kwargs):
        """Returns mock DALI tensors."""
        calls["dali"] = {
            "function": function,
            "theta0": np.asarray(theta0),
            "cov": np.asarray(cov),
            "method": method,
            "n_workers": n_workers,
            "dk_kwargs": dk_kwargs,
        }
        g_tensor = np.zeros((2, 2, 2))
        h_tensor = np.ones((2, 2, 2, 2))
        return g_tensor, h_tensor

    # Patch the helpers that ForecastKit uses internally
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix", fake_build_fisher_matrix, raising=True
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali", fake_build_dali, raising=True
    )

    # inputs
    def model(theta):
        return np.asarray(theta)  # any callable

    theta0 = np.array([0.1, -0.2])
    cov = np.eye(2)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    # The fisher() method defaults to forecast_order=1 and forwards n_workers.
    # The Fisher computation delegates to the helper function and forwards n_workers.
    fish = fk.fisher(n_workers=3)
    assert fish.shape == (2, 2)
    assert np.all(fish == 42.0)

    assert calls["fisher"] is not None
    np.testing.assert_allclose(calls["fisher"]["theta0"], theta0)
    np.testing.assert_allclose(calls["fisher"]["cov"], cov)
    assert calls["fisher"]["function"] is model
    assert calls["fisher"]["n_workers"] == 3

    # The DALI computation delegates to the helper function and forwards n_workers.
    g_tensor, h_tensor = fk.dali(n_workers=4)
    assert g_tensor.shape == (2, 2, 2)
    assert h_tensor.shape == (2, 2, 2, 2)

    assert calls["dali"] is not None
    np.testing.assert_allclose(calls["dali"]["theta0"], theta0)
    np.testing.assert_allclose(calls["dali"]["cov"], cov)
    assert calls["dali"]["function"] is model
    assert calls["dali"]["n_workers"] == 4


def test_default_n_workers_forwarded(monkeypatch):
    """Tests that default n_workers=1 is forwarded to fisher/dali helpers."""
    n_workers_seen = {"fisher": None, "dali": None}

    def fake_build_fisher_matrix(*args, n_workers=1, **kwargs):
        n_workers_seen["fisher"] = n_workers
        return np.zeros((1, 1))

    def fake_build_dali(*args, n_workers=1, **kwargs):
        n_workers_seen["dali"] = n_workers
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1))

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix", fake_build_fisher_matrix, raising=True
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali", fake_build_dali, raising=True
    )

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))
    fk.fisher()
    fk.dali()

    assert n_workers_seen["fisher"] == 1
    assert n_workers_seen["dali"] == 1


def test_return_types_match_helpers(monkeypatch):
    """Tests that return types from ForecastKit match those from helper functions."""

    def fake_build_fisher_matrix(*args, **kwargs):
        """Returns mock Fisher matrix."""
        return np.array([[123.0]])

    def fake_build_dali(*args, **kwargs):
        """Returns mock DALI tensors."""
        return np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_matrix",
        fake_build_fisher_matrix,
        raising=True,
    )
    monkeypatch.setattr(
        "derivkit.forecast_kit.build_dali",
        fake_build_dali,
        raising=True,
    )

    fk = ForecastKit(lambda x: np.asarray(x), np.array([0.0]), np.eye(1))

    fish = fk.fisher()
    assert isinstance(fish, np.ndarray)

    g_tensor, h_tensor = fk.dali()
    assert isinstance(g_tensor, np.ndarray)
    assert isinstance(h_tensor, np.ndarray)


def test_init_resolves_covariance_callable_caches_cov0():
    """Tests that ForecastKit accepts cov=cov_fn and caches cov0."""

    def cov_fn(_theta):
        """Returns a mock covariance matrix."""
        return np.eye(3)

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=cov_fn)

    assert fk.cov_fn is cov_fn
    assert fk.cov0.shape == (3, 3)
    np.testing.assert_allclose(fk.cov0, np.eye(3))
    assert fk.n_observables == 3


def test_fisher_raises_if_function_is_none():
    """Tests that ForecastKit.fisher requires a mean model."""
    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(1))
    with pytest.raises(ValueError, match=r"ForecastKit\.fisher: function must be provided\."):
        fk.fisher()


def test_dali_raises_if_function_is_none():
    """Tests that ForecastKit.dali requires a mean model."""
    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(1))
    with pytest.raises(ValueError, match=r"ForecastKit\.dali: function must be provided\."):
        fk.dali()


def test_fisher_bias_delegates(monkeypatch):
    """Tests that ForecastKit.fisher_bias delegates to build_fisher_bias."""
    seen = {}

    def fake_build_fisher_bias(
        *,
        function,
        theta0,
        cov,
        fisher_matrix,
        delta_nu,
        method=None,
        n_workers=1,
        rcond=1e-12,
        **dk_kwargs,
    ):
        """Mock build_fisher_bias that records inputs and returns fixed outputs."""
        seen["function"] = function
        seen["theta0"] = np.asarray(theta0)
        seen["cov"] = np.asarray(cov)
        seen["fisher_matrix"] = np.asarray(fisher_matrix)
        seen["delta_nu"] = np.asarray(delta_nu)
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["rcond"] = rcond
        seen["dk_kwargs"] = dk_kwargs
        return np.array([0.1, 0.2]), np.array([-0.01, 0.03])

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_fisher_bias",
        fake_build_fisher_bias,
        raising=True,
    )

    def model(theta):
        """Mock mean model function."""
        return np.asarray(theta)

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=model, theta0=theta0, cov=np.eye(3))

    fisher_matrix = np.eye(2)
    delta_nu = np.arange(3.0)

    bias_vec, dtheta = fk.fisher_bias(
        fisher_matrix=fisher_matrix,
        delta_nu=delta_nu,
        method="finite",
        n_workers=7,
        rcond=1e-9,
        step=1e-4,
    )

    np.testing.assert_allclose(bias_vec, np.array([0.1, 0.2]))
    np.testing.assert_allclose(dtheta, np.array([-0.01, 0.03]))

    assert seen["function"] is model
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["cov"], np.eye(3))
    np.testing.assert_allclose(seen["fisher_matrix"], fisher_matrix)
    np.testing.assert_allclose(seen["delta_nu"], delta_nu)
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 7
    assert seen["rcond"] == 1e-9
    assert seen["dk_kwargs"]["step"] == 1e-4


def test_delta_nu_delegates(monkeypatch):
    """Tests that ForecastKit.delta_nu delegates to build_delta_nu."""
    seen = {}

    def fake_build_delta_nu(*, cov, data_biased, data_unbiased):
        """Mock build_delta_nu that records inputs and returns fixed output."""
        seen["cov"] = np.asarray(cov)
        seen["biased"] = np.asarray(data_biased)
        seen["unbiased"] = np.asarray(data_unbiased)
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_delta_nu",
        fake_build_delta_nu,
        raising=True,
    )

    fk = ForecastKit(function=lambda t: np.asarray(t), theta0=np.array([0.0]), cov=np.eye(3))
    out = fk.delta_nu(data_unbiased=np.zeros(3), data_biased=np.ones(3))

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(seen["cov"], np.eye(3))
    np.testing.assert_allclose(seen["biased"], np.ones(3))
    np.testing.assert_allclose(seen["unbiased"], np.zeros(3))


def test_generalized_fisher_delegates_with_cov_fn(monkeypatch):
    """Tests that ForecastKit.gaussian_fisher delegates to build_gaussian_fisher_matrix."""
    seen = {}

    def fake_build_gaussian_fisher_matrix(
        *,
        theta0,
        cov,
        function,
        method=None,
        n_workers=1,
        rcond=1e-12,
        symmetrize_dcov=True,
        **dk_kwargs,
    ):
        """Mock build_gaussian_fisher_matrix that records inputs and returns fixed output."""
        seen["theta0"] = np.asarray(theta0)
        seen["cov"] = cov
        seen["function"] = function
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["rcond"] = rcond
        seen["symmetrize_dcov"] = symmetrize_dcov
        seen["dk_kwargs"] = dk_kwargs
        return np.full((2, 2), 9.0)

    monkeypatch.setattr(
        "derivkit.forecast_kit.build_gaussian_fisher_matrix",
        fake_build_gaussian_fisher_matrix,
        raising=True,
    )

    def cov_fn(_theta):
        """Mock covariance function."""
        return np.eye(3)

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=cov_fn)

    out = fk.gaussian_fisher(
        method="finite",
        n_workers=5,
        rcond=1e-8,
        symmetrize_dcov=False,
        step=1e-4,
    )

    np.testing.assert_allclose(out, np.full((2, 2), 9.0))
    assert callable(seen["cov"])
    assert seen["function"] is None
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 5
    assert seen["rcond"] == 1e-8
    assert seen["symmetrize_dcov"] is False
    assert seen["dk_kwargs"]["step"] == 1e-4


def test_submatrix_fisher_delegates(monkeypatch):
    """Tests that ForecastKit.submatrix_fisher delegates to submatrix_fisher."""
    seen: dict[str, object] = {}

    def fake_submatrix_fisher(*, fisher, idx):
        """Mock submatrix_fisher that records inputs and returns fixed output."""
        seen["fisher"] = np.asarray(fisher)
        seen["idx"] = list(idx)
        return np.full((2, 2), 7.0)

    monkeypatch.setattr(
        "derivkit.forecast_kit.submatrix_fisher",
        fake_submatrix_fisher,
        raising=True,
    )

    fk = ForecastKit(function=None, theta0=np.array([0.0, 1.0]), cov=np.eye(1))

    fisher = np.eye(3)
    idx = [2, 0]

    out = fk.submatrix_fisher(fisher=fisher, idx=idx)

    np.testing.assert_allclose(out, np.full((2, 2), 7.0))
    np.testing.assert_allclose(seen["fisher"], fisher)
    assert seen["idx"] == idx


def test_submatrix_dali_delegates_uses_self_theta0(monkeypatch):
    """Tests that ForecastKit.submatrix_dali delegates and uses self.theta0."""
    seen: dict[str, object] = {}

    def fake_submatrix_dali(*, theta0, fisher, g_tensor, h_tensor, idx):
        """Mock submatrix_dali that records inputs and returns fixed output."""
        seen["theta0"] = np.asarray(theta0)
        seen["fisher"] = np.asarray(fisher)
        seen["g_tensor"] = np.asarray(g_tensor)
        seen["h_tensor"] = None if h_tensor is None else np.asarray(h_tensor)
        seen["idx"] = list(idx)
        return (
            np.array([9.0, 8.0]),
            np.full((2, 2), 1.0),
            np.full((2, 2, 2), 2.0),
            np.full((2, 2, 2, 2), 3.0),
        )

    monkeypatch.setattr(
        "derivkit.forecast_kit.submatrix_dali",
        fake_submatrix_dali,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2, 0.3])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    fisher = np.eye(3)
    g = np.zeros((3, 3, 3))
    h = np.ones((3, 3, 3, 3))
    idx = [1, 2]

    t0_sub, f_sub, g_sub, h_sub = fk.submatrix_dali(
        fisher=fisher,
        g_tensor=g,
        h_tensor=h,
        idx=idx,
    )

    np.testing.assert_allclose(t0_sub, np.array([9.0, 8.0]))
    np.testing.assert_allclose(f_sub, np.full((2, 2), 1.0))
    np.testing.assert_allclose(g_sub, np.full((2, 2, 2), 2.0))
    np.testing.assert_allclose(h_sub, np.full((2, 2, 2, 2), 3.0))

    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["fisher"], fisher)
    np.testing.assert_allclose(seen["g_tensor"], g)
    np.testing.assert_allclose(seen["h_tensor"], h)
    assert seen["idx"] == idx


def test_delta_chi2_fisher_delegates_uses_self_theta0(monkeypatch):
    """Tests that ForecastKit.delta_chi2_fisher delegates and uses self.theta0."""
    seen: dict[str, object] = {}

    def fake_delta_chi2_fisher(*, theta, theta0, fisher):
        """Mock delta_chi2_fisher that records inputs and returns fixed output."""
        seen["theta"] = np.asarray(theta)
        seen["theta0"] = np.asarray(theta0)
        seen["fisher"] = np.asarray(fisher)
        return 123.0

    monkeypatch.setattr(
        "derivkit.forecast_kit.delta_chi2_fisher",
        fake_delta_chi2_fisher,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    theta = np.array([0.3, 0.4])
    fisher = np.eye(2)

    out = fk.delta_chi2_fisher(theta=theta, fisher=fisher)

    assert out == 123.0
    np.testing.assert_allclose(seen["theta"], theta)
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["fisher"], fisher)


def test_delta_chi2_dali_delegates_uses_self_theta0_and_forwards_convention(monkeypatch):
    """Tests that ForecastKit.delta_chi2_dali delegates and forwards convention."""
    seen: dict[str, object] = {}

    def fake_delta_chi2_dali(*, theta, theta0, fisher, g_tensor, h_tensor, convention="delta_chi2"):
        """Mock delta_chi2_dali that records inputs and returns fixed output."""
        seen["theta"] = np.asarray(theta)
        seen["theta0"] = np.asarray(theta0)
        seen["fisher"] = np.asarray(fisher)
        seen["g_tensor"] = np.asarray(g_tensor)
        seen["h_tensor"] = None if h_tensor is None else np.asarray(h_tensor)
        seen["convention"] = convention
        return 456.0

    monkeypatch.setattr(
        "derivkit.forecast_kit.delta_chi2_dali",
        fake_delta_chi2_dali,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    theta = np.array([0.3, 0.4])
    fisher = np.eye(2)
    g = np.zeros((2, 2, 2))

    out = fk.delta_chi2_dali(
        theta=theta,
        fisher=fisher,
        g_tensor=g,
        h_tensor=None,
        convention="matplotlib_loglike",
    )

    assert out == 456.0
    np.testing.assert_allclose(seen["theta"], theta)
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["fisher"], fisher)
    np.testing.assert_allclose(seen["g_tensor"], g)
    assert seen["h_tensor"] is None
    assert seen["convention"] == "matplotlib_loglike"


def test_logposterior_fisher_delegates_uses_self_theta0_and_forwards_priors(monkeypatch):
    """Tests that ForecastKit.logposterior_fisher delegates and forwards prior inputs."""
    seen: dict[str, object] = {}

    def fake_logposterior_fisher(
        *,
        theta,
        theta0,
        fisher,
        prior_terms=None,
        prior_bounds=None,
        logprior=None,
    ):
        """Mock logposterior_fisher that records inputs and returns fixed output."""
        seen["theta"] = np.asarray(theta)
        seen["theta0"] = np.asarray(theta0)
        seen["fisher"] = np.asarray(fisher)
        seen["prior_terms"] = prior_terms
        seen["prior_bounds"] = prior_bounds
        seen["logprior"] = logprior
        return -3.0

    monkeypatch.setattr(
        "derivkit.forecast_kit.logposterior_fisher",
        fake_logposterior_fisher,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    theta = np.array([0.3, 0.4])
    fisher = np.eye(2)
    prior_terms = [("gaussian", {"mu": 0.0, "sigma": 1.0, "idx": 0})]
    prior_bounds = [(None, None), (0.0, 1.0)]

    def lp(_theta):
        return 0.0

    out = fk.logposterior_fisher(
        theta=theta,
        fisher=fisher,
        prior_terms=prior_terms,
        prior_bounds=prior_bounds,
        logprior=lp,
    )

    assert out == -3.0
    np.testing.assert_allclose(seen["theta"], theta)
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["fisher"], fisher)
    assert seen["prior_terms"] == prior_terms
    assert seen["prior_bounds"] == prior_bounds
    assert seen["logprior"] is lp


def test_logposterior_dali_delegates_uses_self_theta0_and_forwards_priors_and_convention(monkeypatch):
    """Tests that ForecastKit.logposterior_dali delegates and forwards priors and convention."""
    seen: dict[str, object] = {}

    def fake_logposterior_dali(
        *,
        theta,
        theta0,
        fisher,
        g_tensor,
        h_tensor,
        convention="delta_chi2",
        prior_terms=None,
        prior_bounds=None,
        logprior=None,
    ):
        """Mock logposterior_dali that records inputs and returns fixed output."""
        seen["theta"] = np.asarray(theta)
        seen["theta0"] = np.asarray(theta0)
        seen["fisher"] = np.asarray(fisher)
        seen["g_tensor"] = np.asarray(g_tensor)
        seen["h_tensor"] = None if h_tensor is None else np.asarray(h_tensor)
        seen["convention"] = convention
        seen["prior_terms"] = prior_terms
        seen["prior_bounds"] = prior_bounds
        seen["logprior"] = logprior
        return -7.0

    monkeypatch.setattr(
        "derivkit.forecast_kit.logposterior_dali",
        fake_logposterior_dali,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    theta = np.array([0.3, 0.4])
    fisher = np.eye(2)
    g = np.zeros((2, 2, 2))
    h = np.ones((2, 2, 2, 2))

    prior_terms = [{"kind": "hard_bounds", "bounds": [(None, None), (-1.0, 1.0)]}]

    out = fk.logposterior_dali(
        theta=theta,
        fisher=fisher,
        g_tensor=g,
        h_tensor=h,
        convention="matplotlib_loglike",
        prior_terms=prior_terms,
        prior_bounds=None,
        logprior=None,
    )

    assert out == -7.0
    np.testing.assert_allclose(seen["theta"], theta)
    np.testing.assert_allclose(seen["theta0"], theta0)
    np.testing.assert_allclose(seen["fisher"], fisher)
    np.testing.assert_allclose(seen["g_tensor"], g)
    np.testing.assert_allclose(seen["h_tensor"], h)
    assert seen["convention"] == "matplotlib_loglike"
    assert seen["prior_terms"] == prior_terms
    assert seen["prior_bounds"] is None
    assert seen["logprior"] is None


def test_negative_logposterior_delegates(monkeypatch):
    """Tests that ForecastKit.negative_logposterior delegates to negative_logposterior."""
    seen: dict[str, object] = {}

    def fake_negative_logposterior(theta, *, logposterior):
        """Mock negative_logposterior that records inputs and returns fixed output."""
        seen["theta"] = np.asarray(theta)
        seen["logposterior"] = logposterior
        return 12.5

    monkeypatch.setattr(
        "derivkit.forecast_kit.negative_logposterior",
        fake_negative_logposterior,
        raising=True,
    )

    fk = ForecastKit(function=None, theta0=np.array([0.0, 1.0]), cov=np.eye(1))

    def logpost(_theta):
        return -1.0

    theta = np.array([0.2, 0.3])
    out = fk.negative_logposterior(theta, logposterior=logpost)

    assert out == 12.5
    np.testing.assert_allclose(seen["theta"], theta)
    assert seen["logposterior"] is logpost


def test_laplace_hessian_delegates_uses_self_theta0_and_forwards_kwargs(monkeypatch):
    """Tests that ForecastKit.laplace_hessian delegates, uses self.theta0, and forwards kwargs."""
    seen: dict[str, object] = {}

    def fake_laplace_hessian(
        *,
        neg_logposterior,
        theta_map,
        method=None,
        n_workers=1,
        **dk_kwargs,
    ):
        """Mock laplace_hessian that records inputs and returns fixed output."""
        seen["neg_logposterior"] = neg_logposterior
        seen["theta_map"] = np.asarray(theta_map)
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["dk_kwargs"] = dk_kwargs
        return np.eye(2)

    monkeypatch.setattr(
        "derivkit.forecast_kit.laplace_hessian",
        fake_laplace_hessian,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    def neglogpost(_theta):
        """Mock logposterior that returns fixed output."""
        return 0.0

    out = fk.laplace_hessian(
        neg_logposterior=neglogpost,
        theta_map=None,
        method="finite",
        n_workers=3,
        step=1e-4,
    )

    np.testing.assert_allclose(out, np.eye(2))
    assert seen["neg_logposterior"] is neglogpost
    np.testing.assert_allclose(seen["theta_map"], theta0)
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 3
    assert seen["dk_kwargs"]["step"] == 1e-4


def test_laplace_hessian_delegates_uses_theta_map_override(monkeypatch):
    """Tests that ForecastKit.laplace_hessian uses theta_map when provided."""
    seen: dict[str, object] = {}

    def fake_laplace_hessian(*, theta_map, **_kwargs):
        """Mock laplace_hessian that records theta_map and returns fixed output."""
        seen["theta_map"] = np.asarray(theta_map)
        return np.eye(1)

    monkeypatch.setattr(
        "derivkit.forecast_kit.laplace_hessian",
        fake_laplace_hessian,
        raising=True,
    )

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=np.eye(1))

    def neglogpost(_theta):
        """Mock logposterior that returns fixed output."""
        return 0.0

    theta_map = np.array([9.0, 8.0])
    fk.laplace_hessian(neg_logposterior=neglogpost, theta_map=theta_map)

    np.testing.assert_allclose(seen["theta_map"], theta_map)


def test_laplace_covariance_delegates(monkeypatch):
    """Tests that ForecastKit.laplace_covariance delegates to laplace_covariance."""
    seen: dict[str, object] = {}

    def fake_laplace_covariance(hessian, *, rcond=1e-12):
        """Mock laplace_covariance that records inputs and returns fixed output."""
        seen["hessian"] = np.asarray(hessian)
        seen["rcond"] = rcond
        return np.full((2, 2), 3.0)

    monkeypatch.setattr(
        "derivkit.forecast_kit.laplace_covariance",
        fake_laplace_covariance,
        raising=True,
    )

    fk = ForecastKit(function=None, theta0=np.array([0.0]), cov=np.eye(1))

    hess = np.eye(2)
    out = fk.laplace_covariance(hessian=hess, rcond=1e-9)

    np.testing.assert_allclose(out, np.full((2, 2), 3.0))
    np.testing.assert_allclose(seen["hessian"], hess)
    assert seen["rcond"] == 1e-9


def test_laplace_approximation_delegates_uses_self_theta0_and_forwards_kwargs(monkeypatch):
    """Tests that ForecastKit.laplace_approximation delegates, uses self.theta0, and forwards kwargs."""
    seen: dict[str, object] = {}

    def fake_laplace_approximation(
        *,
        neg_logposterior,
        theta_map,
        method=None,
        n_workers=1,
        ensure_spd=True,
        rcond=1e-12,
        **dk_kwargs,
    ):
        """Mock laplace_approximation that records inputs and returns fixed output."""
        seen["neg_logposterior"] = neg_logposterior
        seen["theta_map"] = np.asarray(theta_map)
        seen["method"] = method
        seen["n_workers"] = n_workers
        seen["ensure_spd"] = ensure_spd
        seen["rcond"] = rcond
        seen["dk_kwargs"] = dk_kwargs
        return {"ok": True}

    monkeypatch.setattr(
        "derivkit.forecast_kit.laplace_approximation",
        fake_laplace_approximation,
        raising=True,
    )

    theta0 = np.array([0.1, -0.2])
    fk = ForecastKit(function=None, theta0=theta0, cov=np.eye(1))

    def neglogpost(_theta):
        return 0.0

    out = fk.laplace_approximation(
        neg_logposterior=neglogpost,
        theta_map=None,
        method="finite",
        n_workers=4,
        ensure_spd=False,
        rcond=1e-9,
        step=1e-4,
    )

    assert out == {"ok": True}
    assert seen["neg_logposterior"] is neglogpost
    np.testing.assert_allclose(seen["theta_map"], theta0)
    assert seen["method"] == "finite"
    assert seen["n_workers"] == 4
    assert seen["ensure_spd"] is False
    assert seen["rcond"] == 1e-9
    assert seen["dk_kwargs"]["step"] == 1e-4


def test_laplace_approximation_delegates_uses_theta_map_override(monkeypatch):
    """Tests that ForecastKit.laplace_approximation uses theta_map when provided."""
    seen: dict[str, object] = {}

    def fake_laplace_approximation(*, theta_map, **_kwargs):
        """Mock laplace_approximation that records theta_map and returns fixed output."""
        seen["theta_map"] = np.asarray(theta_map)
        return {"ok": True}

    monkeypatch.setattr(
        "derivkit.forecast_kit.laplace_approximation",
        fake_laplace_approximation,
        raising=True,
    )

    fk = ForecastKit(function=None, theta0=np.array([0.1, -0.2]), cov=np.eye(1))

    def neglogpost(_theta):
        return 0.0

    theta_map = np.array([7.0, 6.0])
    fk.laplace_approximation(neg_logposterior=neglogpost, theta_map=theta_map)

    np.testing.assert_allclose(seen["theta_map"], theta_map)
