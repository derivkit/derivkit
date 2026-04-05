"""Numerical health checks for cached vs uncached forecasting outputs.

Run:
    pytest -q tests/benchmarks/test_cache_forecasting_numerics.py -m slow -s
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


def make_theta0(p: int, *, scale: float = 1.0) -> np.ndarray:
    """Return a deterministic test point of length ``p``."""
    base = np.linspace(0.25, 0.85, p, dtype=float)
    signs = np.where(np.arange(p) % 2 == 0, 1.0, -1.0)
    return scale * base * signs


def expensive_forecast_model_factory(
    p: int,
    n_obs: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a deterministic nonlinear vector-valued forecasting model."""
    weights = np.linspace(0.7, 1.5, n_obs, dtype=float)

    def model(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != p:
            raise ValueError(f"Expected theta with size {p}, got {theta.size}.")

        grid = np.linspace(0.0, 1.0, 5000)
        w = np.exp(-0.5 * grid)

        total_theta = np.sum(theta)
        out = np.empty(n_obs, dtype=float)

        for k in range(n_obs):
            acc = 0.0
            for i in range(p):
                phase = theta[i] + 0.08 * (i + 1) * (k + 1)
                acc += np.sum(np.sin(phase * grid) * w)
                acc += 0.2 * weights[k] * theta[i] ** 2

            for i in range(p - 1):
                acc += 0.15 * (k + 1) * theta[i] * theta[i + 1]
                acc += np.sum(
                    np.cos((theta[i] - 0.25 * theta[i + 1] + 0.05 * k) * grid) * w
                )

            acc += np.exp(0.03 * (k + 1) * total_theta)
            out[k] = acc

        return out

    return model


def make_spd_matrix(n: int, diag_boost: float = 1.0) -> np.ndarray:
    """Return a deterministic symmetric positive definite matrix."""
    a = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    mat = a @ a.T
    mat /= np.max(np.abs(mat))
    mat += diag_boost * np.eye(n, dtype=float)
    return mat


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return relative Frobenius difference between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return float(np.linalg.norm(a - b) / denom)


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return max absolute difference between two arrays."""
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def fisher_health_summary(matrix: np.ndarray) -> dict[str, float]:
    """Return a compact numerical-health summary for a Fisher matrix."""
    matrix = np.asarray(matrix, dtype=float)

    sym_err = float(np.max(np.abs(matrix - matrix.T)))
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    cond = float(np.linalg.cond(matrix))
    frob = float(np.linalg.norm(matrix))

    return {
        "symmetry_error": sym_err,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond,
        "frobenius_norm": frob,
    }


def dali_health_summary(dali: dict[int, tuple[np.ndarray, ...]]) -> dict[str, float]:
    """Return compact numerical-health summaries for a DALI tensor dictionary."""
    values: list[np.ndarray] = []

    for order in sorted(dali):
        for tensor in dali[order]:
            arr = np.asarray(tensor, dtype=float)
            values.append(arr.ravel())

    flat = np.concatenate(values) if values else np.array([], dtype=float)

    return {
        "frobenius_norm": float(np.linalg.norm(flat)),
        "max_abs_entry": float(np.max(np.abs(flat))) if flat.size else 0.0,
        "num_entries": float(flat.size),
    }


def print_fisher_comparison(
    label: str,
    uncached: np.ndarray,
    cached: np.ndarray,
) -> None:
    """Print a compact Fisher comparison summary."""
    h0 = fisher_health_summary(uncached)
    h1 = fisher_health_summary(cached)

    print("=" * 80)
    print(label)
    print("=" * 80)
    print(
        "uncached | "
        f"sym={h0['symmetry_error']:.3e} | "
        f"min_eig={h0['min_eigenvalue']:.3e} | "
        f"max_eig={h0['max_eigenvalue']:.3e} | "
        f"cond={h0['condition_number']:.3e} | "
        f"||F||={h0['frobenius_norm']:.3e}"
    )
    print(
        "cached   | "
        f"sym={h1['symmetry_error']:.3e} | "
        f"min_eig={h1['min_eigenvalue']:.3e} | "
        f"max_eig={h1['max_eigenvalue']:.3e} | "
        f"cond={h1['condition_number']:.3e} | "
        f"||F||={h1['frobenius_norm']:.3e}"
    )
    print(
        "diff     | "
        f"rel={relative_difference(uncached, cached):.3e} | "
        f"max_abs={max_abs_difference(uncached, cached):.3e}"
    )
    print()


def print_dali_comparison(
    label: str,
    uncached: dict[int, tuple[np.ndarray, ...]],
    cached: dict[int, tuple[np.ndarray, ...]],
) -> None:
    """Print a compact DALI comparison summary."""
    h0 = dali_health_summary(uncached)
    h1 = dali_health_summary(cached)

    print("=" * 80)
    print(label)
    print("=" * 80)
    print(
        "uncached | "
        f"||D||={h0['frobenius_norm']:.3e} | "
        f"max_abs={h0['max_abs_entry']:.3e} | "
        f"entries={int(h0['num_entries'])}"
    )
    print(
        "cached   | "
        f"||D||={h1['frobenius_norm']:.3e} | "
        f"max_abs={h1['max_abs_entry']:.3e} | "
        f"entries={int(h1['num_entries'])}"
    )

    for order in sorted(uncached):
        for idx, (u, c) in enumerate(zip(uncached[order], cached[order], strict=True)):
            print(
                f"order={order} tensor={idx} | "
                f"rel={relative_difference(u, c):.3e} | "
                f"max_abs={max_abs_difference(u, c):.3e}"
            )
    print()


METHOD_CASES = [
    pytest.param("adaptive", {}, id="adaptive"),
    pytest.param("polyfit", {}, id="polyfit"),
    pytest.param("finite", {}, id="finite"),
    pytest.param("finite", {"extrapolation": "ridders"}, id="finite-ridders"),
]

FORECAST_SIZE_CASES = [
    pytest.param(2, 4, id="p2-n4"),
    pytest.param(3, 6, id="p3-n6"),
]

DALI_SIZE_CASES = [
    pytest.param(2, 4, id="p2-n4"),
    pytest.param(3, 5, id="p3-n5"),
]


@pytest.mark.parametrize(("p", "n_obs"), FORECAST_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_fisher_cache_numerical_health(
    p: int,
    n_obs: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Check that caching preserves Fisher numerical health."""
    theta0 = make_theta0(p, scale=0.8)
    cov = make_spd_matrix(n_obs, diag_boost=2.0)
    model = expensive_forecast_model_factory(p, n_obs)

    fk_no_cache = ForecastKit(
        function=model,
        theta0=theta0,
        cov=cov,
        use_input_cache=False,
    )
    fk_with_cache = ForecastKit(
        function=model,
        theta0=theta0,
        cov=cov,
        use_input_cache=True,
    )

    fisher_uncached = np.asarray(
        fk_no_cache.fisher(method=method, n_workers=1, **method_kwargs),
        dtype=float,
    )
    fisher_cached = np.asarray(
        fk_with_cache.fisher(method=method, n_workers=1, **method_kwargs),
        dtype=float,
    )

    print_fisher_comparison(
        f"Fisher health | p={p} | n_obs={n_obs} | method={method}",
        fisher_uncached,
        fisher_cached,
    )

    assert np.all(np.isfinite(fisher_uncached))
    assert np.all(np.isfinite(fisher_cached))

    np.testing.assert_allclose(
        fisher_cached,
        fisher_uncached,
        rtol=1e-10,
        atol=1e-12,
    )

    sym_uncached = np.max(np.abs(fisher_uncached - fisher_uncached.T))
    sym_cached = np.max(np.abs(fisher_cached - fisher_cached.T))

    assert sym_uncached < 1e-8
    assert sym_cached < 1e-8
    assert abs(sym_cached - sym_uncached) < 1e-8

    fisher_uncached_sym = 0.5 * (fisher_uncached + fisher_uncached.T)
    fisher_cached_sym = 0.5 * (fisher_cached + fisher_cached.T)

    cond_uncached = np.linalg.cond(fisher_uncached_sym)
    cond_cached = np.linalg.cond(fisher_cached_sym)

    assert np.isfinite(cond_uncached)
    assert np.isfinite(cond_cached)

    if cond_uncached > 0.0:
        assert abs(cond_cached - cond_uncached) / cond_uncached < 1e-10


@pytest.mark.parametrize(("p", "n_obs"), DALI_SIZE_CASES)
@pytest.mark.parametrize(("method", "method_kwargs"), METHOD_CASES)
def test_dali_cache_numerical_health(
    p: int,
    n_obs: int,
    method: str,
    method_kwargs: dict[str, Any],
) -> None:
    """Check that caching preserves DALI tensor numerical health."""
    theta0 = make_theta0(p, scale=0.6)
    cov = make_spd_matrix(n_obs, diag_boost=2.5)
    model = expensive_forecast_model_factory(p, n_obs)

    fk_no_cache = ForecastKit(
        function=model,
        theta0=theta0,
        cov=cov,
        use_input_cache=False,
    )
    fk_with_cache = ForecastKit(
        function=model,
        theta0=theta0,
        cov=cov,
        use_input_cache=True,
    )

    dali_uncached = fk_no_cache.dali(
        method=method,
        forecast_order=2,
        n_workers=1,
        **method_kwargs,
    )
    dali_cached = fk_with_cache.dali(
        method=method,
        forecast_order=2,
        n_workers=1,
        **method_kwargs,
    )

    print_dali_comparison(
        f"DALI health | p={p} | n_obs={n_obs} | method={method}",
        dali_uncached,
        dali_cached,
    )

    assert set(dali_uncached) == set(dali_cached)

    for order in sorted(dali_uncached):
        uncached_tensors = dali_uncached[order]
        cached_tensors = dali_cached[order]

        assert len(uncached_tensors) == len(cached_tensors)

        for uncached_tensor, cached_tensor in zip(
            uncached_tensors, cached_tensors, strict=True
        ):
            uncached_arr = np.asarray(uncached_tensor, dtype=float)
            cached_arr = np.asarray(cached_tensor, dtype=float)

            assert np.all(np.isfinite(uncached_arr))
            assert np.all(np.isfinite(cached_arr))

            np.testing.assert_allclose(
                cached_arr,
                uncached_arr,
                rtol=1e-9,
                atol=1e-11,
            )
