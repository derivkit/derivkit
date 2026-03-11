# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**DerivKit** is a Python toolkit for stable numerical derivatives, targeting scientific computing and cosmology. It bridges Fisher forecasts and MCMC through accurate gradients and higher-order expansions.

## Commands

### Install (editable dev mode)
```bash
pip install -e ".[jax]"
pip install --group dev
```

### Run tests
```bash
pytest tests/                          # all tests
pytest tests/test_derivative_kit.py   # single file
pytest -m "not slow"                  # skip slow tests
```

### Lint
```bash
ruff check .
```

### Docs
```bash
tox -e docs        # build HTML docs
tox -e do          # build + open (Mac only)
```

### Tox (multi-env)
```bash
tox -l             # list environments
tox -e lint        # lint only
tox -e py313       # test under Python 3.13
```

## Architecture

The package is under `src/derivkit/` with four public Kit classes exposed from `__init__.py`:

### Kit Classes (public API)
- **`DerivativeKit`** (`derivative_kit.py`): Unified front-end for 1D scalar derivatives. Resolves method names via an alias registry to one of the derivative engine backends. Supports array `x0` for evaluating at multiple points.
- **`CalculusKit`** (`calculus_kit.py`): Gradient, Jacobian, Hessian, and hyper-Hessian for multi-parameter functions. Delegates to `calculus/` internally. Has optional thread-safety via lock wrapping.
- **`ForecastKit`** (`forecast_kit.py`): Fisher matrices, DALI tensors, Laplace approximations, GetDist samples, and bias estimation. Wraps `forecasting/`.
- **`LikelihoodKit`** (`likelihood_kit.py`): Gaussian and Poisson likelihood helpers. Wraps `likelihoods/`.

### Derivative Backends (`derivatives/`)
- **`adaptive/`**: Adaptive fit derivative — evaluates at a grid of step sizes, fits a polynomial, and selects the best result.
- **`finite/`**: Finite difference derivative, with optional Richardson extrapolation (`extrapolators.py`) and stencil helpers.
- **`local_polynomial_derivative/`**: Local polynomial regression-based derivative with configurable sampling.
- **`fornberg.py`**: Fornberg's algorithm for computing finite difference weights.
- **`autodiff/`**: JAX-based automatic differentiation (optional; requires `jax` extra).
- **`tabulated_model/`**: Wraps tabulated `(x, y)` data as a callable for use with any backend.

### Calculus Layer (`calculus/`)
Builds multi-parameter derivative objects by iterating `DerivativeKit` over parameter components. `calculus_core.py` provides `component_scalar_eval` for slicing vector-valued outputs into scalars.

### Forecasting Layer (`forecasting/`)
Contains `fisher.py`, `dali.py`, `expansions.py`, `laplace.py`, `priors_core.py`, `fisher_gaussian.py`, `fisher_xy.py`, and GetDist sample generators. Fisher and DALI computations accept `n_workers` for parallelism via `utils/concurrency.py`.

### Utilities (`utils/`)
- `concurrency.py`: `parallel_execute` — Dask-based parallel execution (default backend `"dask"`). The `"processes"` backend (legacy `multiprocessing.Pool`) remains available. Inner worker counts propagate via `contextvars`.
- `caching.py`: Function-call caching helpers.
- `thread_safety.py`: Lock-wrapping for thread-safe function evaluation.
- `types.py`, `validate.py`, `numerics.py`, `linalg.py`, `extrapolation.py`, `sandbox.py`, `logger.py`.

## Key Conventions

- **Ruff** is the linter; line length 79; docstrings follow Google convention (`pydocstyle`).
- **`n_workers`** controls parallel execution in `ForecastKit` and calculus builders via `parallel_execute`. The default backend is **Dask** (`dask.delayed` + `dask.compute`). Nested parallelism (outer calculus loop + inner derivative stencil) is supported with Dask.
- **Dask scheduler configuration**: the default scheduler is threaded (good for NumPy/BLAS that release the GIL). For CPU-bound pure-Python work, set `dask.config.set(scheduler="processes")`. For HPC clusters, instantiate a `dask.distributed.Client` — nested `dask.compute` calls inside workers are fully supported.
- **`pip install "derivkit[distributed]"`** pulls in `dask[distributed]` for cluster use.
- **Thread-safety** in `CalculusKit` is opt-in via `thread_safe=True`; uses an `RLock` wrapper.
- **JAX autodiff** is optional — guarded by `try/except ImportError` and requires `pip install "derivkit[jax]"`.
- Slow tests are marked with `@pytest.mark.slow`; skip them with `-m "not slow"`.
- Source layout uses `src/` layout; `pytest.ini_options.pythonpath = ["src"]`.
- New derivative engines can be registered at runtime via `derivkit.derivative_kit.register_method(...)` without modifying core files.
