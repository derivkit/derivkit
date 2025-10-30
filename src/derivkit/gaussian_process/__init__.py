"""Gaussian Process subpackage for DerivKit.

This package provides a local Gaussian Process implementation used for
derivative estimation. It exposes the high-level :class:`GaussianProcess`
model and its corresponding DerivativeKit engine adapter
(:mod:`derivkit.gaussian_process.gp_engine`).

Modules:
    gp_engine: DerivativeKit engine adapter that wraps a Gaussian Process model.
    gaussian_process: Core implementation of the local GP derivative estimator.

Public API:
    - GaussianProcess
    - gp_engine
"""

from __future__ import annotations

from derivkit.gaussian_process import gp_engine
from derivkit.gaussian_process.gaussian_process import GaussianProcess

__all__ = ["GaussianProcess", "gp_engine"]
