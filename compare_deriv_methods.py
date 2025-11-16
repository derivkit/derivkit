from __future__ import annotations

import math
import time
from typing import Any

from derivkit.derivative_kit import DerivativeKit

# -------- configuration (no 'order' here!) --------

METHOD_CONFIGS = [
    # finite, no extrapolation
    (
        "finite",
        {
            "stepsize": 1.0e-4,
            "num_points": 5,
            "extrapolation": None,
        },
    ),
    # finite + Richardson
    (
        "finite",
        {
            "stepsize": 1.0e-4,
            "num_points": 5,
            "extrapolation": "richardson",
            "levels": 3,
        },
    ),
    # finite + Ridders
    (
        "finite",
        {
            "stepsize": 1.0e-4,
            "num_points": 5,
            "extrapolation": "ridders",
            "levels": 3,
        },
    ),
    # local polynomial: *no* 'order' here, that comes from dk.differentiate(...)
    (
        "local_polynomial",
        {
            "degree": 1,

        },
    ),
    # adaptive
    (
        "adaptive",
        {},
    ),
]


def relative_error(estimate: float, truth: float) -> float:
    denom = max(1.0, abs(estimate), abs(truth))
    return abs(estimate - truth) / denom


def run_single_method(
    method: str,
    method_kwargs: dict[str, Any],
    x0: float,
    order: int,
) -> tuple[float, float, float]:
    """Run one method on cos(x) and return (estimate, rel_error, runtime_s)."""
    f = math.cos
    truth = -math.sin(x0)

    dk = DerivativeKit(f, x0)
    # IMPORTANT: 'order' is passed only here, not in method_kwargs
    t0 = time.perf_counter()
    est = dk.differentiate(method=method, order=order, **method_kwargs)
    dt = time.perf_counter() - t0

    err = relative_error(est, truth)
    return est, err, dt


def main() -> None:
    x0 = 0.3
    order = 1

    print(f"Comparing methods on cos at x0={x0}, order={order}")
    for method, method_kwargs in METHOD_CONFIGS:
        est, err, dt = run_single_method(method, method_kwargs, x0, order)
        print(
            f"{method:16} {str(method_kwargs):40} "
            f"est={est:+.8e}  err={err:.3e}  time={dt*1e3:7.3f} ms"
        )


if __name__ == "__main__":
    main()
