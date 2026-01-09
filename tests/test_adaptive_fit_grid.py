"""Unit tests for grid generation."""

from __future__ import annotations

import numpy as np
import pytest

from derivkit.derivatives.adaptive.grid import make_offsets

RTOL = 5e-13  # relative tolerance for spacing checks
ATOL_SCALE = 1e-12  # absolute tolerance scale factor


@pytest.mark.parametrize("n_points", [10, 11, 12])
@pytest.mark.parametrize("base", [1.0, 0.1, 1e-6, 3.7e-5])
def test_both_center_free_properties_even_and_odd(n_points, base):
    """Test properties of center-free grids with 'both' direction."""
    t = make_offsets(n_points=n_points, base=base, direction="both")

    # center-free
    assert not np.any(np.isclose(t, 0.0, rtol=0, atol=base * ATOL_SCALE))

    # counts on each side differ by at most 1
    n_neg = int(np.sum(t < 0))
    n_pos = int(np.sum(t > 0))
    assert abs(n_neg - n_pos) <= 1

    # diffs: all base except one central gap of 2*base (between -base and +base)
    ts = np.sort(t)
    diffs = np.diff(ts)
    q = diffs / base
    is_one = np.isclose(q, 1.0, rtol=RTOL, atol=ATOL_SCALE)
    is_two = np.isclose(q, 2.0, rtol=RTOL, atol=ATOL_SCALE)
    assert np.count_nonzero(is_two) == 1
    assert np.all(is_one | is_two)

    # multiset symmetry: for each positive value, a matching negative exists
    tol = base * ATOL_SCALE
    pos = np.sort(t[t > 0])
    neg = np.sort(-t[t < 0])  # reflect negatives to positive side
    # sizes should match for even; for odd, they differ by 1 but smallest |t| == base
    if n_points % 2 == 0:
        assert pos.size == neg.size
    else:
        assert abs(pos.size - neg.size) == 1
        # ensure the unmatched side's smallest magnitude equals base
        assert np.isclose(np.min(np.abs(t)), base, rtol=RTOL, atol=tol)

    # pairwise match within tolerance (min length to be safe)
    m = min(pos.size, neg.size)
    assert np.allclose(pos[:m], neg[:m], rtol=RTOL, atol=tol)


@pytest.mark.parametrize("base", [1.0, 0.1, 1e-6, 3.7e-5])
def test_pos_and_neg_shapes_and_steps(base):
    """Test properties of one-sided grids with 'pos' and 'neg' directions."""
    tp = make_offsets(n_points=7, base=base, direction="pos")
    tn = make_offsets(n_points=7, base=base, direction="neg")

    assert tp.shape == (7,) and tn.shape == (7,)
    assert np.all(tp > 0) and np.all(tn < 0)

    # normalized step ratios
    qpos = np.diff(tp) / base
    qneg = np.diff(tn) / base
    assert np.allclose(qpos, 1.0, rtol=RTOL, atol=ATOL_SCALE)
    assert np.allclose(qneg, -1.0, rtol=RTOL, atol=ATOL_SCALE)
