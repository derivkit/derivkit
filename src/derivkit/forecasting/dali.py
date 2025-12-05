"""Provides tools for computing the DALI tensors.

The user must specify the observables, fiducial values, and covariance matrix
at which the forecast should be evaluated.

More details about available options can be found in the documentation of
the methods.
"""

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from derivkit.calculus_kit import CalculusKit
from derivkit.utils.linalg import solve_or_pinv


def _build_dali(
        expansion,
        d1: NDArray[np.float64],
        d2: NDArray[np.float64],
        invcov: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the doublet-DALI tensors (G, H) from first- and second-order derivatives.

    Computes:
        G_abc = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d1[c,j]
        H_abcd = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d2[c,d,j]

    Args:
        d1: First-order derivatives of the observables with respect to parameters,
            shape (P, N).
        d2: Second-order derivatives of the observables with respect to parameters,
            shape (P, P, N).
        invcov: Inverse covariance matrix of the observables, shape (N, N).

    Returns:
        A tuple ``(G, H)`` where:
            - G has shape (P, P, P)
            - H has shape (P, P, P, P)
    """
    # G_abc = Σ_ij d2[a,b,i] invcov[i,j] d1[c,j]
    g_tensor = np.einsum("abi,ij,cj->abc", d2, invcov, d1)
    # H_abcd = Σ_ij d2[a,b,i] invcov[i,j] d2[c,d,j]
    h_tensor = np.einsum("abi,ij,cdj->abcd", d2, invcov, d2)
    return g_tensor, h_tensor
