"""Shared typing aliases for DerivKit."""

from __future__ import annotations

from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

Float: TypeAlias = np.floating
Array: TypeAlias = NDArray[np.floating]

ArrayLike1D: TypeAlias = Sequence[float] | NDArray[np.floating]
ArrayLike2D: TypeAlias = Sequence[Sequence[float]] | NDArray[np.floating]
