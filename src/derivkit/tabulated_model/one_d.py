"""Interpolation utilities for 1D tabulated functions.

Provides a lightweight wrapper around ``numpy.interp`` for interpolating
tabulated ``y(x)`` data, supporting scalar as well as vector- and
tensor-valued outputs.

Two common entry points are:

* Direct construction with ``(x, y)`` arrays of shape ``(N,)`` and ``(N, ...)``.
* :func:`tabulated1d_from_table` for simple 2D tables containing x and one
  or more y components in columns.
"""


from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from derivkit.derivative_kit import DerivativeKit
from derivkit.utils.validate import validate_tabulated_xy

__all__ = ["Tabulated1DModel", "tabulated1d_from_table"]


class Tabulated1DModel:
    """1D interpolator for tabulated data.

    Interpolates a tabulated function ``y(x)`` using ``numpy.interp``.

    Here ``x`` is a one-dimensional grid of length ``N``, and the first
    dimension of ``y`` must also have length ``N``. All remaining
    dimensions of ``y`` are treated as the output shape.

    For example:
        * ``x`` has shape ``(N,)`` and ``y`` has shape ``(N,)``        -> scalar output
        * ``x`` has shape ``(N,)`` and ``y`` has shape ``(N, M)``      -> vector output of length ``M``
        * ``x`` has shape ``(N,)`` and ``y`` has shape ``(N, d1, d2)`` -> tensor output with shape ``(d1, d2)``

    This class handles interpolation for functions of a single scalar input ``x``.
    Support for tabulated functions with multi-dimensional inputs would require
    a different interface and can be added in a future extension.

    Attributes:
        x: Tabulated x grid, strictly increasing.
        y_flat: Flattened tabulated values with shape ``(N, n_out_flat)``.
        _out_shape: Original trailing output shape of ``y``.
        extrapolate: Whether evaluation outside the tabulated x range is allowed.
        fill_value: Value used for out-of-range evaluation when extrapolation is off.

    Example:
        >>> import numpy as np
        >>> from derivkit.tabulated_model import Tabulated1DModel
        >>>
        >>> x_tab = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y_tab = np.array([[0.0, 0.0],
        ...                   [1.0, 1.0],
        ...                   [4.0, 8.0],
        ...                   [9.0, 27.0]])  # shape (4, 2)
        >>>
        >>> model = Tabulated1DModel(x_tab, y_tab,
        ...                          extrapolate=False, fill_value=-1.0)
        >>>
        >>> x_new = np.array([-1.0, 0.5, 1.5, 2.5, 4.0])
        >>> y_new = model(x_new)
        >>> print(y_new)
        [[-1.   -1.  ]
         [ 0.5  0.5 ]
         [ 2.5  4.5 ]
         [ 6.5 17.5 ]
         [-1.   -1.  ]]
    """
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        extrapolate: bool = False,
        fill_value: float | None = None,
    ) -> None:
        """Initializes a tabulated 1D interpolation model.

        Args:
            x: Strictly increasing tabulated x values with shape ``(N,)``.
            y: Tabulated y values with shape ``(N,)`` (scalar) or ``(N, ...)`` (vector/tensor).
               The first dimension must match ``x``.
            extrapolate: Whether to allow evaluation outside the x range. Default is False.
            fill_value: Value for out-of-bounds evaluation when extrapolation is
                disabled. If ``None``, a ``ValueError`` is raised instead.
        """
        x_arr, y_arr = validate_tabulated_xy(x, y)

        self.x = x_arr

        if y_arr.ndim == 1:
            self._out_shape: tuple[int, ...] = ()
            y_flat = y_arr[:, np.newaxis]
        else:
            self._out_shape = tuple(y_arr.shape[1:])
            y_flat = y_arr.reshape(y_arr.shape[0], -1)

        self.y_flat = np.asarray(y_flat, dtype=float)
        self.extrapolate = extrapolate
        self.fill_value = fill_value

    def differentiate(
            self,
            x0: float | np.ndarray,
            **dk_kwargs: Any,
    ) -> Any:
        """Differentiates the tabulated model at x0 using DerivativeKit.

        This runs DerivKit's derivative engines on the interpolated function
        defined by this model. All additional keyword arguments are forwarded
        to `DerivativeKit.differentiate`.

        Examples:
        ---------
        >>> import numpy as np
        >>> from derivkit.tabulated_model import Tabulated1DModel
        >>>
        >>> x_tab = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y_tab = np.array([0.0, 1.0, 4.0, 9.0])  # y = x^2
        >>>
        >>> model = Tabulated1DModel(x_tab, y_tab)
        >>>
        >>> # First derivative at a single point.
        >>> # For y = x**2 we have dy/dx = 2*x, so at x=0.5 this is 1.0:
        >>> d1 = model.differentiate(x0=0.5, order=1)
        >>> float(d1)
        1.0
        >>>
        >>> xs = np.linspace(0.0, 1.0, 5)
        >>> d2 = model.differentiate(x0=xs, method="finite", order=2)
        >>> np.allclose(d2, 2.0)
        True
        """
        dk = DerivativeKit(function=lambda x: self(x), x0=x0)
        return dk.differentiate(**dk_kwargs)

    def __call__(self, x_new: ArrayLike) -> NDArray[np.floating]:
        """Evaluates the interpolated function at the given x values.

        Args:
            x_new: Points where the function should be interpolated.

        Returns:
            Interpolated values with shape matching ``x_new`` and the original
            y output shape.

        Raises:
            ValueError: If evaluating outside the x range with extrapolation disabled
                and no fill_value is provided.
        """
        x_new_arr = np.asarray(x_new, dtype=float)
        flat_x = x_new_arr.ravel()
        n_out_flat = self.y_flat.shape[1]

        flat_y = np.empty((flat_x.size, n_out_flat), dtype=float)
        for k in range(n_out_flat):
            flat_y[:, k] = np.interp(flat_x, self.x, self.y_flat[:, k])

        if self._out_shape:
            new_shape = x_new_arr.shape + self._out_shape
        else:
            new_shape = x_new_arr.shape

        y_interp = flat_y.reshape(new_shape)

        if not self.extrapolate:
            mask = (x_new_arr < self.x[0]) | (x_new_arr > self.x[-1])
            if np.any(mask):
                if self.fill_value is None:
                    raise ValueError(
                        "Requested x outside tabulated range and extrapolate=False."
                    )
                y_interp = np.asarray(y_interp)
                y_interp[mask, ...] = self.fill_value

        return np.asarray(y_interp, dtype=float)


def tabulated1d_from_table(
    table: ArrayLike,
    *,
    extrapolate: bool = False,
    fill_value: float | None = None,
) -> Tabulated1DModel:
    """Creates a Tabulated1DModel from a simple 2D ``(x, y)`` table.

    This helper covers the common case where data are stored in a 2D array
    or text file with x in one column and one or more y components in the
    remaining columns.

    Supported layouts:
        * ``(N, 2)``: column 0 = x, column 1 = scalar y.
        * ``(N, M+1)``: column 0 = x, columns 1..M = components of y.
        * ``(2, N)``: row 0 = x, row 1 = scalar y.

    For multi-component tables, the resulting ``y`` has shape ``(N, M)``.
    Higher-rank outputs (for example matrices or tensors associated with
    each x) are not encoded via this table format. Such data must be loaded
    and reshaped prior to constructing a :class:`Tabulated1DModel` directly
    from ``(x, y)``.

    Args:
        table: 2D array containing x and y columns.
        extrapolate: Whether to allow evaluation outside the tabulated x range.
            If False (default), any x outside [x.min(), x.max()] either raises
            a ValueError (when fill_value is None) or is set to fill_value.
            If True, out-of-range x are handled using numpy.interp’s standard
            edge behaviour.
        fill_value: Value for out-of-range evaluation when extrapolation is
            disabled.

    Returns:
        A :class:`Tabulated1DModel` constructed from the parsed table.
    """
    x, y = parse_xy_table(table)
    return Tabulated1DModel(x, y, extrapolate=extrapolate, fill_value=fill_value)


def parse_xy_table(
    table: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Parses a 2D table into ``(x, y)`` arrays.

    Supported layouts:

    * ``(N, 2)``:
        Column 0 = x, column 1 = scalar y.
    * ``(N, M+1)``:
        Column 0 = x, columns 1..M = components of y, returned with shape ``(N, M)``.
    * ``(2, N)``:
        Row 0 = x, row 1 = scalar y.

    In all cases ``x`` is returned as a 1D array of length ``N``. For
    multi-component tables, ``y`` is returned as a 2D array of shape ``(N, M)``.

    This function does not handle tensor-valued outputs directly.
    When each x corresponds to a matrix or higher-rank object, the data
    must be parsed and reshaped prior to constructing a
    :class:`Tabulated1DModel`.

    Args:
        table: 2D array containing x and y columns (e.g. data loaded from a text
            file with shape (N, 2) or (N, M+1)). For higher-rank outputs,
            reshape your data into (N, ...) and call ``Tabulated1DModel(x, y)``
            directly.

    Returns:
        A tuple ``(x, y)`` as NumPy arrays.

    Raises:
        ValueError: If the input does not match any of the supported layouts.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError("table must be a 2D array.")

    n_rows, n_cols = arr.shape

    match (n_rows, n_cols):
        case (2, n) if n >= 2:
            # row 0 = x, row 1 = y
            x = arr[0, :]
            y = arr[1, :]
        case (n, m) if n >= 2 and m >= 2:
            # column 0 = x, remaining columns = y components
            x = arr[:, 0]
            y = arr[:, 1] if m == 2 else arr[:, 1:]
        case _:
            raise ValueError(
                f"Unexpected table shape {arr.shape}; expected (N, 2), (N, M+1) or (2, N)."
            )

    return x, y
