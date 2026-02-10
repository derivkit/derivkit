"""Provides the DerivativeKit API.

This class is a lightweight front end over DerivKit’s derivative engines.
You provide the function to differentiate and the expansion point `x0`,
then choose a backend by name (e.g., ``"adaptive"`` or ``"finite"``).

Examples:
---------
Basic usage:

    >>> import numpy as np
    >>> from derivkit import DerivativeKit
    >>> dk = DerivativeKit(function=np.cos, x0=1.0)
    >>> dk.differentiate(method="adaptive", order=1)  # doctest: +SKIP

Using tabulated data directly:

    >>> import numpy as np
    >>> from derivkit import DerivativeKit
    >>>
    >>> x_tab = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y_tab = x_tab**2
    >>> dk = DerivativeKit(x0=0.5, tab_x=x_tab, tab_y=y_tab)
    >>> dk.differentiate(order=1, method="finite", extrapolation="ridders")  # doctest: +SKIP

Listing built-in aliases:

    >>> from derivkit.derivative_kit import available_methods
    >>> available_methods()  # doctest: +SKIP

Adding methods:
---------------
New engines can be registered without modifying this class by calling
:func:`derivkit.derivative_kit.register_method` (see example below).

Registering a new method:

    >>> from derivkit.derivative_kit import register_method  # doctest: +SKIP
    >>> from derivkit.some_new_method import NewMethodDerivative  # doctest: +SKIP
    >>> register_method(  # doctest: +SKIP
    ...     name="new-method",
    ...     cls=NewMethodDerivative,
    ...     aliases=("new_method", "nm"),
    ... )  # doctest: +SKIP
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Protocol, Type

import numpy as np
from numpy.typing import ArrayLike

from derivkit.derivatives.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.derivatives.finite.finite_difference import (
    FiniteDifferenceDerivative,
)
from derivkit.derivatives.fornberg import FornbergDerivative
from derivkit.derivatives.local_polynomial_derivative.local_polynomial_derivative import (
    LocalPolynomialDerivative,
)
from derivkit.derivatives.tabulated_model.one_d import Tabulated1DModel


class DerivativeEngine(Protocol):
    """Protocol each derivative engine must satisfy.

    This defines the minimal interface expected by DerivKit’s derivative
    backends. Any class registered as a derivative engine must be
    constructible with a target function ``function`` and an expansion
    point ``x0``, and must provide a ``.differentiate(...)`` method that
    performs the actual derivative computation. It serves only as a
    structural type check (similar to an abstract base class) and carries
    no runtime behavior. In other words, this is a template for derivative
    engine implementations.
    """
    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initialize the engine with a target function and expansion point."""
        ...
    def differentiate(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the derivative using the engine’s algorithm."""
        ...


# These are the built-in derivative methods available by default.
# For each method we allow up to five aliases:
#   - 3 obvious spelling / punctuation variants
#   - 2 common short-hands users are likely to type
# This keeps the interface flexible without bloating the lookup table
# or introducing ambiguous scheme-level names.
_METHOD_SPECS: list[tuple[str, Type[DerivativeEngine], list[str]]] = [
    ("adaptive", AdaptiveFitDerivative, ["adaptive-fit", "adaptive_fit", "ad", "adapt"]),
    ("finite", FiniteDifferenceDerivative, ["finite-difference", "finite_difference", "fd", "findiff", "finite_diff"]),
    ("local_polynomial", LocalPolynomialDerivative, ["local-polynomial", "local_polynomial", "lp", "localpoly", "local_poly"]),
    ("fornberg", FornbergDerivative, ["fb", "forn", "fornberg-fd", "fornberg_fd", "fornberg_weights"]),
]


def _norm(s: str) -> str:
    """Normalize a method string for robust matching (case/spacing/punct insensitive).

    Args:
        s: Input string.

    Returns:
        Normalized string.
    """
    return re.sub(r"[^a-z0-9]+", "", s.lower())


@lru_cache(maxsize=1)
def _method_maps() -> tuple[Mapping[str, Type[DerivativeEngine]], tuple[str, ...]]:
    """Construct and cache lookup tables for derivative methods.

    This function builds the internal mappings that link user-provided method
    names (and their aliases) to the corresponding derivative engine classes.
    It also records the canonical method names used for display in help and
    error messages. The result is cached after the first call for efficiency.
    Caching means that any changes to the registered methods (via
    ``register_method``) will not be reflected until the cache is cleared.

    Returns:
        A tuple containing:
            A pair ``(method_map, canonical_names)`` where:
                - ``method_map`` maps normalized names and aliases to engine classes.
                - ``canonical_names`` lists the sorted canonical method names.
    """
    method_map: dict[str, Type[DerivativeEngine]] = {}
    canonical: set[str] = set()
    for name, cls, aliases in _METHOD_SPECS:
        k = _norm(name)
        method_map[k] = cls
        canonical.add(k)
        for a in aliases:
            method_map[_norm(a)] = cls
    return method_map, tuple(sorted(canonical))


def register_method(
    name: str,
    cls: Type[DerivativeEngine],
    *,
    aliases: Iterable[str] = (),
) -> None:
    """Register a new derivative method.

    Adds a new derivative engine that can be referenced by name in
    :class:`derivkit.derivative_kit.DerivativeKit`. This function can be called
    from anywhere in the package (for example, inside a submodule’s
    ``__init__.py``) and is safe regardless of import order. The internal cache
    is automatically cleared and rebuilt on the next lookup.

    Args:
        name: Canonical public name of the method (e.g., ``"gp"``).
        cls: Engine class implementing the
            :class:`derivkit.derivative_kit.DerivativeEngine` protocol.
        aliases: Additional accepted spellings (e.g., ``"gaussian-process"``).

    Registering a new method:

        >>> from derivkit.derivative_kit import register_method  # doctest: +SKIP
        >>> from derivkit.some_new_method import NewMethodDerivative  # doctest: +SKIP
        >>> register_method(  # doctest: +SKIP
        ...     name="new-method",
        ...     cls=NewMethodDerivative,
        ...     aliases=("new_method", "nm"),
        ... )  # doctest: +SKIP
    """
    _METHOD_SPECS.append((name, cls, list(aliases)))
    _method_maps.cache_clear()


def _resolve(method: str) -> Type[DerivativeEngine]:
    """Resolve a user-provided method name or alias to an engine class.

    Args:
        method: User-provided method name or alias.

    Returns:
        Corresponding derivative engine class.
    """
    method_map, canon = _method_maps()
    try:
        return method_map[_norm(method)]
    except KeyError:
        opts = ", ".join(canon)
        raise ValueError(f"Unknown derivative method '{method}'. Choose one of {{{opts}}}.") from None


class DerivativeKit:
    """Unified interface for computing numerical derivatives.

    The class provides a simple way to evaluate derivatives using any of
    DerivKit’s available backends (e.g., adaptive fit or finite difference).
    By default, the adaptive-fit method is used.

    You can supply either a function and x0, or tabulated tab_x/tab_y and x0
    in case you want to differentiate a tabulated function.
    The chosen backend is invoked when you call the ``.differentiate()`` method.

    Example:
        >>> import numpy as np
        >>> from derivkit import DerivativeKit
        >>> dk = DerivativeKit(np.cos, x0=1.0)
        >>> deriv = dk.differentiate(order=1)  # uses the default "adaptive" method

    Attributes:
        function: The callable to differentiate.
        x0: The point or points at which the derivative is evaluated.
        default_method: The backend used when no method is specified.
    """

    def __init__(
            self,
            function: Callable[[float | np.ndarray], Any] | None = None,
            x0: float | np.ndarray | None = None,
            *,
            tab_x: ArrayLike | None = None,
            tab_y: ArrayLike | None = None,
    ) -> None:
        """Initializes the DerivativeKit with a target function and expansion point.

        Args:
            function: The function to be differentiated. Must accept a single float
                      and return a scalar or array-like output.
            x0: Point or array of points at which to evaluate the derivative.
            tab_x: Optional tabulated x values for creating a
                :class:`tabulated_model.one_d.Tabulated1DModel`.
            tab_y: Optional tabulated y values for creating a
                :class:`tabulated_model.one_d.Tabulated1DModel`.
        """
        # Enforce "either function or tabulated", not both.
        if function is not None and (tab_x is not None or tab_y is not None):
            raise ValueError("Pass either `function` or (`tab_x`, `tab_y`), not both.")

        if function is not None:
            self.function = function

        elif tab_x is not None or tab_y is not None:
            if tab_x is None or tab_y is None:
                raise ValueError("Both `tab_x` and `tab_y` must be provided for tabulated mode.")
            model = Tabulated1DModel(tab_x, tab_y)
            self.function = model

        else:
            raise ValueError("Need either `function` or (`tab_x`, `tab_y`).")

        if x0 is None:
            raise ValueError("`x0` must be provided.")
        self.x0 = x0
        self.default_method = "adaptive"

    def differentiate(
            self,
            *,
            method: str | None = None,
            **kwargs: Any,
    ) -> Any:
        """Compute derivatives using the chosen method.

        Forwards all keyword arguments to the engine’s ``.differentiate()``.

        Args:
            method: Method name or alias (e.g., ``"adaptive"``, ``"finite"``,
                ``"fd"``). Default is ``"adaptive"``.
            **kwargs: Passed through to the chosen engine.

        Returns:
            The derivative result from the underlying engine.

            If ``x0`` is a single value, returns the usual derivative output.

            If ``x0`` is an array of points, returns an array where the first
            dimension indexes the points in ``x0``. For example, if you pass
            5 points and each derivative has shape ``(2, 3)``, the result has
            shape ``(5, 2, 3)``.

        Notes:
            Thread-level parallelism across derivative evaluations can be
            controlled by passing ``n_workers`` via ``**kwargs``. Note that
            this does not launch separate Python processes. All work occurs
            within a single process using worker threads.

        Raises:
            ValueError: If ``method`` is not recognized.
        """
        chosen = method or self.default_method  # use default if None
        Engine = _resolve(chosen)

        x0_arr = np.asarray(self.x0)

        # scalar x0
        if x0_arr.ndim == 0:
            return Engine(self.function, float(x0_arr)).differentiate(**kwargs)

        # array of x0 values
        results = []
        for xi in x0_arr.ravel():
            res = Engine(self.function, float(xi)).differentiate(**kwargs)
            results.append(res)

        return np.stack(results, axis=0).reshape(
            x0_arr.shape + np.shape(results[0])
        )


def available_methods() -> dict[str, list[str]]:
    """Lists derivative methods exposed by this API, including aliases.

    Returns:
        Dict mapping canonical method name -> list of accepted aliases.
    """
    return {name: list(aliases) for name, _, aliases in _METHOD_SPECS}
