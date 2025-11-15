"""Provides the DerivativeKit API.

This class is a lightweight front end over DerivKit’s derivative engines.
You provide the function to differentiate and the expansion point `x0`,
then choose a backend by name (e.g., ``"adaptive"`` or ``"finite"``).

Adding methods
--------------
New engines can be registered without modifying this class by calling
``register_method`` (see example below).

Examples:
    Basic usage:

        >>> import numpy as np
        >>> from derivkit.derivative_kit import DerivativeKit
        >>> dk = DerivativeKit(function=np.cos, x0=1.0)
        >>> # First derivative via the adaptive-fit method:
        >>> # dk.differentiate(method="adaptive", order=1)  # doctest: +SKIP

    Registering a new method:

        >>> from derivkit.derivative_kit import register_method
        >>> from derivkit.some_new_method import NewMethodDerivative
        >>> register_method(
        ...     name="new-method",
        ...     cls=NewMethodDerivative,
        ...     aliases=("new_method", "nm"),
        ... )

Notes:
    - Method names are case/spacing/punctuation insensitive; aliases like
      ``"adaptive-fit"`` or ``"finite_difference"`` are supported when
      registered.
    - For available canonical method names at runtime, call
      ``available_methods()``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Protocol, Type

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.finite.finite_difference import FiniteDifferenceDerivative
from derivkit.local_polynomial_derivative.local_polynomial_derivative import (
    LocalPolynomialDerivative,
)


class DerivativeEngine(Protocol):
    """Protocol each derivative engine must satisfy.

    This defines the minimal interface expected by DerivKit’s derivative
    backends. Any class registered as a derivative engine must be
    constructible with a target function `function` and an expansion point `x0`,
    and must provide a `.differentiate(...)` method that performs the actual
    derivative computation. It serves only as a structural type check
    (similar to an abstract base class) and carries no runtime behavior.
    In other words, this is a template for derivative engine implementations.
    """
    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initialize the engine with a target function and expansion point."""
        ...
    def differentiate(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the derivative using the engine’s algorithm."""
        ...


# These are the built-in methods available in the package by default.
_METHOD_SPECS: list[tuple[str, Type[DerivativeEngine], list[str]]] = [
    ("adaptive", AdaptiveFitDerivative, ["adaptive-fit", "adaptive_fit", "ad"]),
    ("finite",   FiniteDifferenceDerivative, ["finite-difference", "finite_difference", "fd"]),
    ("local_polynomial", LocalPolynomialDerivative, ["local-polynomial", "local_polynomial", "lp"]),
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
    `register_method`) will not be reflected until the cache is cleared.

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
    :class:`DerivativeKit`. This function can be called from anywhere in the
    package (for example, inside a submodule’s ``__init__.py``) and is safe
    regardless of import order. The internal cache is automatically cleared
    and rebuilt on the next lookup.

    Args:
        name: Canonical public name of the method (e.g., "gp").
        cls: Engine class implementing the DerivativeEngine protocol.
        aliases: Additional accepted spellings (e.g., "gaussian-process").

    Example:
        >>> from derivkit.derivative_api import register_method
        >>> from derivkit.gp.gp_derivative import GPDerivative
        >>> register_method(
        ...     name="gp",
        ...     cls=GPDerivative,
        ...     aliases=("gaussian-process", "gaussproc"),
        ... )
        >>> # After registration, it can be used via:
        >>> # DerivativeKit(f, x0).differentiate(method="gp")  # doctest: +SKIP
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
    You only need to supply a function and the point ``x0`` at which to
    compute the derivative.  By default, the adaptive-fit method is used.

    Example:
        >>> import numpy as np
        >>> from derivkit.derivative_kit import DerivativeKit
        >>> d = DerivativeKit(np.cos, x0=1.0)
        >>> d.differentiate(order=1)  # uses the default "adaptive" method

    Attributes:
        function: The callable to differentiate.
        x0: The point at which the derivative is evaluated.
        DEFAULT_METHOD: The default backend used when no method is specified.
    """

    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initializes the DerivativeKit with a target function and expansion point.

        Args:
            function: The function to be differentiated. Must accept a single float
                      and return a scalar or array-like output.
            x0: Point at which to evaluate the derivative.
        """
        self.function = function
        self.x0 = x0
        self.default_method = "adaptive"

    def differentiate(self,
                      *,
                      method: str = None,
                      **kwargs: Any) -> Any:
        """Compute derivatives using the chosen method.

        Forwards all keyword arguments to the engine’s `.differentiate()`.

        Args:
            method: Method name or alias (e.g., "adaptive", "finite", "fd"). Default is "adaptive".
            **kwargs: Passed through to the chosen engine.

        Returns:
            The derivative result from the underlying engine.

        Raises:
            ValueError: If `method` is not recognized.
        """
        chosen = method or self.default_method  # use default if None
        Engine = _resolve(chosen)
        return Engine(self.function, self.x0).differentiate(**kwargs)


def available_methods() -> list[str]:
    """List canonical method names exposed by this API.

    Returns:
        List of method names.
    """
    _, canon = _method_maps()
    return list(canon)
