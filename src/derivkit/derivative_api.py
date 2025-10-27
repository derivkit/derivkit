"""Provides the DerivativeAPI class.

The class unifies :class:`AdaptiveFitDerivative` and
:class:`FiniteDifferenceDerivative` under a single interface. The user must
specify the function to differentiate, the central value at which the
derivative should be evaluated, and the desired method. More details about
available options can be found in the documentation of the methods.

Typical usage example:

>>> derivative = DerivativeAPI(function, x0=1)
>>> result = derivative.differentiate(order=1, method="adaptive-fit")

``result`` is the derivative of ``function`` evaluated at value 1 using the
adaptive-fit method.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative
from derivkit.finite.finite_difference import FiniteDifferenceDerivative

ALIASES: dict[str, str] = {
    # adaptive
    "adaptive-fit": "adaptive",
    "adaptive": "adaptive",
    "poly": "adaptive",
    "af": "adaptive",
    # finite
    "finite-difference": "finite",
    "finite": "finite",
    "fd": "finite",
    # future examples:
    # "gp": "gp",
    # "gaussian-process": "gp",
}

MethodName = Literal[
    "adaptive-fit", "adaptive", "poly", "af",
    "finite-difference", "finite", "fd",
]


class DerivativeAPI:
    """Non-breaking unified derivative entry point.

    This facade exposes a single method, :meth:`differentiate`, that routes calls to
    one of several derivative engines (currently adaptive polynomial-fit and finite
    differences). It keeps the legacy classes intact while providing a streamlined,
    discoverable API:

        >>> dk = DerivativeAPI(function, x0=0.3)
        >>> dk.differentiate(order=1, method="adaptive-fit", n_points=25, spacing=0.25)
        >>> dk.differentiate(order=2, method="fd", stepsize=1e-2, num_points=5)

    Under the hood, engines are registered in a small backend registry and selected
    via normalized method aliases (e.g., ``"adaptive-fit" | "adaptive" | "af"`` and
    ``"finite-difference" | "finite" | "fd"``). Keyword arguments are *automatically*
    filtered against the chosen backend’s signature so you can safely pass a union
    of kwargs without manually keeping per-engine lists.

    Key properties:
      * **Non-breaking:** Lives alongside the existing ``DerivativeKit``; no changes
        required for legacy code.
      * **Alias-driven selection:** Human-friendly names mapped to canonical keys.
      * **Automatic kwarg filtering:** Typos and wrong-engine kwargs raise clear,
        method-specific errors.
      * **Extensible:** Adding a new engine is a one-liner in the registry plus
        aliases (no public API changes).

    Examples:
      Basic usage with both engines:

      >>> dk = DerivativeAPI(lambda x: x**3, x0=2.0)
      >>> g1 = dk.differentiate(order=1, method="adaptive-fit", n_points=11, spacing=0.2)
      >>> g2 = dk.differentiate(order=2, method="fd", stepsize=1e-3, num_points=5)

    Notes:
      * Return type mirrors the backend: a scalar for scalar-valued functions or a
        1D array for vector-valued outputs.
      * See ``supported_methods()`` or ``help()`` for discoverability and signatures.
      * We will change this dcostring once everyone is on board with the new API.
    """

    def __init__(self, function, x0: float):
        """Initialize the unified API with a target function and expansion point.

        Args:
          function: Callable of one float returning either a float (scalar observable)
            or a 1D array-like (multi-observable). This is the function whose
            derivatives will be estimated.
          x0: Central value at which derivatives are evaluated.

        Implementation details:
          A small backend registry is created, binding canonical keys to the
          corresponding backend ``.differentiate`` callables. To add a new engine
          (e.g., ``"gp"``), instantiate it here and register its ``differentiate``:
        """
        # Backend registry (add new engines here once, e.g., "gp")
        self._backends: dict[str, Any] = {
            "adaptive": AdaptiveFitDerivative(function, x0).differentiate,
            "finite":   FiniteDifferenceDerivative(function, x0).differentiate,
            # "gp": GPDerivative(function, x0).differentiate,  # future
        }

    def differentiate(self, order: int, *, method: str | MethodName, **kwargs) -> Any:
        """Compute the derivative of the specified order using the chosen backend.

        This method normalizes the provided ``method`` (handling aliases and case),
        looks up the corresponding backend, and forwards only those keyword
        arguments that are accepted by that backend’s signature. Unknown or
        misspelled kwargs (for backends that don’t accept ``**kwargs``) raise a
        precise, readable error that lists the allowed keywords.

        Args:
          order: Derivative order to compute (>= 1). The selected backend must
            support the requested order.
          method: User-friendly method name or alias. Supported aliases include:
            - Adaptive polynomial fit: ``"adaptive-fit"``, ``"adaptive"``, ``"poly"``, ``"af"``
            - Finite differences: ``"finite-difference"``, ``"finite"``, ``"fd"``
            Use :meth:`supported_methods` for a programmatic list or :meth:`help`
            for readable guidance.
          **kwargs: Backend-specific options forwarded after automatic filtering.
            Examples include:
              * Adaptive: ``n_points``, ``spacing``, ``base_abs``, ``n_workers``,
                ``grid``, ``domain``, ``ridge``, ``diagnostics``, ``meta``.
              * Finite: ``stepsize``, ``num_points``, ``n_workers``.

        Returns:
          A float for scalar-valued functions or a 1D NumPy array for vector-valued
          functions, representing the derivative of the specified order at ``x0``.

        Raises:
          ValueError: If the method alias is unknown, the backend is not
            initialized, the order/stencil combination is unsupported by the
            chosen backend, or if unknown kwargs are supplied to a backend that
            doesn’t accept ``**kwargs``.

        Examples:
          >>> dk = DerivativeAPI(lambda x: (x**3, x**2), x0=0.2)
          >>> dk.differentiate(order=1, method="adaptive", n_points=9, spacing=0.2)
          array([...])  # d/dx [x^3, x^2] at 0.2
          >>> dk.differentiate(order=2, method="fd", stepsize=1e-3, num_points=5)
          array([...])

        Notes:
          * If a backend defines ``**kwargs`` in its signature, all extra keyword
            arguments are passed through unchanged.
          * Backend-specific calling quirks (e.g., positional vs keyword ``order``)
            are handled internally to keep the public API uniform.
        """
        key = _resolve_key(method)
        if key not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(f"Backend '{key}' not initialized. Available: {available}.")

        f = self._backends[key]

        # Tiny match reserved for backend-specific call quirks; default covers future engines.
        match key:
            case "adaptive":
                return f(order, **_filter_kwargs_for(f, kwargs))
            case "finite":
                # Also accepts positional order; we pass as kw for symmetry
                return f(order=order, **_filter_kwargs_for(f, kwargs))
            case _:
                return f(order, **_filter_kwargs_for(f, kwargs))

    @staticmethod
    def supported_methods() -> dict[str, list[str]]:
        """Return a mapping of canonical method labels to their aliases.

        Builds a dictionary where each key is a canonical, user-facing label
        (e.g., ``"adaptive-fit"`` or ``"finite-difference"``) and the value is
        a list of all supported aliases that resolve to that backend. The
        canonical label is placed first in each list for stable, predictable
        presentation.

        This function derives its result from the module-level alias registry,
        so it automatically reflects any newly added backends or aliases.

        Returns:
          dict[str, list[str]]: Mapping from canonical display label to a list
          of aliases. The first element of each list is the canonical label
          itself, followed by the remaining aliases in deterministic order.

        Examples:
          >>> DerivativeAPI.supported_methods()
          {
            'adaptive-fit': ['adaptive-fit', 'adaptive', 'af', 'poly'],
            'finite-difference': ['finite-difference', 'finite', 'fd']
          }

        Notes:
          * The canonical labels are those used in help messages and docs.
          * To add a new backend (e.g., Gaussian-process derivatives), extend the
            alias registry; this function will pick it up automatically.
        """
        return _grouped_supported_methods()


    def help(self, method: str | MethodName | None = None) -> str:
        """Return a human-readable guide or backend-specific signature/docstring.

        When called without arguments, this function returns a short overview of all
        supported derivative methods grouped by their canonical display labels
        (e.g., ``"adaptive-fit"`` and ``"finite-difference"``), including their
        aliases and a minimal usage example.

        When a method name (or alias) is provided, the function normalizes it to the
        corresponding backend, retrieves that backend’s ``.differentiate`` callable,
        and returns a string containing its Python signature followed by its
        docstring. This is useful for quickly discovering the exact keyword options
        accepted by a particular backend.

        Args:
          method: Optional method name or alias (case-insensitive; underscores and
            hyphens are treated equivalently). If ``None``, a general guide with the
            list of methods and aliases is returned.

        Returns:
          str: A formatted string. If ``method`` is ``None``, a multi-line guide of
          supported methods and aliases with a usage example. If ``method`` is
          provided, the backend’s qualified name and signature are shown, followed
          by its docstring text.

        Examples:
          >>> dk = DerivativeAPI(lambda x: x**2, x0=0.0)
          >>> print(dk.help())  # overview with aliases and example
          Methods:
            - adaptive-fit (aliases: adaptive, af, poly)
            - finite-difference (aliases: finite, fd)
          Example: dk.differentiate(order=1, method='adaptive-fit', n_points=25)

          >>> print(dk.help("fd"))  # finite-difference signature + docstring
          FiniteDifferenceDerivative.differentiate(order: int = 1, stepsize: float = 0.01, ...)

        Notes:
          * Method normalization is handled by :func:`_resolve_key`, which consults the
            global alias registry. If a new backend or alias is added, this helper
            reflects it automatically.
          * If the resolved backend is not initialized in this instance (e.g., a
            future engine is aliased but not registered), a short diagnostic message
            listing available backends is returned instead of raising.
        """
        if method is None:
            lines = ["Methods:"]
            for label, aliases in _grouped_supported_methods().items():
                alias_str = ", ".join(a for a in aliases if a != label)
                lines.append(f"  - {label}" + (f" (aliases: {alias_str})" if alias_str else ""))
            lines.append("Example: dk.differentiate(order=1, method='adaptive-fit', n_points=25)")
            return "\n".join(lines)

        key = _resolve_key(method)
        if key not in self._backends:
            available = ", ".join(sorted(self._backends))
            return f"Backend '{key}' not initialized. Available: {available}."

        f = self._backends[key]
        return f"{f.__qualname__}{inspect.signature(f)}\n\n{(f.__doc__ or '').strip()}"


def _resolve_key(method: str | None) -> str:
    """Return the canonical backend key for a given method alias.

    This helper normalizes a user-provided method name (e.g. "adaptive-fit",
    "adaptive", "fd") and resolves it to a canonical backend key (currently
    ``"adaptive"`` or ``"finite"``). Normalization is case-insensitive,
    strips surrounding whitespace, and converts underscores to hyphens so that
    values like ``"Adaptive_Fit"`` and ``"adaptive-fit"`` are treated
    equivalently.

    The mapping from aliases to canonical keys is defined by the module-level
    :data:`ALIASES` dictionary and serves as the single source of truth for
    supported names. If the provided alias is not recognized, a clear
    ``ValueError`` is raised listing all supported aliases.

    Args:
      method: The user-specified method name or alias. May be ``None``,
        in which case the function will raise with the supported options.

    Returns:
      str: The canonical backend key corresponding to the alias, e.g.
      ``"adaptive"`` for any of {"adaptive-fit", "adaptive", "poly", "af"},
      or ``"finite"`` for any of {"finite-difference", "finite", "fd"}.

    Raises:
      ValueError: If ``method`` is not a known alias. The error message
        includes the list of supported aliases.

    Examples:
      >>> _resolve_key("adaptive-fit")
      'adaptive'
      >>> _resolve_key("AF")            # case-insensitive
      'adaptive'
      >>> _resolve_key("finite_difference")  # underscores normalized to hyphens
      'finite'
      >>> _resolve_key("unknown")
      Traceback (most recent call last):
          ...
      ValueError: Unknown method 'unknown'. Supported: adaptive, adaptive-fit, af, fd, ...

    Notes:
      * Normalization steps: ``strip()`` → ``lower()`` → replace ``"_"`` with ``"-"``.
      * To add a new backend (e.g., Gaussian process derivatives), extend
        :data:`ALIASES` with the new aliases mapping to a new canonical key
        (e.g., ``"gp"``). Downstream registries can then reference that key automatically.
    """
    m = (method or "").strip().lower().replace("_", "-")
    try:
        return ALIASES[m]
    except KeyError as e:
        supported = ", ".join(sorted(ALIASES.keys()))
        raise ValueError(f"Unknown method '{method}'. Supported: {supported}.") from e


def _label_for(canon: str) -> str:
    """Return the canonical display label for a backend key.

    This function maps internal canonical backend identifiers
    (e.g., ``"adaptive"`` or ``"finite"``) to their preferred
    user-facing display names. These display labels are used in
    help messages, documentation, and logging to provide consistent
    and human-readable naming across the codebase.

    If a backend key is not explicitly defined in the mapping,
    the function simply returns the input value unchanged. This
    behavior allows new backends (e.g., ``"gp"``) to be introduced
    without requiring updates to this helper.

    Args:
      canon: The canonical backend key (for example, ``"adaptive"`` or
        ``"finite"``) used internally in the registry.

    Returns:
      str: The preferred human-readable label for display purposes.
      For example, ``"adaptive-fit"`` for ``"adaptive"`` or
      ``"finite-difference"`` for ``"finite"``. If the key is not
      found, the original value of ``canon`` is returned.

    Examples:
      >>> _label_for("adaptive")
      'adaptive-fit'
      >>> _label_for("finite")
      'finite-difference'
      >>> _label_for("gp")
      'gp'  # unchanged, since not in the mapping

    Notes:
      * The returned labels correspond to the canonical names used
        in documentation and examples.
      * This helper intentionally avoids raising exceptions to keep
        the registry robust to new or experimental backends.
    """
    return {"adaptive": "adaptive-fit", "finite": "finite-difference"}.get(canon, canon)


def _grouped_supported_methods() -> dict[str, list[str]]:
    """Group all method aliases by their canonical display label.

    Builds a mapping from a canonical, user-facing label (e.g.,
    ``"adaptive-fit"`` or ``"finite-difference"``) to the list of all
    supported aliases that resolve to that backend. The canonical label is
    placed first in each alias list for stable, predictable presentation.

    The grouping is derived from the global :data:`ALIASES` dictionary and the
    label mapping provided by :func:`_label_for`. This function is primarily
    used to render help text, documentation snippets, and CLI hints where a
    concise display of available methods and their aliases is desirable.

    Returns:
      dict[str, list[str]]: A dictionary where each key is a canonical display
      label and the value is a list of alias strings. The first element of the
      list is the canonical label itself, followed by other aliases in
      lexicographic order.

    Examples:
      >>> # Given ALIASES with entries mapping to "adaptive" and "finite"
      >>> groups = _grouped_supported_methods()
      >>> groups["adaptive-fit"][0]
      'adaptive-fit'
      >>> "adaptive" in groups["adaptive-fit"]
      True
      >>> "fd" in groups["finite-difference"]
      True

    Notes:
      * The canonical label is determined by :func:`_label_for`, which returns
        human-readable names for internal backend keys.
      * Sorting places the canonical label first and orders remaining aliases
        alphabetically to keep output deterministic.
      * Adding a new backend only requires updating :data:`ALIASES` (and
        optionally :func:`_label_for`); this function adapts automatically.
    """
    groups: dict[str, list[str]] = {}
    for alias, canon in ALIASES.items():
        label = _label_for(canon)
        groups.setdefault(label, []).append(alias)
    for label, aliases in groups.items():
        aliases.sort(key=lambda a: (a != label))  # canonical first
    return groups


def _filter_kwargs_for(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only the keyword arguments accepted by a callable.

    This helper inspects the signature of ``func`` and filters ``kwargs`` so that
    only parameters actually accepted by ``func`` are forwarded. If ``func`` defines
    a ``**kwargs`` parameter, all items are passed through unchanged. Otherwise,
    any unknown keywords trigger a clear ``ValueError`` listing the allowed names.

    This provides a safe “fan-out” mechanism when a higher-level API forwards
    user-provided kwargs to multiple backends with different signatures.

    Args:
      func: The callable whose signature will be inspected. Typically a
        ``.differentiate`` method of a backend.
      kwargs: The candidate keyword arguments intended for ``func``.

    Returns:
      dict[str, Any]: A new dictionary containing only the items from ``kwargs``
      that match keyword-capable parameters of ``func`` (i.e., parameters of kind
      ``POSITIONAL_OR_KEYWORD`` or ``KEYWORD_ONLY``). If ``func`` accepts
      ``**kwargs``, this is simply a shallow copy of the input ``kwargs``.

    Raises:
      ValueError: If ``func`` does **not** accept ``**kwargs`` and one or more
        keys in ``kwargs`` are not present in ``func``’s signature. The error
        message lists the unexpected keys and the allowed keywords.

    Examples:
      >>> def f(x, *, n=1, ridge=0.0): ...
      >>> _filter_kwargs_for(f, {"n": 25, "ridge": 1e-8, "spacing": 0.2})
      Traceback (most recent call last):
          ...
      ValueError: Unexpected keyword(s) for f: spacing. Allowed keywords: n, ridge.

      >>> def g(x, **kwargs): ...
      >>> _filter_kwargs_for(g, {"foo": 1, "bar": 2})
      {'foo': 1, 'bar': 2}

    Notes:
      * Only parameters that can be provided as keywords are considered
        (``POSITIONAL_OR_KEYWORD`` and ``KEYWORD_ONLY``).
      * This function does **not** validate parameter types or values—it only
        checks names against the callable’s signature.
      * The returned dictionary is safe to forward directly, e.g.,
        ``func(**_filter_kwargs_for(func, kwargs))``.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)

    allowed = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(kwargs) - allowed
    if unknown:
        allowed_sorted = ", ".join(sorted(allowed)) or "none"
        bad_sorted = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unexpected keyword(s) for {func.__qualname__}: {bad_sorted}. "
            f"Allowed keywords: {allowed_sorted}."
        )
    return {k: v for k, v in kwargs.items() if k in allowed}
