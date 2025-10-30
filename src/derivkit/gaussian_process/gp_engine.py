
from __future__ import annotations

from typing import Any, Callable

from derivkit.derivative_kit import register_method
from derivkit.gaussian_process.gaussian_process import GaussianProcess


class GPDerivative:
    """Adapter exposing a GaussianProcess through the DerivativeEngine API.

    Wraps a scalar function and an expansion point, then delegates derivative
    queries to a locally fitted Gaussian Process model.
    """
    def __init__(self, function: Callable[[float], Any], x0: float):
        """Initialize the adapter.

        Args:
          function: Scalar function f(x) to differentiate.
          x0: Expansion point at which derivatives are requested.
        """
        self.function = function
        self.x0 = float(x0)

    def differentiate(self, **kwargs: Any) -> Any:
        """Compute a derivative via a local Gaussian Process.

        GP constructor options (peeled from kwargs):
          kernel (str|object): Kernel name or object. Defaults to "rbf".
          kernel_params (dict|None): Kernel hyperparameters.
          noise_variance (float): Observation noise variance. Defaults to 1e-6.
          normalize (bool): Standardize targets internally. Defaults to True.
          optimize (bool): Enable basic hyperparameter tuning. Defaults to False.

        Query options (forwarded to ``GaussianProcess.differentiate``):
          order (int): 1 or 2. **Required.**
          samples (np.ndarray|None): Custom sample locations (n, d).
          n_points (int): Auto grid size when ``samples`` is None. Defaults to 13.
          spacing (float|str): Grid half-width or "auto". Defaults to "auto".
          base_abs (float): Baseline half-width for "auto". Defaults to 0.5.
          axis (int): Dimension to differentiate (for d>1). Defaults to 0.
          return_variance (bool): Return (mean, variance) if True. Defaults to False.
          local_frac_span (float): Local fitting window as fraction of span. Defaults to 0.35.

        Returns:
          Any: Either the derivative estimate (float) or a tuple (mean, variance)
          if ``return_variance=True``.

        Raises:
          NotImplementedError: If ``order`` is not 1 or 2.
          ValueError: If provided sampling arguments are invalid.
        """
        # peel off GaussianProcess ctor args (with defaults matching your GP)
        gp_ctor_keys = {
            "kernel", "kernel_params", "noise_variance", "normalize", "optimize"
        }
        gp_ctor = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in gp_ctor_keys}

        # Build the model and run
        gp = GaussianProcess(self.function, self.x0, **gp_ctor)
        return gp.differentiate(**kwargs)

# Register this engine under "gp" (with a couple aliases)
register_method(
    name="gp",
    cls=GPDerivative,
    aliases=("gaussian-process", "gaussproc", "gp"),
)
