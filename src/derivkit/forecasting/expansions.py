"""Provides tools for facilitating experimental forecasts.

The user must specify the observables, fiducial values and covariance matrix
at which the derivative should be evaluated. Derivatives of the first order
are Fisher derivatives. Derivatives of second order are evaluated using the
derivative approximation for likelihoods (DALI) technique as described in
https://doi.org/10.1103/PhysRevD.107.103506.

More details about available options can be found in the documentation of
the methods.
"""

import warnings
from copy import deepcopy

import numpy as np

from derivkit.derivative_kit import DerivativeKit
from derivkit.forecasting.calculus import jacobian
from derivkit.utils import get_partial_function, solve_or_pinv


class LikelihoodExpansion:
    """Provides tools for facilitating experimental forecasts.

    Attributes:
         function (callable): The scalar or vector-valued function to
             differentiate. It should accept a list or array of parameter
             values as input and return either a scalar or a
             :class:`np.ndarray` of observable values.
         theta0 (class:`np.ndarray`): The point(s) at which the
             derivative is evaluated. A 1D array or list of parameter values
             matching the expected input of the function.
         cov (class:`np.ndarray`): The covariance matrix of
             the observables. Should be a square matrix with shape
             (n_observables, n_observables), where n_observables is the
             number of observables returned by the function.
         n_parameters (int): The number of elements of `theta0`.
         n_observables (int): The number of cosmic observables. Determined
             from the dimension of `cov`.
    """

    def __init__(self, function, theta0, cov):
        """Initialises the class.

        Args:
            function (callable): The scalar or vector-valued function to
                differentiate. It should accept a list or array of parameter
                values as input and return either a scalar or a
                :class:`np.ndarray` of observable values.
            theta0 (class:`np.ndarray`): The points at which the
                derivative is evaluated. A 1D array or list of parameter values
                matching the expected input of the function.
            cov (class:`np.ndarray`): The covariance matrix of
                the observables. Should be a square matrix with shape
                (n_observables, n_observables), where n_observables is the
                number of observables returned by the function.

        Raises:
            ValueError: raised if cov is not a square numpy array.
        """
        self.function = function
        self.theta0 = np.atleast_1d(theta0)

        cov = np.asarray(cov)
        if cov.ndim > 2:
            raise ValueError(
                f"cov must be at most two-dimensional; got ndim={cov.ndim}."
            )
        if cov.ndim == 2 and cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square; got shape={cov.shape}.")

        self.cov = cov
        self.n_parameters = self.theta0.shape[0]
        self.n_observables = self.cov.shape[0]

    def get_forecast_tensors(self, forecast_order=1, n_workers=1):
        """Returns a set of tensors according to the requested order of the forecast.

        Args:
            forecast_order (int): The requested order D of the forecast:

                    - D = 1 returns a Fisher matrix.
                    - D = 2 returns the 3-d and 4-d tensors required for the
                      doublet-DALI approximation.
                    - D = 3 would be the triplet-DALI approximation.

                Currently only D = 1, 2 are supported.
            n_workers (int, optional): Number of worker to use in multiprocessing.
                Default is 1 (no multiprocessing).

        Returns:
            :class:`np.ndarray`: A list of numpy arrays:

                    - D = 1 returns a square matrix of size n_parameters, where
                      n_parameters is the number of parameters included in the
                      forecast.
                    - D = 2 returns one array of shapes
                      (n_parameters, n_parameters, n_parameters) and one array
                      of shape (n_parameters, n_parameters, n_parameters, n_parameters),
                      where n_parameters is the number of parameters included
                      in the forecast.

        Raises:
            ValueError: If `forecast_order` is not 1 or 2.

        Warns:
            RuntimeWarning: If `cov` is not symmetric (it is symmetrized),
                is ill-conditioned (large condition number), or inversion
                falls back to the pseudoinverse.
        """
        if forecast_order not in [1, 2]:
            raise ValueError(
                "Only Fisher (order 1) and doublet-DALI (order 2) forecasts are currently supported."
            )

        # Check model output dimension
        y0 = np.atleast_1d(self.function(self.theta0))
        if y0.shape[0] != self.n_observables:
            raise ValueError(
                f"Expected {self.n_observables} observables from model (from cov {self.cov.shape}), "
                f"but got {y0.shape[0]} (output shape {y0.shape})."
            )

        # Compute inverse covariance matrix
        invcov = self._inv_cov()
        # Compute first-order derivatives
        d1 = self._get_derivatives(order=1, n_workers=n_workers)

        if forecast_order == 1:
            return self._build_fisher(d1, invcov)  # Fisher

        # Compute second-order derivatives
        d2 = self._get_derivatives(order=2, n_workers=n_workers)
        return self._build_dali(d1, d2, invcov)  # doublet-DALI (G, H)

    def _get_derivatives(self, order, n_workers=1):
        """Returns derivatives of the observables of the requested order.

        Args:
            order (int): The requested order d of the derivatives:

                - d = 1 returns first-order derivatives.
                - d = 2 returns second-order derivatives.

                Currently only d = 1, 2 are supported.

            n_workers (int, optional): Number of worker to use in
                multiprocessing. Default is 1 (no multiprocessing).

        Returns:
            :class:`np.ndarray`: An array of derivative values:

                - d = 1 returns an array with shape
                  (`n_parameters`, `n_observables`) containing
                  first-order derivatives.
                - d = 2 returns an array with shape
                  `n_parameters`, `n_parameters`, `n_observables`)
                  containing second-order derivatives.

        Raises:
            ValueError: An error occurred if a derivative was requested of
                higher order than 2.
            RuntimeError: An error occurred if a ValueError was not raised
                after calling the function.
        """
        if order not in [1, 2]:
            raise ValueError(
                "Only first- and second-order derivatives are currently supported."
            )

        if order == 1:
            # Get the first-order derivatives
            first_order_derivatives = np.zeros(
                (self.n_parameters, self.n_observables), dtype=float
            )
            for m in range(self.n_parameters):
                # 1 parameter to differentiate, and n_parameters-1 parameters to hold fixed
                theta0_x = deepcopy(self.theta0)
                function_to_diff = get_partial_function(
                    self.function, m, theta0_x
                )
                kit = DerivativeKit(function_to_diff, self.theta0[m])
                first_order_derivatives[m] = kit.adaptive.differentiate(
                    order=1, n_workers=n_workers
                )
            return first_order_derivatives

        elif order == 2:
            # Get the second-order derivatives
            second_order_derivatives = np.zeros(
                (self.n_parameters, self.n_parameters, self.n_observables),
                dtype=float,
            )

            for m1 in range(self.n_parameters):
                for m2 in range(self.n_parameters):
                    if m1 == m2:
                        # 1 parameter to differentiate twice, and n_parameters-1 parameters to hold fixed
                        theta0_x = deepcopy(self.theta0)
                        function_to_diff1 = get_partial_function(
                            self.function, m1, theta0_x
                        )
                        kit1 = DerivativeKit(
                            function_to_diff1, self.theta0[m1]
                        )
                        second_order_derivatives[m1][m2] = (
                            kit1.adaptive.differentiate(
                                order=2, n_workers=n_workers
                            )
                        )

                    else:
                        # 2 parameters to differentiate once, with other parameters held fixed
                        def function_to_diff2(y):
                            theta0_y = deepcopy(self.theta0)
                            theta0_y[m2] = y
                            function_to_diff1 = get_partial_function(
                                self.function, m1, theta0_y
                            )
                            kit1 = DerivativeKit(
                                function_to_diff1, self.theta0[m1]
                            )
                            return kit1.adaptive.differentiate(order=1)

                        kit2 = DerivativeKit(
                            function_to_diff2, self.theta0[m2]
                        )
                        second_order_derivatives[m1][m2] = (
                            kit2.adaptive.differentiate(
                                order=1, n_workers=n_workers
                            )
                        )

            return second_order_derivatives

        raise RuntimeError("Unreachable code reached in get_forecast_tensors.")

    def _inv_cov(self):
        """Return the inverse covariance matrix with minimal diagnostics.

        Warns:
            RuntimeWarning: If cov is non-symmetric (checked via allclose),
                ill-conditioned (cond > 1e12), or if inversion falls back to pinv.

        Returns:
            np.ndarray: Inverse covariance matrix, shape (n_observables, n_observables).
        """
        cov = self.cov

        # warn only; do not symmetrize, to match historical fixture values
        classname = self.__class__.__name__
        if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
            warnings.warn(
                f"[{classname}] `cov` is not symmetric; proceeding as-is (no symmetrization).",
                RuntimeWarning,
            )

        # condition number warning (helps debug instability)
        try:
            cond = np.linalg.cond(cov)
            if cond > 1e12:
                warnings.warn(
                    f"[{classname}] `cov` is ill-conditioned (cond≈{cond:.2e}); "
                    "results may be unstable.",
                    RuntimeWarning,
                )
        except Exception:
            pass  # if cond() fails, just skip

        # invert with pinv fallback
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"[{classname}] `cov` inversion failed; using pseudoinverse.",
                RuntimeWarning,
            )
            return np.linalg.pinv(cov, rcond=1e-12, hermitian=False)

    def _build_fisher(self, d1, invcov):
        """Assemble the Fisher information matrix F from first derivatives.

        Args:
            d1 (np.ndarray): First-order derivatives of observables w.r.t. parameters,
                shape (n_parameters, n_observables).
            invcov (np.ndarray): Inverse covariance of observables,
                shape (n_observables, n_observables).

        Returns:
            np.ndarray: Fisher matrix, shape (n_parameters, n_parameters).

        Notes:
            Uses `np.einsum("ai,ij,bj->ab", d1, invcov, d1)`.
        """
        # F_ab = Σ_ij d1[a,i] invcov[i,j] d1[b,j]
        return np.einsum("ai,ij,bj->ab", d1, invcov, d1)

    def _build_dali(self, d1, d2, invcov):
        """Assemble the doublet-DALI tensors (G, H) from first/second derivatives.

        Computes:
            G_abc = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d1[c,j]
            H_abcd = Σ_{i,j} d2[a,b,i] · invcov[i,j] · d2[c,d,j]

        Args:
            d1 (np.ndarray): First-order derivatives d(obs)/dθ, shape (P, N).
            d2 (np.ndarray): Second-order derivatives d²(obs)/dθ², shape (P, P, N).
            invcov (np.ndarray): Inverse covariance of observables, shape (N, N).

        Returns:
            tuple[np.ndarray, np.ndarray]: G with shape (P, P, P) and H with shape (P, P, P, P).
        """
        # G_abc = Σ_ij d2[a,b,i] invcov[i,j] d1[c,j]
        g_tensor = np.einsum("abi,ij,cj->abc", d2, invcov, d1)
        # H_abcd = Σ_ij d2[a,b,i] invcov[i,j] d2[c,d,j]
        h_tensor = np.einsum("abi,ij,cdj->abcd", d2, invcov, d2)
        return g_tensor, h_tensor

    def fisher_bias(self, *, delta=None, datavec_with=None, datavec_without=None, n_workers=1):
        """Compute the Fisher bias for parameters.

        Args:
            delta: Data-vector mismatch Δμ. Can be 1D of length N or a 2D array
                (corr, ell) that will be flattened C-order. If provided, takes
                precedence over `data_with`/`data_without`.
            datavec_with: Data vector with the systematic present. 1D (N,) or 2D
                (corr, ell). Used only if `delta` is None.
            datavec_without: Data vector without the systematic. Same shape rules
                as `data_with`. Used only if `delta` is None.
            n_workers: Number of workers passed to derivative routines.

        Returns:
            dict: A dictionary with:
                - "F": Fisher matrix, shape (P, P).
                - "b": Bias “force” vector Jᵀ C⁻¹ Δμ, shape (P,).
                - "delta_theta": Parameter bias estimate Δθ, shape (P,).

        Raises:
            ValueError: If shapes are inconsistent with (N, P), (P, N), or (N, N),
                or if neither `delta` nor the pair (`data_with`, `data_without`)
                is provided.
        """
        n_obs, n_params = self.n_observables, self.n_parameters

        inv_cov = self._inv_cov()  # (N, N) shape
        if inv_cov.shape != (n_obs, n_obs):
            raise ValueError(f"inv_cov has shape {inv_cov.shape}, expected {(n_obs, n_obs)}")

        # deriv matrix d1 (P, N) shape
        # try jacobian first for speed; fallback to _get_derivatives for robustness
        try:
            jac = jacobian(self.function, self.theta0, n_workers=n_workers)  # (N, P) shape
            if jac.shape != (n_obs, n_params):
                raise ValueError(f"Jacobian has shape {jac.shape}, expected {(n_obs, n_params)}.")
            d1 = jac.T  # (P, N) shape here
        except Exception:
            d1 = self._get_derivatives(order=1, n_workers=n_workers)  # (P, N) shape niko sanity check

        if d1.shape != (n_params, n_obs):
            raise ValueError(f"d1 has shape {d1.shape}, expected {(n_params, n_obs)}.")
        fisher = self._build_fisher(d1, inv_cov)  # (P, P)

        delta_mu = self._build_delta_vector(
            delta=delta, data_with=datavec_with, data_without=datavec_without, n_obs=n_obs
        )

        # bias vector
        bias_vec = d1 @ (inv_cov @ delta_mu)

        delta_theta = solve_or_pinv(jac, bias_vec)

        return {"F": fisher, "b": bias_vec, "delta_theta": delta_theta}

    def _flatten_data_vector(self, data: np.ndarray, n_obs: int) -> np.ndarray:
        """Return a 1D data vector of length N from 1D/2D input.

        Args:
            data: Input data. Either 1D of length N or 2D (corr, ell), which will be
                flattened in C-order.
            n_obs: Expected length N of the flattened data vector.

        Returns:
            np.ndarray: Flattened 1D array of length N.

        Raises:
            ValueError: If `data` (a ) is not 1D or 2D, or if the flattened size does not
                equal `n_obs`.
        """
        data_array = np.asarray(data, dtype=float)
        if data_array == 1:
            if data_array.size != n_obs:
                raise ValueError(f"vector length {data_array.size} != covariance dim {n_obs}")
            return data_array
        if data_array.ndim == 2:
            data_flattened = data_array.ravel(order="C")
            if data_flattened.size != n_obs:
                raise ValueError("flattened 2D data length != covariance dimension")
            return data_flattened
        raise ValueError("data must be 1D or 2D")

    def _build_delta_vector(
            self,
            *,
            delta: np.ndarray | None,
            data_with: np.ndarray | None,
            data_without: np.ndarray | None,
            n_obs: int,
    ) -> np.ndarray:
        """Construct the data mismatch vector delta.

        Args:
            delta: Precomputed mismatch vector. If provided, it is validated and used
                directly (1D length N or 2D to be flattened).
            data_with: Data vector with the systematic present. Used only if
                `delta` is None. 1D length N or 2D to be flattened.
            data_without: Data vector without the systematic. Same shape rules as
                `data_with`. Used only if `delta` is None.
            n_obs: Expected length N of the flattened data vector.

        Returns:
            np.ndarray: delta as a 1D vector of length N.

        Raises:
            ValueError: If neither `delta` nor both `data_with` and `data_without`
                are provided, or if shapes are inconsistent with `n_obs`.
        """
        if delta is not None:
            return self._flatten_data_vector(delta, n_obs)
        if data_with is None or data_without is None:
            raise ValueError("provide `delta` or both `data_with` and `data_without`")
        return self._flatten_data_vector(data_with, n_obs) - self._flatten_data_vector(data_without, n_obs)

