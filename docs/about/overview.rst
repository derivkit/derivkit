What is DerivKit?
=================

**DerivKit** is a modular, derivative-centric toolkit for scientific computing.
It provides a common foundation for reliable numerical differentiation and a
collection of higher-level tools that are built on top of that foundation.

At the base of DerivKit is **DerivativeKit**, which handles numerical derivatives
explicitly and robustly. All other components in the package rely on this base
layer for computing and propagating derivatives in a controlled way.

DerivKit is designed for scientific and engineering workflows (including physics,
astronomy, cosmology, and related fields) where functions are often noisy, expensive,
or non-smooth, and where standard automatic differentiation may be unavailable or
inappropriate.

Building on the derivative layer, DerivKit provides additional kits:

- **CalculusKit**, for gradients, Jacobians, Hessians, and mixed partial derivatives
- **ForecastKit**, for Fisher matrices, Fisher bias vectors, and higher-order DALI expansions
- **LikelihoodKit**, for lightweight likelihood utilities that integrate cleanly with
  derivative-based workflows

Each kit can be used independently, but all share the same derivative backend,
ensuring consistent numerical behavior across the library.

If you want to jump straight to usage, start with :doc:`../examples/index`.
