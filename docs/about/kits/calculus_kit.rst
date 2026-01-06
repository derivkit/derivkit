Calculus methods
================

DerivKit provides a unified interface for computing gradients, Jacobians,
Hessians, and higher-order derivative tensors through
:class:`derivkit.calculus_kit.CalculusKit`.

These objects describe the local behaviour of a model around a chosen expansion
point and are central to optimization, uncertainty propagation, Fisher and
Laplace approximations, and higher-order likelihood expansions such as DALI.

All calculus objects are built on top of DerivKit’s numerical differentiation
infrastructure. In practice, this means that *any* derivative backend supported
by :class:`DerivativeKit` (finite differences, polynomial fits, adaptive fits,
tabulated functions, etc.) can be used transparently to compute gradients and
higher-order tensors.

Runnable examples are collected in :doc:`../examples/index`.


Gradient
--------

**What it is**

The gradient is the vector of first derivatives of a scalar-valued function
with respect to its parameters,

.. math::

   \nabla f(\theta) = \left( \frac{\partial f}{\partial \theta_1},
   \dots, \frac{\partial f}{\partial \theta_N} \right).

It describes the local linear sensitivity of the function and defines the
direction of steepest ascent or descent.

**How it is computed**

Each component of the gradient is computed as a first-order derivative with
respect to one parameter, holding all others fixed. Internally,
:class:`CalculusKit` constructs the required derivative calls and delegates the
actual numerical differentiation to :class:`DerivativeKit`.

The choice of differentiation method (finite differences, adaptive polynomial
fit, etc.) is fully configurable through the same interface used for scalar
derivatives.


**Examples:**
A basic gradient computation is shown in :doc:`../examples/calculus/gradient`.


Jacobian
--------

**What it is**

The Jacobian generalizes the gradient to vector-valued functions. For a function
mapping an input vector to an output vector,

.. math::

   f : \mathbb{R}^N \rightarrow \mathbb{R}^M,

the Jacobian describes the local linear map between input and output,

.. math::

   J_{ij} = \frac{\partial f_i}{\partial \theta_j}.

It encodes how each output component responds to small changes in each input
parameter.

**How it is computed**

The Jacobian is assembled by computing gradients of each output component with
respect to the input parameters. :class:`CalculusKit` handles the bookkeeping
over output indices, while all numerical derivatives are evaluated via
:class:`DerivativeKit`.

This allows Jacobians of arbitrary shape to be computed using the same
derivative engines as scalar gradients.


**Examples:**
A basic Jacobian computation is shown in :doc:`../examples/calculus/jacobian`.

Hessian
-------

**What it is**

The Hessian is the matrix of second derivatives of a scalar-valued function,

.. math::

   H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}.

It describes the local curvature of the function and is central to quadratic
approximations of likelihoods and posteriors.

In many inference applications, the Hessian of the negative log-likelihood
evaluated at its minimum defines the observed Fisher information matrix.

**How it is computed**

Second derivatives are computed by applying numerical differentiation twice,
either directly or through nested derivative evaluations. :class:`CalculusKit`
constructs the full Hessian tensor and ensures symmetry where appropriate.

All second-order derivatives rely on the same derivative backend selected in
:class:`DerivativeKit`, ensuring consistent numerical behaviour across orders.


**Examples:**
A basic Hessian computation is shown in :doc:`../examples/calculus/hessian`.

Hyper-Hessian
-------------

**What it is**

The hyper-Hessian refers to third- and higher-order derivative tensors
(often called higher-order Hessians or hyper-Hessians) of a scalar-valued function.
These tensors encode departures from purely quadratic
behaviour and capture local non-Gaussian structure.

Such higher-order derivatives are required for systematic likelihood expansions
beyond the Fisher or Laplace approximation, most notably in DALI-based methods.

**How it is computed**

Higher-order tensors are built by recursively applying numerical differentiation
to lower-order derivatives. :class:`CalculusKit` manages the tensor structure and
index ordering, while :class:`DerivativeKit` performs the underlying derivative
evaluations.

Because tensor size and numerical noise grow rapidly with order, these objects
are typically computed only when explicitly requested.


Relationship to DerivativeKit
-----------------------------

:class:`CalculusKit` does not implement numerical differentiation itself.
Instead, it provides a structured interface for assembling derivative objects
(gradients, Jacobians, Hessians, and beyond) using the differentiation engines
implemented in :class:`DerivativeKit`.

This separation ensures that improvements or extensions to derivative methods
automatically propagate to all calculus objects, without changing user-facing
APIs.


References
----------

For numerical differentiation methods and stability considerations, see the
references listed in the :doc:`derivative_methods` documentation page.
