Installation
============

DerivKit is currently distributed from source.

.. code-block:: bash

   git clone https://github.com/derivkit/derivkit.git
   cd derivkit
   pip install -e .


Optional dependencies
=====================

JAX autodiff
------------

DerivKit includes optional JAX-based autodiff helpers and an opt-in autodiff backend.

To enable them, install the JAX extra::

  pip install "derivkit[jax]"

For GPU or accelerator support, follow the official JAX installation instructions
first, then install DerivKit with the extra above.
