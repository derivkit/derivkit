.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Installation
=====================

DerivKit can be installed from `PyPI <https://pypi.org>`_ through _e.g._ ``pip``:

.. code-block:: bash
   pip install derivkit

DerivKit can additionally be installed from source.
To create an editable installation, run

.. code-block:: bash

   pip install -e derivkit@git+https://github.com/derivkit/derivkit



Optional dependencies
---------------------


JAX autodiff
^^^^^^^^^^^^


DerivKit includes optional JAX-based autodiff helpers and an opt-in autodiff backend.

To enable them, install the JAX extra::

  pip install "derivkit[jax]"

For GPU or accelerator support, follow the official JAX installation instructions
first, then install DerivKit with the extra above.
