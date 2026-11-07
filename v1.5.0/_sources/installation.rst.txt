.. |dklogo| image:: /assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Installation
=====================

DerivKit is available through the following channels:

.. dropdown::  `PyPI <https://pypi.org/project/derivkit/>`_

    To install DerivKit with *e.g.* ``pip``, use::

        pip install derivkit

.. dropdown::  `Conda-forge <https://anaconda.org/channels/conda-forge/packages/derivkit/overview>`_

    .. code-block::

        conda install --channel conda-forge derivkit

.. dropdown::  `DerivKit source <https://github.com/derivkit/derivkit>`_

    .. code-block::

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
