Quick Start
===========

.. code-block:: python

    from derivkit import DerivativeKit

    def simple_function(x):
        return x**2 + x

    dk = DerivativeKit(simple_function, x0=1.0)
    print("Adaptive:", dk.differentiate(order=1, method="adaptive"))
    print("Finite Difference:", dk.differentiate(order=1, method="finite"))
