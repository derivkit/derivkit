Quick Start
===========

.. code-block:: python

    from derivkit import DerivativeKit

    def function(x):
        return x**2 + x

    dk = DerivativeKit(function, x0=1.0)
    print("Polynomial Fit:", dk.differentiate(order=1, method="local-polynomial"))
    print("Adaptive Fit:", dk.differentiate(order=1, method="adaptive"))
    print("Finite Difference:", dk.differentiate(order=1, method="finite"))
    print("Finite Difference with Richardson:", dk.differentiate(order=1, method="finite", extrapolation="richardson"))
    print("Finite Difference with Ridders:", dk.differentiate(order=1, method="finite", extrapolation="ridders"))
    print("Finite Difference with Gauss-Richardson:", dk.differentiate(order=1, method="finite", extrapolation="gauss-richardson"))

