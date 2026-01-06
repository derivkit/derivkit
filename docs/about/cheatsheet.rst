.. |dklogo| image:: ../assets/logos/logo-black.png
   :alt: DerivKit logo black
   :width: 32px


|dklogo| Cheat Sheet: Choosing the Right Method
===============================================

This cheat sheet provides practical guidance for choosing a numerical
differentiation strategy in DerivKit based on the characteristics of the
function being differentiated.

It is intended as a quick reference rather than a strict decision rule;
in practice, multiple methods may be worth trying for difficult cases.

----

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - **Situation**
     - **Recommended method**
     - **Why**
   * - Smooth, cheap function
     - Finite differences
     - Fast and accurate for smooth functions
   * - Slightly noisy function
     - Ridders finite differences
     - Richardson extrapolation improves stability over simple finite differences
   * - Moderate or structured noise
     - Local polynomial fit
     - Local regression smooths noise better than finite differences
   * - High noise / messy signal
     - Adaptive polynomial fit (Chebyshev)
     - Robust trimming, Chebyshev grid, and fit diagnostics
   * - Expensive function
     - Adaptive polynomial fit (Chebyshev)
     - Achieves stable derivatives with fewer function evaluations near ``x0``
   * - Need robustness and diagnostics
     - Adaptive polynomial fit (Chebyshev)
     - Provides fit quality metrics, degree adjustment, and suggestions
   * - Unsure / first attempt
     - Local polynomial fit
     - Good default when function behavior is not well known
