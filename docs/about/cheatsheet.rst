Cheat Sheet: Choosing the Right Method
======================================

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - **Situation**
     - **Recommended method**
     - **Why**
   * - Smooth, cheap function
     - Finite Difference
     - Fast and accurate for clean functions
   * - Slightly noisy function
     - Ridders Finite Difference
     - Richardson extrapolation with error control
   * - Moderate or structured noise
     - Simple Polynomial Fit
     - Local regression smooths noise better than finite differences
   * - High noise / messy signal
     - Adaptive PolyFit (Chebyshev)
     - Robust trimming, Chebyshev grid, diagnostics
   * - Expensive function
     - Adaptive PolyFit (Chebyshev)
     - Fewer evaluations and stable fit around ``x0``
   * - Need robustness + diagnostics
     - Adaptive PolyFit (Chebyshev)
     - Fit quality metrics, degree adjustment, suggestions
