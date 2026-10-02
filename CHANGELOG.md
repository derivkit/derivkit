# Changelog

## v1.4.0

Released on 2026-10-02.

### Added

* `CalculusKit` is now able to compute hyper-Hessians of order greater than 3.
* `ForecastKit` is now able to construct DALI quadruplet tensors.
* `ForecastKit` is now able to perform DALI forecasts using the quadruplet tensors.

### Changed

* `ForecastKit.logposterior_dali()` or `ForecastKit.delta_chi2_dali()` will now compute the forecast to the highest available order if `forecast_order=None` is set.

### Removed

* `ForecastKit` will no longer issue a warning if higher-order forecasts do not use the `finite` method.


## v1.3.0

Released on 2026-09-14.

### Added

* `ForecastKit.dali()` now has a `symmetrize` option.
  A value of `True` (default) will tell derivkit to symmetrize all DALI tensors with respect to permutation of any pair of axes.
  **Note**: This implies that, in general, `Forecastkit.dali()` yields a different output compared to older versions.

### Changed

* The default derivative method of `DerivativeKit` has been changed from the adaptive fit to the local polynomial method.
* `ForecastKit` will no longer force a fallback for derivative backends.
  Instead it will warn the user about potential unwanted behaviour.

### Fixed

*  Updates the numerical factors in the chi-squared polynomial constructed in `derivkit.forecasting.expansions.build_delta_chi2_dali()`.


## v1.2.1

### Fixed

* Forced a fallback for derivative backends which washed out higher curvature.


## v1.2.0

### Added

* Added thread-safety options to `ForecastKit` and `CalculusKit`.
* Added a multiprocessing support to `parallel_execute()`, which avoids the Python GIL.
* Added input-level caching of function evaluations.

### Changed

* Exposed the $X-Y$ Gaussian Fisher matrix through `ForecastKit.xy_fisher()`.
* Improved the general documentation.


## v1.1.0

### Added

* Added caching of function and derivative values for forecasting.


## v1.0.2

### Fixed

* Fixed how `dk_kwargs` are passed to `ForecastKit`.


## v1.0.1

### Fixed

* Fixed an issue where the Hessian routine would repeatedly evaluate the input function if the function is vector-valued.
* Fixed an issue where the Hyper-Hessian routine would repeatedly evaluate the input function if the function is vector-valued.
