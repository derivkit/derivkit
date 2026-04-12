# Changelog

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
