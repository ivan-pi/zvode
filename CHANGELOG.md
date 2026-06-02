# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Demo scripts in `docs/` for three new use cases: a complex linear system
  (`y' = Ay`), a non-autonomous linear complex ODE, and a reproduction of
  the SciPy `scipy.integrate.ode` documentation example
- Robustness tests for five complex-analytic ODE problem classes

### Changed

- Matplotlib declared as an optional dependency group (`examples`) rather
  than a required dependency
- Example file names and docstrings harmonised; `docs/README.md` added to
  describe the examples folder
- pytest configuration updated to pytest 9 syntax (`[tool.pytest]`);
  removed `filterwarnings = error` override

## [0.1.0] - 2026-06-01

First release of `zvode` — Python bindings to the ZVODE solver for stiff
complex-valued ordinary differential equations.

### Added

- `ZVODE` class implementing the `scipy.integrate.OdeSolver` interface,
  compatible with `scipy.integrate.solve_ivp`
- Adams (non-stiff) and BDF (stiff) linear multistep methods selected via
  the `lmm` parameter; convenience subclasses `ZVODE.Adams` and `ZVODE.BDF`
  provided
- Flexible Jacobian support: dense, banded, and diagonal approximation
- Dense-output interpolation via the ZVINDY routine
- Configurable linear algebra backend (LAPACK or LINPACK) selected at
  build time; LAPACK is the default
- Guard against int32 overflow when computing dense and banded Jacobian
  workspace sizes, with a clear error message
- `UserWarning` when the initial condition `y0` has a real dtype
  (integration proceeds after casting to `complex128`)
- Thread-safety note in the `ZVODE` class docstring: the solver uses
  process-global Fortran COMMON blocks, so a process-wide lock is required
  for concurrent use from multiple threads
- `__all__` restricting the public API to the `ZVODE` class
- CI workflow for building and verifying source distributions

### Changed

- Replaced LINPACK factorization routines with LAPACK equivalents for both
  dense and banded systems
- Removed `max_steps` parameter: MXSTEP is a per-call step limit and is
  now passed through the existing solver options, not as a constructor
  argument

[Unreleased]: https://github.com/ivan-pi/zvode/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ivan-pi/zvode/releases/tag/v0.1.0
