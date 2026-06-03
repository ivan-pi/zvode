# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-06-03

### Added

- `solve_complex_ivp(fun, tspan, y0, ...)` — single-call procedural interface
  in the spirit of `scipy.integrate.odeint`; returns a `ZVODEResult`
- `ZVODEResult` — dict-like result object with attribute access
- Quick-start and how-to guide for `solve_complex_ivp` in the README and
  `docs/how-to-procedural-api.md`
- Banded Jacobian tutorial (`docs/banded_jacobian.md`) and demo
- Demo scripts for three additional use cases: complex linear system, non-
  autonomous linear ODE, and reproduction of the SciPy `ode` docs example
- Robustness tests for five complex-analytic ODE problem classes
  (`test/test_extra.py`)

### Changed

- pytest configuration updated to pytest 9 syntax (`[tool.pytest]`)

## [0.1.0] - 2026-06-01

First release of `zvode` — Python bindings to the ZVODE solver for stiff
complex-valued ordinary differential equations.

### Added

- `ZVODE` class implementing the `scipy.integrate.OdeSolver` interface,
  compatible with `scipy.integrate.solve_ivp`
- Adams (non-stiff) and BDF (stiff) linear multistep methods selected via
  the `lmm` parameter; convenience subclasses `ZVODE_Adams` and `ZVODE_BDF`
  provided
- Flexible Jacobian support: dense, banded, and diagonal approximation
- Dense-output interpolation via the ZVINDY routine
- Configurable linear algebra backend (LAPACK or LINPACK) selected at
  build time; LAPACK is the default
- `__all__` restricting the public API to the `ZVODE` class
- CI workflow for building and verifying source distributions

### Changed

- Replaced LINPACK factorization routines with LAPACK equivalents for both
  dense and banded systems

[0.2.0]: https://github.com/ivan-pi/zvode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ivan-pi/zvode/releases/tag/v0.1.0
