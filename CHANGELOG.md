# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `solve_complex_ivp(fun, tspan, y0, ...)` — single-call procedural interface
  in the spirit of `scipy.integrate.odeint`; returns a `ZVODEResult`
- `ZVODEResult` — dict-like result object with attribute access; always
  carries `.t`, `.y`, `.nfev`, `.njev`, `.nlu`
- Three output modes controlled by `tspan` and `save_steps`:
  - Adaptive output (`save_steps=True`, default): every accepted internal step
    is collected
  - Endpoint-only (`save_steps=False`): returns scalar `t` and 1-D `y`
  - Knot output (three or more `tspan` values): solution returned at
    caller-supplied times via ZVINDY interpolation
- `refine=N` option: inserts *N*−1 interpolated points between each accepted
  step (adaptive output mode only)
- `allow_overshoot` option: lets the solver step past the endpoint before
  returning (ITASK=2 vs ITASK=5)
- `max_num_steps` option: per-solve cap on accepted steps
- Compiled-callback support via numba `@cfunc` or ctypes `CFUNCTYPE`
  (requires `in_place=True`); the C function pointer bypasses the Python
  interpreter on every RHS/Jacobian evaluation
- Method-aware default `miter`: Adams defaults to `miter=0` (functional
  iteration), BDF defaults to `miter=2` (internally generated Jacobian)
- `scipy.integrate.ode` entry and C-ZVODE transition note added to the README
  Links table
- Quick-start and how-to guide for `solve_complex_ivp` in the README and
  `docs/how-to-procedural-api.md`
- Banded Jacobian tutorial (`docs/banded_jacobian.md`) and demo
- Demo scripts for three additional use cases: complex linear system, non-
  autonomous linear ODE, and reproduction of the SciPy `ode` docs example
- Robustness tests for five complex-analytic ODE problem classes
  (`test/test_extra.py`)

### Fixed

- `solve_complex_ivp` backward integration (negative time direction)
- Partial output in knots mode when integration fails mid-interval

### Changed

- Matplotlib declared as an optional dependency group (`examples`) rather
  than a required dependency
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
- Guard against int32 overflow when computing dense and banded Jacobian
  workspace sizes
- `UserWarning` when the initial condition `y0` has a real dtype
  (integration proceeds after casting to `complex128`)
- Thread-safety note in the `ZVODE` class docstring: the solver uses
  process-global Fortran COMMON blocks; a process-wide lock is required for
  concurrent use from multiple threads
- `__all__` restricting the public API to the `ZVODE` class
- CI workflow for building and verifying source distributions

### Changed

- Replaced LINPACK factorization routines with LAPACK equivalents for both
  dense and banded systems
- Removed `max_steps` parameter: MXSTEP is a per-call step limit, exposed
  through solver options rather than the constructor

[Unreleased]: https://github.com/ivan-pi/zvode/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ivan-pi/zvode/releases/tag/v0.1.0
