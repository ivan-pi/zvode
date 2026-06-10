# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `ZVODEResult` gains `success` (bool), `status` (int, SciPy semantics), and
  `message` (str) fields, so callers no longer interpret raw ZVODE `ISTATE`
  values; the result is now duck-type compatible with
  `scipy.integrate.OdeResult` (`if not sol.success: print(sol.message)`)
- `ZVODEError` (subclass of `RuntimeError`) raised on solver failure; it
  carries the partial trajectory accumulated up to the failure point on its
  `result` attribute (`success=False`, `status=-1`), instead of discarding it.
  Existing `except RuntimeError` handlers keep working
- `Memory Safety` CI workflow (`.github/workflows/memory-safety.yml`):
  builds the C binding layer with AddressSanitizer + UndefinedBehaviorSanitizer
  and runs the test suite under them, plus a strict-warnings job that compiles
  `src/_zvode.c` with `-Wall -Wextra -Wpedantic -Werror` under both gcc and
  clang (with `-Wno-error=pedantic`, so the one unavoidable
  `void *`->function-pointer callback cast still warns but does not fail the
  build).  Two opt-in CMake options drive these (`ZVODE_SANITIZE`,
  `ZVODE_STRICT_WARNINGS`), both scoped to the C source so the vendored Fortran
  is untouched
- `numba` optional-dependency extra, kept separate from `test`.  The numba
  callback tests self-skip when numba is absent, so only two CI jobs install
  `.[test,numba]` and exercise them — `Tests (Debug)` (debug build) and the
  ubuntu-latest / Python 3.12 cell of the `Tests` matrix (release build); every
  other job stays lean and avoids the heavier numba + llvmlite download

### Changed

- `drive_adaptive`'s internal `StepBuf` now uses a plain `malloc`/`realloc`
  backing store instead of NumPy arrays.  The adaptive stepping loop performs
  no Python/NumPy C API calls on its hot path (and defers any error reporting
  to after the loop exits); the NumPy output arrays are built from the raw
  buffer once, after the loop.  This removes the last per-step Python C API
  dependency from the loop, a prerequisite for releasing the GIL around the
  whole `drive_adaptive` integration.  No behavioural change
- `ZVODEResult.__repr__` / `__str__` reworked to a SciPy/MATLAB-aligned
  `key: value` layout: a `message` / `success` / `status` verdict block, then
  `t` / `y`, then the solver counters; arrays render as `[shape dtype]`
  placeholders (the printout is not API and may change in any release)

## [0.3.0] - 2026-06-09

### Added

- Compiled callbacks enabled in `solve_complex_ivp`: ctypes `CFUNCTYPE` and
  numba `@cfunc` function pointers are now accepted for `fun` and `jac`
  (previously raised `NotImplementedError`)
- `ctx` parameter added to `solve_complex_ivp` (`ctypes.c_void_p`): passes
  user data through to compiled callbacks without requiring a Python closure
- C-level integration loops `_zvode.drive_knots` and `_zvode.drive_adaptive`;
  C backend is now the default
- Sphinx documentation site deployed to GitHub Pages
  (`docs/` converted to RST; autodoc API reference; intersphinx cross-links
  to NumPy and SciPy)
- Python-layer argument validation: all user-facing parameters validated with
  informative exceptions before reaching the Fortran layer; warnings added for
  `max_order` caps and bandwidth mismatches
- Python type annotations on the `solve_complex_ivp` public signature
- Docstring improvements: `Examples` sections in `solve_complex_ivp` and
  `ZVODE`, and a `See Also` section in `ZVODE`

### Removed

- `in_place` parameter removed from `solve_complex_ivp`; callback kind is now
  detected automatically from the argument type

### Fixed

- `nfev` and `njev` counters now correctly account for the shape-probe call
  made during `_validate_fun_shape` / `_validate_jac_shape` at setup
- H0 sign corrected for backward integration in `solve_complex_ivp`
- XERRWD diagnostic output routed to stderr (was stdout); library now silent
  by default (`MESFLG = 0`), matching library convention
- Three comment typos in `zvode.f` fixed, ported from SciPy's upstream review

### Changed

- Redundant input checks in the C wrapper replaced with debug-only assertions

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

[0.3.0]: https://github.com/ivan-pi/zvode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ivan-pi/zvode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ivan-pi/zvode/releases/tag/v0.1.0
