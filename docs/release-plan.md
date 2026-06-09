# Release Plan

This document tracks planned and completed work across all zvode releases.

**Status key**

| Mark | Meaning |
|------|---------|
| `[x]` | Done |
| `[~]` | In progress / partial |
| `[ ]` | Not yet started |

---

## [RELEASED] 0.1.0 — SciPy OdeSolver interface

- [x] In `zvode.f`, replace `rpar` and `ipar` with the `ctx` argument
  (done via `c_zvode.f90` wrapper — Python/C layer uses a `ctx` void pointer;
  `rpar`/`ipar` are never exposed to the user)
- [x] Replace LINPACK factorization routines with LAPACK or the Hairer routines —
  both dense and banded variants
- [x] Raise an error if the initial condition is real instead of complex; this
  solver is only supposed to work with complex
  (`UserWarning` raised when `np.isrealobj(y0)`; cast to `complex128` still proceeds)
- [x] "Borrow" any SciPy complex test paths
  (`test_zvode_ivp.py` adapts `scipy/integrate/tests/test_banded_ode_solvers.py`;
  40 pytest tests total across the two test files)
- [x] Implement testing with pytest
- [x] List of references for users (README § References)
- [x] Links to related libraries — ODEPACK, SUNDIALS, and availability of Python wrappers
  (README § Links covers ODEPACK, scipy, R/deSolve; SUNDIALS link added in 0.2.0)
- [x] `_zvode.c`: add thread-safety note — ZVODE uses Fortran COMMON blocks, so
  concurrent calls to `c_zvode` from multiple threads will race; document or add a
  GIL-hold / mutex around `c_zvode` (or any API built on top of it); still
  compatible with multiprocessing
  (Notes section added to `ZVODE` class docstring; even separate instances share
  the same process-global COMMON blocks, so a single process-wide `Lock` is
  required — one instance per thread is not sufficient)


---

## [RELEASED] 0.2.0 — Procedural interface (Python layer)

The goal of this release is to provide a standalone `solve_complex_ivp()` function.
The integration loop is implemented in Python rather than compiled C, so there is
still a round-trip through the Python interpreter on every accepted step.  Pushing
the loop into C (and enabling fully compiled runs with cfunc callbacks) is deferred
to 0.4.0.

- [x] Add a procedural `solve_complex_ivp()` function:
  - [x] Version A: driver stores every automatically chosen step (`save_steps=True`)
  - [x] Version B: uses `ZVINDY` to insert interpolated points per step (`refine=N`)
  - [ ] `tcrit` support (`ITASK=4`): lets users specify intermediate times the solver
    must not overshoot, e.g. at discontinuities in a forcing function.  Related to
    `allow_overshoot` but distinct — `tcrit` is a set of critical times, not a
    global on/off switch.  Needs design and testing.
    **Deferred to a future version.**
- [ ] Ensure that when the RHS callback (and Jacobian) are compiled (e.g. via numba
  `@cfunc`, ctypes, or Cython), no Python frames are entered during integration.
  **Deferred to 0.4.0 — compiled cfunc path raises `NotImplementedError` for now.**
- [ ] Document the `ZVINDY` COMMON block constraint: `ZVINDY` is only valid
  immediately after a step on the same ZVODE instance; make this explicit in the API.
  (N/A for the Python bindings — the C/Python `ZVINDY` does not share this Fortran
  COMMON block design constraint)
- [x] Add SUNDIALS link to README Links table
- [x] Mention the ZVODE rewrite in C and point to the (older) scipy `ode` integrator
  that wraps ZVODE
- [x] Add a `solve_complex_ivp` quick-start section to the README
- [x] Start a CHANGELOG (keep-a-changelog format); backfill 0.1.0 entry
- [x] Replace the bare `(t, y[, stats])` tuple return from `solve_complex_ivp` with a
  result object (`t`, `y`, `stats` attributes always present); eliminates the
  variable-arity `ret_stats=True` pattern and keeps the return type stable for
  future fields (e.g. `dense_output`).  `ret_stats` keyword removed.


---

## 0.3.0 — Fortran, build system, and developer sugar

Work in this release is concentrated on the Fortran layer, the CMake build system,
and developer-facing conveniences such as type stubs and validation.

- [x] Add a test that asserts `NotImplementedError` is raised when `in_place=True` is
  used with a ctypes or numba cfunc function pointer, so the compiled-callback stub
  is covered and a regression will fire automatically when 0.4.0 enables the path
  (`test_compiled_callback_not_yet_implemented` in `test_solve_complex_ivp.py`;
  `test_compiled_callback_requires_in_place` covers the `in_place=False` guard)
- [x] Domain checking of arguments in Python: detect invalid inputs before the
  integration loop begins so that proper exceptions with informative messages are
  raised, without duplicating checks across all three layers (Python / C / Fortran).
  ZVODE performs its own runtime checks and reports errors via `istate`; the Python
  layer covers what ZVODE cannot catch early or cannot report clearly.
  > All user-facing parameters are now validated in Python before the Fortran layer
  > is reached. Redundant checks in the C wrapper have been downgraded to asserts or
  > guarded with `ZVODE_DEBUG`. The `_resolve_miter` helper consolidates Jacobian and
  > band argument validation in one place.
- [ ] Fix `nfev`/`njev` evaluation counters: `_validate_fun_shape` and
  `_validate_jac_shape` each call the user callback once at construction time to
  probe the return shape, but this call is not counted toward `nfev` / `njev`.
  Applies only to the `in_place=False` Python callback path; `in_place=True` and
  compiled callbacks skip the probe and are already exact.
  > Fix: maintain a Python-side probe counter inside the `_wrapped_fun` /
  > `_wrapped_jac` closures and add it to `iwork[11/12]` at readout.  Both probe
  > calls are marked `FIXME` in `_helpers.py`.
  > `test_counters.py` documents the expected correct behaviour with `xfail` tests
  > and pins the current off-by-one discrepancy so a regression is detectable.
- [ ] Add option to expose `ZEWSET` and `ZWNORM` as callback functions
- [x] Provide CMake option to use external BLAS; fallback to vendored procedures
  (`ZVODE_LINALG_BACKEND` cache variable: `LAPACK` (default, uses external LAPACK)
  or `LINPACK` (uses vendored routines + external BLAS))
- [x] Ensure `implicit none` is used and all variables are strictly typed
  (`c_zvode.f90` uses `implicit none` throughout; `zvode.f` is upstream — not modified)
- [x] Add Python type annotations to `solve_complex_ivp` (inline annotations on the
  public signature; `.pyi` stubs for the internal `_zvode` C extension are
  intentionally omitted — users do not interact with it directly)
- [x] Add short docstring `Examples` sections to `ZVODE` and `solve_complex_ivp`
  (a few `>>>` lines each; enough for `help()` to be useful in a REPL)
- [x] Add `See Also` section to `ZVODE` linking to `scipy.integrate.OdeSolver`
  (the base class) and `scipy.integrate.solve_ivp` (the driver that accepts
  `method=ZVODE`); section sits between Attributes and Notes per numpydoc order
- [x] Check older SciPy versions for any bug fixes not yet incorporated, and verify
  original licenses are properly attributed
  > Reviewed SciPy git history for `zvode.f`. Applied: three comment typo fixes
  > (`interrrupted`, `Threshhold`, `succesful`); XERRWD output redirected to stderr;
  > MESFLG defaulted to 0 so the library is silent unless the user calls `XSETF(1)`.
  > LINPACK→LAPACK migration and the MXSTEP warning removal were already handled
  > in this project. PR #99.
- [x] Note on NEQ reset: when NEQ is reset during integration (`ISTATE = 3`), we must
  also modify the NEQ component of `fun` and `jac` class instances; not relevant to
  the SciPy interface as `neq` can't change during integration — fixed in PR #98


---

## 0.4.0 — Performance: C-layer time-stepping, compiled callbacks, and API hardening

The procedural interface from 0.2.0 called back into Python once per accepted step.
This release pushes the integration loop into the C extension and completes the
compiled-callback path (originally planned for 0.2.0), so that when RHS/Jacobian
are compiled the entire run executes without touching the Python interpreter.

- [x] Lower the time-stepping loop from Python into the C extension layer:
  - [x] `_zvode.drive_knots`: C loop over output knots using `ITASK=1`; accepts
    pre-allocated output arrays; returns `(istate, knots_completed)` tuple so
    Python can truncate on failure without raising inside C
  - [x] `_zvode.drive_adaptive`: C loop in single-step mode (`ITASK=5` or `2`);
    uses a growable `StepBuf` backed by NumPy arrays (starts at 10 columns,
    doubles on overflow); supports `ZVINDY` interpolation when `refine > 1`
  - [x] `ZVODE_BACKEND` environment variable: set to `"python"` to fall back to
    the pure-Python loops for debugging; C is the default.  The Python loops are
    retained for now as a reference / debugging aid; they will be deprecated and
    removed before 1.0.0 once the C path is battle-tested.
  - [ ] GIL-free path: no GIL release yet — requires compiled callbacks (next item)
- [x] Enable compiled callbacks (numba `@cfunc`, ctypes `CFUNCTYPE`): wire up the
  `fun_addr` / `jac_addr` path in `solve_complex_ivp` through `drive_knots` /
  `drive_adaptive`, removing the `NotImplementedError` stub added in 0.2.0.
  The C loops are in place; this item is now unblocked.
- [~] API hardening: review and stabilise the procedural interface signatures,
  return types, and error reporting ahead of the 1.0.0 API freeze.
  > Type annotations on `solve_complex_ivp` are done (landed in 0.3.0 work).
  > `ZVODEResult.__repr__` implemented; fields `t`, `y`, `nfev`, `njev`, `nlu`,
  > `nsteps`, `nni`, `ncfn`, `netf` are all exposed.  Final decision on which of
  > `nsteps`, `nni`, `ncfn`, `netf` to keep vs. drop is still pending.
- [ ] Richer `ZVODEResult` attributes: add `success` (bool) and `message` (str)
  fields so users do not have to interpret raw `istate` values themselves; aligns
  the return type with the conventions set by `scipy.integrate.OdeResult`
- [ ] Add a test that `solve_complex_ivp` raises a catchable exception on solver
  failure so that users can safely wrap calls in a `try`/`except` block; verify
  that the partial trajectory accumulated up to the failure point is accessible
  from the exception or the result object
- [ ] Benchmarks: add a small benchmark suite to quantify the overhead reduction
  relative to 0.2.0 and to the OdeSolver interface


---

## 1.0.0 — Stable release with online documentation

The 1.0.0 milestone signals API stability.  The fast procedural interface completed
in 0.4.0 must handle the case where the RHS and Jacobian are fully compiled (no
Python callbacks), so that the entire integration runs in compiled code without GIL
round-trips.

- [ ] Procedural interface declared stable (no breaking changes after this point)
- [~] Minimalistic documentation hosted on GitHub Pages:
  - [x] Sphinx build configured (`docs/conf.py`, `furo` theme)
  - [x] GitHub Actions workflow to build and deploy on each push to `main`
    (`docs.yml`)
  - [x] API reference (`docs/api.rst`, autodoc)
  - [x] User guide: how-to guides for the procedural API and compiled callbacks
    (`docs/how-to-procedural-api.rst`, `docs/how-to-compiled-callbacks.rst`)
  - [x] `intersphinx_mapping` for scipy and numpy configured in `conf.py`
  - [ ] At least one standalone worked example included in the docs
- [~] Binary wheel distribution via cibuildwheel:
  - [x] `wheels.yml` builds Linux x86-64 (manylinux) and macOS arm64 wheels
  - [~] PyPI publish job with trusted publishing exists but needs hardening before
    it is considered production-ready
- [ ] Windows wheel builds: add a `windows-latest` runner to the CI matrix,
  document the required Fortran toolchain (gfortran via MSYS2/mingw-w64 or Intel
  ifx), and add `"Operating System :: Microsoft :: Windows"` to the
  `pyproject.toml` classifiers


---

## Post-1.0 — Future work (no API breaks expected)

These items are intentionally deferred.  Thread safety and COMMON block removal
are internal concerns that should not require any user-visible API changes.

- [ ] Thread safety: ZVODE uses process-global Fortran COMMON blocks, so concurrent
  calls from multiple threads will race.  A process-wide `Lock` (or reentrant
  wrapper) is the correct fix.  Deferred because it requires careful testing and
  adds overhead that single-threaded users should not pay.
- [ ] (minor release) Replace COMMON blocks with a Fortran derived type encapsulating
  the integrator state; employ automated refactoring tools for this transformation.
  This is a prerequisite for true thread safety.
- [ ] Python Array-API / cupy support: support execution in the memory spaces of
  other array libraries.
- [ ] 64-bit integer build variant (ILP64) for very large systems.


---

## Out of scope

These features are genuinely useful but are architecturally out of scope for a
binding to the classic ZVODE Fortran solver, or require effort that is
disproportionate to the library's current stage.  They are documented here so users
understand why they are absent and know where to look instead.  If you have ideas
on how any of these could be tackled, contributions and discussion are welcome.

**Sparse or matrix-free Jacobians**
: ZVODE is a dense/banded Adams-BDF solver and has never had a sparse path.  Adding
  sparse support would require hooking in a sparse direct solver (e.g. SuperLU,
  UMFPACK, or Y12M) or an iterative/Krylov path for the linear algebra step.  This
  is a substantial extension to the solver internals and is not a simple drop-in.

**Event/root-finding**
: ZVODE has no built-in root-detection mechanism.  At the Fortran step-based level,
  users can implement their own event detection by calling `ZVINDY` to interpolate
  within a step and bisect to the root — it requires some work but is doable.
  The forward Python interface (`solve_complex_ivp`) does not yet expose this path.
  Users who need event handling today should use `scipy.integrate.solve_ivp` with
  the `events=` argument — either on the complex system directly via
  `method=ZVODE` (passing the class to `solve_ivp`), or on the equivalent
  doubled real system with a standard solver.

**Automatic detection of non-analytic stiff RHS**
: The analytic requirement for stiff complex-valued problems (Cauchy-Riemann
  equations) cannot be checked automatically for an arbitrary callable.
  Documentation is the correct and only feasible response.


---

## Developer references

- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/array.html)
- [NumPy style guide (numpydoc)](https://numpydoc.readthedocs.io/en/latest/format.html)
