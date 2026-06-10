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
  80 pytest tests across the two test files as of 0.3.0)
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
the loop into C (and enabling fully compiled runs with cfunc callbacks) was
deferred — it ultimately landed in 0.3.0.

- [x] Add a procedural `solve_complex_ivp()` function:
  - [x] Version A: driver stores every automatically chosen step (`save_steps=True`)
  - [x] Version B: uses `ZVINDY` to insert interpolated points per step (`refine=N`)
  - [ ] `tcrit` support (`ITASK=4`): lets users specify intermediate times the solver
    must not overshoot, e.g. at discontinuities in a forcing function.  Related to
    `allow_overshoot` but distinct — `tcrit` is a set of critical times, not a
    global on/off switch.  Needs design and testing.
    **Deferred to a future version.**
- [x] Ensure that when the RHS callback (and Jacobian) are compiled (e.g. via numba
  `@cfunc`, ctypes, or Cython), no Python frames are entered during integration.
  **Landed in 0.3.0** — compiled ctypes/numba function pointers run through the
  C-level `drive_knots` / `drive_adaptive` loops; the `NotImplementedError` stub
  is gone.
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
- [x] *(backfilled)* Process-wide `ZVODE_LOCK` held around every integration call in
  `solve.py`, serialising concurrent `solve_complex_ivp` calls as an interim
  thread-safety measure (see Post-1.0 for the proper COMMON-block fix; the `ZVODE`
  class still documents user-side locking instead)
- [x] *(backfilled)* SciPy made an optional dependency: `solve_complex_ivp` works
  without SciPy; importing `ZVODE` without SciPy raises an `ImportError` with an
  install hint (`pip install 'zvode[scipy]'`)


---

## [RELEASED] 0.3.0 — Fortran, build system, compiled callbacks, and C-layer loops

Work in this release was originally scoped to the Fortran layer, the CMake build
system, and developer-facing conveniences such as type stubs and validation.  In
practice it also absorbed most of the performance work planned for 0.4.0 (C-level
integration loops, compiled callbacks) and the documentation infrastructure planned
for 1.0.0; those items are recorded here, where they actually shipped.

- [x] Add a test that asserts `NotImplementedError` is raised when a ctypes or numba
  cfunc function pointer is passed, so the compiled-callback stub is covered
  (superseded within this release: compiled callbacks were enabled, the `in_place`
  parameter was removed from `solve_complex_ivp` — callback kind is now detected
  automatically from the argument type — and the stub tests were replaced by
  `test_compiled_callback_works` plus the dedicated `test_ctypes_callbacks.py` and
  `test_numba_callbacks.py` suites)
- [x] Domain checking of arguments in Python: detect invalid inputs before the
  integration loop begins so that proper exceptions with informative messages are
  raised, without duplicating checks across all three layers (Python / C / Fortran).
  ZVODE performs its own runtime checks and reports errors via `istate`; the Python
  layer covers what ZVODE cannot catch early or cannot report clearly.
  > All user-facing parameters are now validated in Python before the Fortran layer
  > is reached. Redundant checks in the C wrapper have been downgraded to asserts or
  > guarded with `ZVODE_DEBUG`. The `_resolve_miter` helper consolidates Jacobian and
  > band argument validation in one place.
- [x] Fix `nfev`/`njev` evaluation counters: `_validate_fun_shape` and
  `_validate_jac_shape` each call the user callback once at construction time to
  probe the return shape, but this call is not counted toward `nfev` / `njev`.
  Applies only to the `in_place=False` Python callback path; `in_place=True` and
  compiled callbacks skip the probe and are already exact.
  > `solve_complex_ivp`: local `nfev`/`njev` ints initialised to the probe count
  > (1 or 0) before integration; Fortran `iwork[11/12]` is added at result
  > construction.  `ZVODE` class: delta accumulator (`_nfe_last`/`_nje_last`)
  > carries the probe offset through every step without re-reading the full
  > Fortran counter.  `test_counters.py`: `xfail` decorators removed (tests now
  > pass); offset-pinning tests removed as redundant.
- [ ] Add option to expose `ZEWSET` and `ZWNORM` as callback functions
  **Deferred to a future version (post-1.0.0).**
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
- [x] Lower the time-stepping loop from Python into the C extension layer
  (originally planned for 0.4.0):
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
- [x] Enable compiled callbacks (numba `@cfunc`, ctypes `CFUNCTYPE`): wire up the
  `fun_addr` / `jac_addr` path in `solve_complex_ivp` through `drive_knots` /
  `drive_adaptive`, removing the `NotImplementedError` stub added in 0.2.0
  (originally planned for 0.4.0)
- [x] `ctx` parameter on `solve_complex_ivp` (`ctypes.c_void_p`): passes user data
  through to compiled callbacks without requiring a Python closure
- [x] Public ctypes signature types `ZVODE_FUN_CTYPE` / `ZVODE_JAC_CTYPE` and lazy
  numba signatures `zvode_fun_sig` / `zvode_jac_sig` (numba is imported only on
  first access, so it remains an optional dependency)
- [x] Fix H0 sign for backward integration in `solve_complex_ivp`
- [x] `solve_complex_ivp` raises a catchable `RuntimeError` on solver failure,
  reporting the ZVODE `ISTATE` code, message, and how far integration got;
  covered by `test_max_num_steps_exceeded` (the partial-trajectory half of this
  item remains open — see 0.4.0)
- [x] Documentation infrastructure (originally planned for 1.0.0): Sphinx build
  with `furo` theme (`docs/conf.py`), GitHub Pages deploy workflow (`docs.yml`),
  autodoc API reference (`docs/api.rst`), how-to guides for the procedural API
  and compiled callbacks, `intersphinx_mapping` for scipy and numpy


---

## 0.4.0 — Performance follow-through and API hardening

The C-level integration loops and compiled callbacks originally planned for this
release landed early, in 0.3.0.  What remains here is the GIL-free execution path,
result-object polish, and benchmarks ahead of the 1.0.0 API freeze.

- [ ] GIL-free path: release the GIL around `drive_knots` / `drive_adaptive` when
  both callbacks are compiled.  `drive_knots` is already unblocked because its loop
  body contains no Python API calls other than inside the `cb.error` branch, which
  is never taken for compiled callbacks; `drive_adaptive` is blocked on the
  `StepBuf` refactor below.
- [ ] Refactor `StepBuf` to use plain `malloc`/`realloc` instead of NumPy arrays as
  its backing store, so the adaptive loop in `drive_adaptive_py` contains no Python
  C API calls when compiled callbacks are in use.  The final output arrays are
  constructed from the raw buffer only after the loop exits (and the GIL is
  reacquired).  This is a prerequisite for releasing the GIL around the entire
  `drive_adaptive` loop.
- [~] API hardening: review and stabilise the procedural interface signatures,
  return types, and error reporting ahead of the 1.0.0 API freeze.
  > Type annotations on `solve_complex_ivp` are done (landed in 0.3.0).
  > `ZVODEResult.__repr__` implemented; fields `t`, `y`, `nfev`, `njev`, `nlu`,
  > `nsteps`, `nni`, `ncfn`, `netf` are all exposed.  `ZVODEResult` itself was
  > removed from the public namespace (importable from `zvode.solve`, not
  > re-exported from `zvode`).  Final decision on which of `nsteps`, `nni`,
  > `ncfn`, `netf` to keep vs. drop is still pending.
- [x] Finish the `ZVODEResult` struct: add `success` (bool) and `message` (str)
  fields so users do not have to interpret raw `istate` values themselves; aligns
  the return type with the conventions set by `scipy.integrate.OdeResult`.
  Before freezing the field set, also compare with what diffrax does
  (`diffrax.Solution`: a `result` enum plus a `stats` dict) and borrow whatever
  conventions make sense.
  > Implemented per `docs/result-object-design.md`: `success`, `status`
  > (SciPy semantics, `success := status >= 0`), and `message` added; failure
  > raises `ZVODEError` carrying the partial result; counters stay flat (no
  > nested `stats` object); `sol` / `t_events` / `y_events` remain reserved.
  > `__repr__` / `__str__` reworked to the SciPy/MATLAB-aligned layout.
  > Covered by `test/test_result_object.py`.
- [x] Make the partial trajectory accumulated up to the failure point accessible
  from the `RuntimeError` raised on solver failure (or from a result object);
  the exception and its test landed in 0.3.0, but the partial trajectory is
  currently discarded when the exception is raised
  > `solve_complex_ivp` now raises `ZVODEError` (a `RuntimeError` subclass)
  > whose `result` attribute holds the truncated `t` / `y` trajectory plus all
  > counters; verified in `test/test_result_object.py`.
- [ ] Benchmarks: add a small benchmark suite to quantify the overhead reduction
  relative to 0.2.0 and to the OdeSolver interface; publish work-precision
  diagrams (accuracy vs. cost on a few standard stiff complex problems, e.g.
  the QME demo) as a documentation page
- [ ] Cross-validation suite against `scipy.integrate.ode('zvode')`: run identical
  problems through both wrappers and assert the trajectories agree to tolerance.
  Both wrap the same Fortran core, so this is a near-free regression guard for
  the C-layer loops and option mapping.
- [x] Memory-safety CI job: build the C extension with ASan/UBSan (or run the
  test suite under valgrind) in a dedicated workflow.  The hand-written C loops,
  the growable `StepBuf`, and the raw function-pointer callbacks are the risk
  surface; the `malloc`/`realloc` refactor above makes this more important,
  not less.
  > `.github/workflows/memory-safety.yml` (ubuntu-latest, gfortran/gcc).  Job
  > `sanitizers` builds `src/_zvode.c` with `-fsanitize=address,undefined`
  > (`-fno-sanitize-recover=all`) via the `ZVODE_SANITIZE` CMake option and runs
  > the suite with the ASan runtime `LD_PRELOAD`ed (`detect_leaks=0`, since
  > CPython retains allocations at shutdown).  Job `strict-warnings` compiles the
  > C layer under `-Wall -Wextra -Wpedantic -Werror` against both gcc and clang
  > via the `ZVODE_STRICT_WARNINGS` option (with `-Wno-error=pedantic`, so the
  > one unavoidable `void *`→function-pointer callback cast still warns but does
  > not fail the build).  Both options are scoped to the C source so the
  > vendored Fortran is untouched.
- [x] Wire the standalone Fortran test programs into CTest and run them in CI.
  > The three orphaned drivers targeted ZVODE calling conventions (legacy F77
  > `external` and a raw `bind(c)` procedure) that the current functor-based
  > `zvode_mod` API no longer provides, so they could never have compiled.  They
  > were rewritten as native drivers and split across both compiled APIs: two
  > free-form `.f90` drivers (`test/test_zvode_decay.f90`,
  > `test_zvode_complex_oscillator.f90`) extend the abstract `zvode_fun` /
  > `zvode_jac` types and assert with `if (predicate) error stop <n>`, while
  > `test/test_zvode_constant.c` drives the C API in `extern/zvode.h`, covering the
  > `bind(c)` layer in `extern/c_zvode.f90` that was otherwise exercised only
  > through Python.  This covers the compiled core, the linalg backend, and the
  > C binding without the Python layer.  A standalone CMake configure
  > (`-DZVODE_BUILD_TESTS=ON`, default ON outside the scikit-build wheel build)
  > builds them against the shared `zvode` core library (`libzvode`, also linked
  > by the `_zvode` Python extension so the sources and linalg-backend deps are
  > defined once) and registers each with `add_test`; the `Fortran tests`
  > workflow runs `ctest` under gfortran/gcc for both the LAPACK and LINPACK
  > backends.  This unblocks the multi-compiler conformance item below.
- [ ] Build-side hardening (needs scoping): the pure C, Fortran, and CMake build
  side still needs work in general — e.g. clean compiles under strict warning
  flags for the C extension and the Fortran layer, and a review of the CMake
  setup against current best practice


---

## 1.0.0 — Stable release with online documentation

The 1.0.0 milestone signals API stability.  The fast procedural interface completed
in 0.4.0 must handle the case where the RHS and Jacobian are fully compiled (no
Python callbacks), so that the entire integration runs in compiled code without GIL
round-trips.

- [ ] Procedural interface declared stable (no breaking changes after this point)
- [ ] Remove the Python-level integration loops (`_zvode_adaptive`, `_zvode_knots`
  in `solve.py`) that were retained in 0.3.0 as a debugging reference, along with
  the `ZVODE_BACKEND` environment-variable fallback, once the C path has been
  sufficiently battle-tested.  No deprecation cycle is needed pre-1.0.
- [~] Minimalistic documentation hosted on GitHub Pages (the build infrastructure —
  Sphinx/`furo`, `docs.yml` deploy workflow, autodoc API reference, how-to guides,
  intersphinx — all landed in 0.3.0; see that section):
  - [~] At least one standalone worked example included in the docs
    (`banded_jacobian.rst` is a full worked tutorial in the toctree; decide
    whether it satisfies this item or whether the `docs/demo_*.py` scripts
    should also be linked from the documentation pages)
  - [ ] Non-linear examples: the existing examples are mostly linear; add some
    non-linear ones that are more attractive and more taxing on the solver
    (e.g. complex Ginzburg–Landau, a Kerr / nonlinear Schrödinger oscillator)
- [ ] Fortran standard conformance: build and run the test suite with multiple
  compilers — gfortran (current CI default), ifx, flang, nagfor, lfortran.
  nagfor's strict checking mode is particularly valuable for conformance;
  lfortran support may be limited by the Fortran 2003 abstract-class callbacks
  in the modified `zvode.F`.
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

- [ ] (minor release) Thread safety: refactor the Fortran library to remove the
  process-global COMMON blocks, replacing them with a derived type encapsulating
  the integrator state; employ automated refactoring tools for this transformation.
  Until then, the process-wide `ZVODE_LOCK` in `solve.py` (added in 0.2.0)
  serialises all `solve_complex_ivp` calls as an interim measure, and the `ZVODE`
  class documents user-side locking.
  - [ ] Remove the interim `ZVODE_LOCK` once the COMMON-block refactor is complete
    and concurrent calls are genuinely safe
- [ ] Python Array-API / cupy support: support execution in the memory spaces of
  other array libraries.
- [ ] 64-bit integer build variant (ILP64) for very large systems.
- [ ] conda-forge feedstock: for the scientific audience, installability via
  conda is itself a trust signal.  Wait until after 1.0.0 so the recipe tracks
  a stable API.


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
