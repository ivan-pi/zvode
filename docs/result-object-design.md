# `ZVODEResult` Design Specification

Status: draft for the 0.4.0 release.

## Goals

- Add `success`, `status`, and `message` fields so users do not have to
  interpret raw ZVODE `ISTATE` values themselves.
- Remain duck-type compatible with `scipy.integrate.OdeResult` (the return
  type of `scipy.integrate.solve_ivp`), so that code written against SciPy —
  `if not sol.success: print(sol.message)` — works verbatim on a
  `ZVODEResult`.  SciPy is the compatibility target, not diffrax or
  DifferentialEquations.jl: zvode already follows SciPy's naming (`t`, `y`,
  `nfev`, `njev`, `nlu`) and callback conventions, and practitioners
  switching solvers are switching between `solve_ivp` and
  `solve_complex_ivp`.
- Stay minimalistic: every field is plain data with an obvious meaning.
  Names are reserved (not added) for features that do not exist yet.
- Provide a stable surface: future enhancements (dense output, events) must
  slot in as *new* fields without changing the meaning of existing ones.

---

## Field inventory

`ZVODEResult` remains a `dict` subclass with attribute access.  After this
change it always contains:

| Field     | Type                  | Origin   | Meaning |
|-----------|-----------------------|----------|---------|
| `t`       | float or `(m,)` float64 | existing | Output time(s); scalar in endpoint-only mode |
| `y`       | `(n,)` or `(n, m)` complex128 | existing | Solution state(s); `y[:, k]` is the state at `t[k]` |
| `success` | bool                  | **new**  | `True` iff the solver reached the end of the integration interval (`status >= 0`, see below) |
| `status`  | int                   | **new**  | Termination reason, SciPy semantics: `0` reached end of tspan, `-1` integration step failed; `1` is reserved for future event termination |
| `message` | str                   | **new**  | Human-readable description of the termination reason |
| `nfev`    | int                   | existing | Number of right-hand side evaluations |
| `njev`    | int                   | existing | Number of Jacobian evaluations |
| `nlu`     | int                   | existing | Number of LU decompositions |
| `nsteps`  | int                   | existing | Number of internal solver steps |
| `nni`     | int                   | existing | Number of nonlinear (Newton/functional) iterations |
| `ncfn`    | int                   | existing | Number of nonlinear convergence failures |
| `netf`    | int                   | existing | Number of local error test failures |

All counters are cumulative tallies over the entire integration, never
per-step quantities: ZVODE zeroes them on the initial call only and
increments them across the whole solve (e.g. `NETF` is documented as "the
number of error test failures of the integrator so far"), and the drivers
carry the `ISTATE=2` continuation state between output knots so the totals
span the full `solve_complex_ivp` call.  Any counter added in the future
must follow the same rule; per-step diagnostics (ZVODE's `HU`, `NQU`, ...)
do not belong on the result object.

The first eight rows make `ZVODEResult` a strict superset of the
`OdeResult` fields that can exist without dense output and events.  The
ZVODE-specific counters (`nsteps`, `nni`, `ncfn`, `netf`) are kept flat
alongside the SciPy ones rather than nested in a `stats` sub-object
(diffrax/DifferentialEquations.jl style), because SciPy keeps counters flat
and extra attributes do not interfere with duck-typing.  Whether all four
are kept past 1.0 is still an open release-plan item; this spec does not
decide it.

### `status` and `success` semantics

`status` follows the `scipy.integrate.solve_ivp` convention, **not** the raw
ZVODE `ISTATE` convention — exposing `ISTATE` is exactly the burden this
change removes:

| Condition                | `status` | `success` |
|--------------------------|----------|-----------|
| Reached the end of tspan | `0`      | `True`    |
| Event termination (future) | `1`    | `True`    |
| Integration step failed  | `-1`     | `False`   |

`success` is **defined as `status >= 0`**, not `status == 0`.  Success is a
set of status codes, not a single code (the lesson behind
`SciMLBase.successful_retcode`): when event support lands, an
event-terminated solve (`status == 1`) is a successful solve, and code that
checks `success` must not need updating.

`status` is a plain `int` in 0.4.0.  Promoting it to an `IntEnum` later
(symbolic names that still compare equal to `-1`/`0`/`1`, in the spirit of
diffrax's `RESULTS` and DiffEq's named retcodes) is a compatible,
purely-additive enhancement and is explicitly out of scope here.

### `message` contents

- On success: SciPy's generic text, `"The solver successfully reached the
  end of the integration interval."`
- On failure: the location of the failure plus the ZVODE-specific detail
  already maintained in `_helpers.MESSAGES`, e.g.
  `"Integration failed at t=1.234, before reaching t=10.0.
  ZVODE ISTATE=-4: Repeated error test failures."`
  Where a remedy is known, the message should state it (e.g. for
  `ISTATE=-1`, suggest increasing `max_num_steps`) — actionable messages,
  in the spirit of diffrax's `RESULTS` texts.

`message` is documentation for humans.  Programmatic discrimination uses
`status` (or `success`); user code must never need to parse `message`.

### `ISTATE` mapping

| ZVODE `ISTATE` | `status` | message detail |
|----------------|----------|----------------|
| `2`            | `0`      | success text |
| `-1`           | `-1`     | Excess work done on this call (try increasing `max_num_steps`) |
| `-2`           | `-1`     | Excess accuracy requested (tolerances too tight for this precision) |
| `-3`           | `-1`     | Illegal input detected |
| `-4`           | `-1`     | Repeated error test failures (check for a singularity, or wrong `method`) |
| `-5`           | `-1`     | Repeated convergence failures (check the Jacobian, or try `method='BDF'`) |
| `-6`           | `-1`     | Error weight became zero (a component vanished with `atol=0`) |

Raw `ISTATE` is not stored on the result.  The value appears verbatim inside
`message`, which is sufficient for bug reports and debugging.

---

## Failure behaviour: raise, carrying the failed result

`solve_complex_ivp` keeps raising on solver failure rather than adopting
SciPy's return-with-`success=False` behaviour.  Raising is safer by
default: SciPy-style code that forgets to check `success` gets a loud
exception instead of silently consuming a truncated trajectory.

The bare `RuntimeError` is replaced by a dedicated exception that carries
the fully-populated result:

```python
class ZVODEError(RuntimeError):
    """Raised when ZVODE cannot advance to the next output point.

    Attributes
    ----------
    result : ZVODEResult
        The partial result accumulated up to the failure point, with
        ``success=False``, ``status=-1``, the failure ``message``, the
        truncated trajectory in ``t``/``y``, and all solver counters.
    """
```

The exception message equals `result.message`.  This gives both
conventions from one mechanism:

```python
try:
    sol = solve_complex_ivp(fun, tspan, y0)
except ZVODEError as exc:
    partial = exc.result          # status == -1, success is False
    plt.plot(partial.t, partial.y[0].real)   # inspect how far it got
```

`ZVODEError` subclasses `RuntimeError`, so existing `except RuntimeError`
handlers keep working.  A `success=False` result is therefore never
*returned* in 0.4.0 — the fields establish the contract, and an opt-in
flag for SciPy's return-instead-of-raise behaviour can be added later if
users ask, without any change to the result object itself.

---

## Reserved names (do not repurpose)

The following names are **reserved** for future features and must not be
used for anything else.  They are *not* added as dead `None` fields in
0.4.0: SciPy's own contract is "None if not requested", and zvode has no
`dense_output` or `events` parameters yet, so omission is consistent.

| Reserved name | Future feature | Notes |
|---------------|----------------|-------|
| `sol`         | Dense output: a callable interpolant over the solved interval | See signature note below |
| `t_events`    | Event support: times at which each event triggered | list of ndarrays, SciPy layout |
| `y_events`    | Event support: states at `t_events` | list of ndarrays, SciPy layout |
| `status == 1` | Event termination | already reserved in the `status` table above |

**Dense-output signature note.**  When `sol` is added, its call signature
shall be `sol(t, k=0)` where `k` is the derivative order, returning the
k-th derivative of the interpolating polynomial.  ZVINDY natively computes
k-th derivatives (the `refine` path already calls it with `k=0`), so this
capability is free at the Fortran level, and it is something
`scipy.integrate.OdeSolution` cannot do (DiffEq's `sol(t, deriv=...)` can).
Committing to the signature now means it never has to change.  The
canonical spelling is SciPy's `result.sol(t)`; making the result object
itself callable (DiffEq-style `sol(t)`) is possible sugar later and is not
part of this spec.

---

## Anti-goals

These are deliberate non-features, to be stated in the `ZVODEResult`
docstring so they survive future contributions:

- **`ZVODEResult` is plain data.**  Arrays, ints, floats, strings, bools —
  nothing else.  It holds no references to `fun`, `jac`, `ctx`, or solver
  workspace arrays, and is therefore picklable as-is.  (Cautionary tale:
  DifferentialEquations.jl stores the problem and algorithm on its
  solution object and consequently needs `strip_solution` to make results
  serializable.)
- **No array interface on the result.**  NumPy slicing on `result.y`
  already provides `sol[i, j]`-style access; the result object itself does
  not implement `__getitem__` beyond its dict behaviour, statistics, or
  plotting hooks.
- **No raw solver state.**  `ISTATE`, `RWORK`, `IWORK`, Nordsieck arrays
  and the like are implementation details; diffrax-style `solver_state` /
  `controller_state` fields are artifacts of JAX's functional constraints
  and have no place here.

---

## Implementation notes

- `success`/`status`/`message` are constructed in `solve_complex_ivp`
  (`src/zvode/solve.py`) at the single point where the result dict is
  built; the failure path constructs the same dict (with the truncated
  `t`/`y` already computed there) before raising `ZVODEError`.
- `ZVODEResult.__repr__` is extended to lead with `success` and `status`
  so a glance at the REPL answers "did it work".
- The class-based `ZVODE` stepper API is unaffected; this spec covers only
  the procedural `solve_complex_ivp` return value.

## Acceptance criteria

1. A successful solve returns a result with `success is True`,
   `status == 0`, and the generic success `message`.
2. A failing solve raises `ZVODEError`; `exc.result.success is False`,
   `exc.result.status == -1`, `exc.result.message` contains the ZVODE
   `ISTATE` value and detail, and `exc.result.t` / `exc.result.y` hold the
   partial trajectory (this also closes the release-plan item on catchable
   exceptions with accessible partial trajectories).
3. `pickle.loads(pickle.dumps(result))` round-trips.
4. A `ZVODEResult` passes for an `OdeResult` in duck-typed code that reads
   `t`, `y`, `success`, `status`, `message`, `nfev`, `njev`, `nlu`.
