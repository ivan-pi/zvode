# `ZVODEResult` Design Specification

Status: draft for the 0.4.0 release.

## Goals

- Add `success`, `status`, and `message` so users never interpret raw
  ZVODE `ISTATE` values.
- Stay duck-type compatible with `scipy.integrate.OdeResult`: code like
  `if not sol.success: print(sol.message)` works verbatim.  SciPy — not
  diffrax or DifferentialEquations.jl — is the compatibility target, since
  practitioners switch between `solve_ivp` and `solve_complex_ivp`.
- Minimal and stable: every field is plain data; names are reserved (not
  added) for features that do not exist yet, so future enhancements slot
  in without changing existing meanings.

## Fields

`ZVODEResult` remains a `dict` subclass with attribute access, always
containing:

| Field     | Type                  | Origin   | Meaning |
|-----------|-----------------------|----------|---------|
| `t`       | float or `(m,)` float64 | existing | Output time(s); scalar in endpoint-only mode |
| `y`       | `(n,)` or `(n, m)` complex128 | existing | Solution state(s); `y[:, k]` is the state at `t[k]` |
| `success` | bool                  | **new**  | `True` iff `status >= 0` |
| `status`  | int                   | **new**  | `0` reached end of tspan, `-1` step failed; `1` reserved for event termination |
| `message` | str                   | **new**  | Human-readable termination reason |
| `nfev`    | int                   | existing | RHS evaluations |
| `njev`    | int                   | existing | Jacobian evaluations |
| `nlu`     | int                   | existing | LU decompositions |
| `nsteps`  | int                   | existing | Internal solver steps |
| `nni`     | int                   | existing | Nonlinear iterations |
| `ncfn`    | int                   | existing | Nonlinear convergence failures |
| `netf`    | int                   | existing | Local error test failures |

This is a strict superset of the `OdeResult` fields that can exist without
dense output and events.  The ZVODE-specific counters stay flat alongside
the SciPy ones (no nested `stats` object); whether all four survive past
1.0 is an open release-plan item.

All counters are cumulative tallies over the entire `solve_complex_ivp`
call (ZVODE zeroes them on the initial call only, and the drivers carry
the `ISTATE=2` continuation between knots).  Future counters must follow
the same rule; per-step diagnostics (`HU`, `NQU`, ...) do not belong on
the result.

### `y` shape and memory layout

- **Indexing** (API, same as SciPy and DifferentialEquations.jl): shape
  `(n, m)`; `y[i, :]` is component `i` over time, `y[:, k]` the state at
  `t[k]`.
- **Layout** (guaranteed; SciPy leaves it unspecified): column-major
  (`y.flags.f_contiguous`).  This is the natural layout end to end — the
  knots driver fills one column per knot, the adaptive driver appends a
  column per accepted step — and makes `y[:, k]` a contiguous view that
  can go back into Fortran/LAPACK without a copy.

In endpoint-only mode `y` is 1-D `(n,)` and layout does not arise.

### `status`, `success`, `message`

`status` uses SciPy semantics, not raw `ISTATE`.  `success` is defined as
`status >= 0`, **not** `status == 0` — success is a set of codes (the
lesson behind `SciMLBase.successful_retcode`), so future event
termination (`status == 1`) counts as success without callers updating.
`status` is a plain `int`; promoting it to an `IntEnum` later (symbolic
names that still compare equal to the ints) is compatible and out of
scope.

`message` is for humans — code discriminates on `status`, never by
parsing `message`.  On success it is SciPy's generic text ("The solver
successfully reached the end of the integration interval.").  On failure
it gives the failure location plus the ZVODE condition text (the shared
`MESSAGES` table, also used by the class-based API):

| `ISTATE` | `status` | message detail |
|----------|----------|----------------|
| `2`      | `0`      | success text |
| `-1`     | `-1`     | Excess work done on this call. |
| `-2`     | `-1`     | Excess accuracy requested. |
| `-3`     | `-1`     | Illegal input detected. |
| `-4`     | `-1`     | Repeated error test failures. |
| `-5`     | `-1`     | Repeated convergence failures. |
| `-6`     | `-1`     | Error weight became zero during problem integration. |

No remedy hints are appended: the condition text plus the raw `ISTATE`
value is enough, and solver-specific advice (which would name
`solve_complex_ivp` parameters) does not belong in the message text shared
with the class-based API.  Raw `ISTATE` is not a field; it appears verbatim
inside `message`, which suffices for bug reports.

## Failure behaviour: raise, carrying the failed result

Unlike SciPy (which returns with `success=False`), `solve_complex_ivp`
keeps raising — code that forgets to check `success` gets a loud error
instead of silently consuming a truncated trajectory.  The bare
`RuntimeError` becomes `ZVODEError(RuntimeError)` with a `result`
attribute: the fully-populated `ZVODEResult` with `success=False`,
`status=-1`, the failure `message` (which is also the exception text),
the partial `t`/`y` trajectory, and all counters.

```python
try:
    sol = solve_complex_ivp(fun, tspan, y0)
except ZVODEError as exc:
    partial = exc.result    # success=False; plot partial.t, partial.y
```

Existing `except RuntimeError` handlers keep working.  An opt-in flag for
SciPy's return-instead-of-raise behaviour can be added later without
touching the result object.

## Reserved names

Reserved for future features, **not** added as dead `None` fields now
(zvode has no `dense_output`/`events` parameters, so omission matches
SciPy's "None if not requested" contract):

| Reserved | Future feature |
|----------|----------------|
| `sol`    | Dense output: callable interpolant, signature `sol(t, k=0)` |
| `t_events`, `y_events` | Event support (SciPy layout: lists of ndarrays) |
| `status == 1` | Event termination |

The `k` in `sol(t, k=0)` is the derivative order: ZVINDY computes k-th
derivatives natively (the `refine` path already uses it), so zvode can
offer what `scipy.integrate.OdeSolution` cannot, at no Fortran-level
cost.  Committing to the signature now means it never changes.  Making
the result itself callable (DiffEq-style) is possible sugar later.

## Anti-goals

Deliberate non-features:

- **Plain data only** — no references to `fun`/`jac`/`ctx` or workspace
  arrays; picklable as-is.  (DifferentialEquations.jl stores the problem
  on its solution and needs `strip_solution` to serialize — avoid that.)
- **No array interface** — slicing lives on `result.y`, not on the result.
- **No raw solver state** — no `ISTATE`/`RWORK`/`IWORK`/Nordsieck fields;
  diffrax-style `solver_state` is a JAX artifact with no place here.

## Pretty-printing

The SciPy/MATLAB aligned `key: value` layout (what `solve_ivp` and MATLAB
users already see), but with MATLAB-style array placeholders instead of
SciPy's numpy-formatted array contents — arrays are summarised as
`[shape dtype]`, never dumped.  The exception is `t`: seeing the
integration interval is genuinely useful (SciPy, DifferentialEquations.jl,
and diffrax all print the time values; only MATLAB hides them), and since
`t` is monotonic its first and last entries convey the interval without a
dump:

```
 message: The solver successfully reached the end of the integration interval.
 success: True
  status: 0
       t: [58 float64] 0.0 to 6.2832
       y: [2x58 complex128]
    nfev: 131
    njev: 0
     nlu: 0
  nsteps: 57
     nni: 0
    ncfn: 0
    netf: 1
```

Rules: keys right-justified, verdict block (`message`/`success`/`status`)
first, then `t`, `y`, then counters; dict fields outside the canonical
order are appended so nothing goes missing; `t` carries the
`first to last` interval suffix; scalars render with `repr`;
`__repr__` and `__str__` are identical (the shell shows `__repr__`).  The
printout is *not* API — never parse it — so the rendering can change in
any release.

## Implementation notes

- `success`/`status`/`message` are built at the single point in
  `solve_complex_ivp` where the result dict is constructed; the failure
  path builds the same dict (truncated `t`/`y`) before raising
  `ZVODEError`.
- The class-based `ZVODE` stepper API is unaffected.

## Acceptance criteria

1. Successful solve: `success is True`, `status == 0`, generic success
   `message`.
2. Failing solve raises `ZVODEError`; `exc.result` has `success is False`,
   `status == -1`, a `message` containing the `ISTATE` detail, and the
   partial trajectory (closes the release-plan item on catchable failures).
3. `pickle.loads(pickle.dumps(result))` round-trips.
4. A `ZVODEResult` passes for an `OdeResult` in duck-typed code reading
   `t`, `y`, `success`, `status`, `message`, `nfev`, `njev`, `nlu`.
