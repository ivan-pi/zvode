# zvode

Python bindings to the classic ZVODE ODE solver.

[![Tests](https://github.com/ivan-pi/zvode/actions/workflows/test.yml/badge.svg)](https://github.com/ivan-pi/zvode/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/zvode)](https://pypi.org/project/zvode/)
[![Python](https://img.shields.io/pypi/pyversions/zvode)](https://pypi.org/project/zvode/)
[![License](https://img.shields.io/github/license/ivan-pi/zvode)](https://github.com/ivan-pi/zvode/blob/main/LICENSE)

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state. It is
part of ODEPACK and uses a fixed-leading-coefficient Adams or BDF method
depending on the problem type.

This package wraps ZVODE as a [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html) subclass,
so it can be passed directly to `scipy.integrate.solve_ivp` via the `method`
argument.

> **Warning** — This integrator is not re-entrant. You cannot have two `ode`
> instances using the `"zvode"` integrator at the same time.

## Differences from upstream ZVODE

This package ships a **modified** copy of ZVODE (see [`extern/zvode.f`](extern/zvode.f)).
The [Netlib source](https://netlib.org/ode/zvode.f) has been changed in the following ways:

- **Module wrapping** — the entire code is enclosed in a Fortran 90 `MODULE ZVODE_MOD`
  with `IMPLICIT NONE` and explicit `PUBLIC`/`PRIVATE` declarations.
- **Abstract class callbacks** — the right-hand side function `F` and the Jacobian `JAC`
  were external subroutines in the original; here they are polymorphic `CLASS(ZVODE_FUN)`
  and `CLASS(ZVODE_JAC)` objects with deferred `EVAL` procedures (Fortran 2003 abstract
  classes). This is the most significant departure from the original design.
- **`RPAR`/`IPAR` removed** — the original interface passes user context through a real/complex
  array `RPAR` and an integer array `IPAR`. In the functor design, context is carried by the
  class object itself, so these arguments are no longer present.

The internal numerics — the `ZVOD01`/`ZVOD02` Fortran `COMMON` blocks, the Adams and BDF
stepping logic, and the vendored LINPACK routines (`ZGEFA`, `ZGESL`, `ZGBFA`, `ZGBSL`) —
are **unchanged** from the 2006 LLNL release.

## Installation

```bash
pip install zvode
```

## Quick start

**Non-stiff problem** — rotating complex exponential:

```python
import numpy as np
from scipy.integrate import solve_ivp
from zvode import ZVODE

sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    method=ZVODE,
    method_options=dict(zvode_method='Adams'),
)
```

**Stiff problem** — with a user-supplied Jacobian:

```python
sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=np.array([1.0 + 0.0j]),
    method=ZVODE,
    jac=lambda t, y: np.array([[-1j]]),
    method_options=dict(zvode_method='BDF'),
)
```

## Solver options

Pass these via `method_options=dict(...)` in `solve_ivp`, or as keyword
arguments when constructing `ZVODE` directly.

| Option | Type | Default | Description |
|---|---|---|---|
| `zvode_method` | `'BDF'` or `'Adams'` | `'BDF'` | Integration method. BDF (max order 5) for stiff problems; Adams (max order 12) for non-stiff. |
| `rtol` | float or array | `1e-3` | Relative error tolerance, per component or global. |
| `atol` | float or array | `1e-6` | Absolute error tolerance, per component or global. |
| `jac` | callable or None | `None` | Jacobian `jac(t, y)` → `(n, n)` array. Estimated by finite differences if not provided. |
| `lband`, `uband` | int or None | `None` | Lower/upper half-bandwidths for a banded Jacobian. |
| `max_order` | int | `5` / `12` | Maximum integration order (capped by method). |
| `first_step` | float | auto | Initial step size. |
| `max_step` | float | `np.inf` | Maximum step size. |
| `jsv` | `1` or `-1` | `1` | `1` saves and reuses the Jacobian; `-1` recomputes every step. |

> **Note** — For stiff problems, `f` must be analytic (each component must be
> an analytic function of each state variable). For stiff systems where `f` is
> not analytic, use a real-valued solver on the equivalent doubled real system.

## Limitations

- complex floats only
- no event-handling/root-finding capabilities
- not thread-safe (ZVODE uses global Fortran COMMON blocks)
- no solution back-tracking available

## References

<a id="1">[1]</a>
A. C. Hindmarsh,
"ODEPACK, A Systematized Collection of ODE Solvers,"
in *Scientific Computing*, R. S. Stepleman et al. (eds.),
North-Holland, Amsterdam, 1983 (vol. 1 of IMACS Transactions on Scientific Computation), pp. 55–64.
https://computing.llnl.gov/projects/odepack

<a id="2">[2]</a>
P. N. Brown, G. D. Byrne, and A. C. Hindmarsh,
"VODE, A Variable-Coefficient ODE Solver,"
*SIAM J. Sci. Stat. Comput.*, 10 (1989), pp. 1038–1051.
https://doi.org/10.1137/0910062

<a id="3">[3]</a>
G. D. Byrne and A. C. Hindmarsh,
"A Polyalgorithm for the Numerical Solution of Ordinary Differential Equations,"
*ACM Trans. Math. Soft.*, 1(1), pp. 71–96, 1975.
https://doi.org/10.1145/355626.355636

For a broader perspective on the history and design philosophy behind ODEPACK and related solvers, see the
[SIAM oral history interview with Alan C. Hindmarsh](https://history.siam.org/oralhistories/hindmarsh.htm).

## Links

### ZVODE upstream

| Resource | URL |
|---|---|
| Official ODEPACK page — Lawrence Livermore National Laboratory | <https://computing.llnl.gov/projects/odepack> |
| Source on Netlib | <https://netlib.org/ode/zvode.f> |
| Sandia Netlib mirror | <https://netlib.sandia.gov/ode/zvode.f> |

### Python / R ecosystem

| Resource | URL |
|---|---|
| `scipy.integrate.OdeSolver` (base class) | <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html> |
| R wrappers — deSolve `zvode` | <https://www.rdocumentation.org/packages/deSolve/versions/1.42/topics/zvode> |

## License

`zvode` is distributed under the BSD license. See [LICENSE](LICENSE) for details.

## Contributing

Bug reports and suggestions are welcome via the [issue tracker](https://github.com/ivan-pi/zvode/issues).
The most useful reports are:

- **Documentation errors** — typos, incorrect parameter descriptions, or misleading examples.
- **Integration failures** — cases where the solver returns a wrong result, fails to converge,
  or raises an unexpected error. A minimal reproducer (ODE, initial condition, tolerances) is
  very helpful.
- **Feature requests** — even if a feature is not planned, requests help track what practitioners
  actually need.
