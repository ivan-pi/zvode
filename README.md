# zvode

Python bindings to the classic ZVODE ODE solver.

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state. It is
part of ODEPACK and uses a fixed-leading-coefficient Adams or BDF method
depending on the problem type.

This package wraps ZVODE as a [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html) subclass,
so it can be passed directly to `scipy.integrate.solve_ivp` via the `method`
argument.

> **Warning** — This integrator is not re-entrant. You cannot have two `ode`
> instances using the `"zvode"` integrator at the same time.
## Installation

```bash
pip install zvode
```

## Quick start

```python
import numpy as np
from scipy.integrate import solve_ivp
from zvode import ZVODE

sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    method=ZVODE,
)

print(sol)
```

## Limitations

- complex floats only
- no event-handling/root-finding capabilities
- not thread-safe (ZVODE uses global data)
- no solution back-tracking available

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

