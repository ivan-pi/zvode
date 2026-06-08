# zvode

Python bindings to the classic ZVODE ODE solver.

[![Tests](https://github.com/ivan-pi/zvode/actions/workflows/test.yml/badge.svg)](https://github.com/ivan-pi/zvode/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/zvode)](https://pypi.org/project/zvode/)
[![Python](https://img.shields.io/pypi/pyversions/zvode)](https://pypi.org/project/zvode/)
[![License](https://img.shields.io/github/license/ivan-pi/zvode)](https://github.com/ivan-pi/zvode/blob/main/LICENSE)

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state, written
by P. N. Brown, G. D. Byrne, and A. C. Hindmarsh [2]. It is
part of ODEPACK and uses a fixed-leading-coefficient Adams or BDF method,
selectable by the user.

This package exposes two interfaces to ZVODE:

- **Procedural API** — `solve_complex_ivp(fun, tspan, y0, ...)`: a single-call
  function in the spirit of `scipy.integrate.odeint`. This is the recommended
  starting point.
- **OdeSolver API** — `ZVODE` / `ZVODE_BDF` / `ZVODE_Adams`: a
  [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html)
  subclass for use with
  [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

:::{warning}
This integrator is not thread-safe. You cannot have two threads using the ZVODE
integrator simultaneously.
:::

---

## Quick start

### Procedural API — `solve_complex_ivp` (new in 0.2.0)

`solve_complex_ivp` is the recommended entry point. Pass the RHS function, a
time span, and an initial condition; get back a result object with `sol.t`,
`sol.y`, and integration statistics (`sol.nfev`, `sol.njev`, …) as attributes.

```python
from zvode import solve_complex_ivp

def rhs(t, y):
    return [-100j * y[0] + y[1], -1j * y[1]]

def jac(t, y):
    return [[-100j, 1.0], [0.0, -1j]]

sol = solve_complex_ivp(
    fun=rhs,
    tspan=(0.0, 5.0),
    y0=[1.0 + 0j, 0.0 + 1j],
    method='BDF',
    jac=jac,
)

print(sol)
```

See {doc}`how-to-procedural-api` for output modes, banded Jacobians, backward
integration, and other options.

### OdeSolver API (scipy-compatible)

Pass a `ZVODE_*` class as the `method` argument to `scipy.integrate.solve_ivp`.

**Non-stiff problem** — rotating complex exponential:

```python
import numpy as np
from scipy.integrate import solve_ivp
from zvode import ZVODE_Adams

sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    method=ZVODE_Adams,
)
```

**Stiff problem** — with a user-supplied Jacobian:

```python
from zvode import ZVODE_BDF

sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    method=ZVODE_BDF,
    jac=lambda t, y: [[-1j]],
)
```

---

## Installation

```bash
pip install zvode          # procedural API only (numpy only)
pip install zvode[scipy]   # also enables ZVODE / ZVODE_BDF / ZVODE_Adams
```

The OdeSolver classes (`ZVODE`, `ZVODE_BDF`, `ZVODE_Adams`) subclass
`scipy.integrate.OdeSolver` and require SciPy. If your code only uses
`solve_complex_ivp` you do not need SciPy.

---

## Solver options

### `solve_complex_ivp` key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fun` | callable | — | RHS `f(t, y) → array_like`. With `in_place=True`: `f(t, y, dy)` fills `dy` in place. |
| `tspan` | array-like | — | `(t0, tf)` collects every accepted step; three or more values output at exactly those times; `(t0, tf)` with `save_steps=False` returns only the endpoint. |
| `y0` | array-like | — | Initial state; cast to `complex128`. |
| `method` | `'BDF'` or `'Adams'` | `'BDF'` | BDF (max order 5) for stiff problems; Adams (max order 12) for non-stiff. |
| `rtol` | float or array | `1e-3` | Relative error tolerance, per component or global. |
| `atol` | float or array | `1e-6` | Absolute error tolerance, per component or global. |
| `jac` | callable or None | `None` | Jacobian `jac(t, y)`. Dense: return `(n, n)`; banded: return `(lband + uband + 1, n)`. Estimated by finite differences if omitted. |
| `lband`, `uband` | int or None | `None` | Lower/upper half-bandwidths; activates the banded solver path. |
| `save_steps` | bool | `True` | Collect every accepted step (`True`) or return only the endpoint (`False`). Ignored when `tspan` has three or more elements. |

### OdeSolver API options

Keyword arguments accepted by `ZVODE` / `ZVODE_BDF` / `ZVODE_Adams`; passed
through unchanged when supplied via `solve_ivp`.

| Option | Type | Default | Description |
|---|---|---|---|
| `lmm` | `'BDF'` or `'Adams'` | `'BDF'` | Linear multistep method. BDF (max order 5) for stiff problems; Adams (max order 12) for non-stiff. |
| `rtol` | float or array | `1e-3` | Relative error tolerance, per component or global. |
| `atol` | float or array | `1e-6` | Absolute error tolerance, per component or global. |
| `jac` | callable or None | `None` | Jacobian `jac(t, y)`. Dense: `(n, n)` array; banded: `(lband + uband + 1, n)` array. |
| `lband`, `uband` | int or None | `None` | Lower/upper half-bandwidths; activates the banded solver path. |

:::{note}
For stiff problems, `f` must be analytic (each component must be an analytic
function of each state variable). For stiff systems where `f` is not analytic,
use a real-valued solver on the equivalent doubled real system.
:::

---

## Limitations

- Complex floats (fp64) only
- No event-handling/root-finding capabilities
- Not thread-safe (ZVODE uses global Fortran COMMON blocks)
- No solution back-tracking available
- Only dense or banded Jacobians

---

## References

[1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE Solvers," in
*Scientific Computing*, R. S. Stepleman et al. (eds.), North-Holland,
Amsterdam, 1983 (vol. 1 of IMACS Transactions on Scientific Computation),
pp. 55–64. <https://computing.llnl.gov/projects/odepack>

[2] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh, "VODE, A
Variable-Coefficient ODE Solver," *SIAM J. Sci. Stat. Comput.*, 10 (1989),
pp. 1038–1051. <https://doi.org/10.1137/0910062>

[3] G. D. Byrne and A. C. Hindmarsh, "A Polyalgorithm for the Numerical
Solution of Ordinary Differential Equations," *ACM Trans. Math. Soft.*,
1(1), pp. 71–96, 1975. <https://doi.org/10.1145/355626.355636>

---

## Contents

:::{toctree}
:maxdepth: 1
:caption: API reference

api
:::

:::{toctree}
:maxdepth: 1
:caption: How-to guides

how-to-procedural-api
banded_jacobian
:::

:::{toctree}
:maxdepth: 1
:caption: Internals

compiled-callbacks-design
:::
