# zvode

Python bindings to the classic ZVODE ODE solver.

[![Tests](https://github.com/ivan-pi/zvode/actions/workflows/test.yml/badge.svg)](https://github.com/ivan-pi/zvode/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/zvode)](https://pypi.org/project/zvode/)
[![Python](https://img.shields.io/pypi/pyversions/zvode)](https://pypi.org/project/zvode/)
[![License](https://img.shields.io/github/license/ivan-pi/zvode)](https://github.com/ivan-pi/zvode/blob/main/LICENSE)

[Source on GitHub](https://github.com/ivan-pi/zvode)

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state, written
by P. N. Brown, G. D. Byrne, and A. C. Hindmarsh [2]. It is part of ODEPACK
and uses a fixed-leading-coefficient Adams or BDF method, selectable by the
user.

This package exposes two interfaces to ZVODE:

- **Procedural API** — `solve_complex_ivp(fun, tspan, y0, ...)`: a single-call
  function in the spirit of {func}`scipy.integrate.odeint`. This is the recommended
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

### Procedural API — `solve_complex_ivp`

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

### OdeSolver API (scipy-compatible)

Pass a `ZVODE_*` class as the `method` argument to {func}`scipy.integrate.solve_ivp`.

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
{class}`scipy.integrate.OdeSolver` and require SciPy. If your code only uses
`solve_complex_ivp` you do not need SciPy.

---

## Limitations

- Complex floats (fp64) only
- No event-handling/root-finding capabilities
- Not thread-safe (ZVODE uses global Fortran COMMON blocks)
- No solution back-tracking available
- Only dense or banded Jacobians
- No built-in mass matrix support (a constant mass matrix can be handled by pre-factoring with LU decomposition)

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

For a broader perspective on the history and design philosophy behind ODEPACK and related solvers, see the
[SIAM oral history interview with Alan C. Hindmarsh](https://history.siam.org/oralhistories/hindmarsh.htm).

---

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
how-to-compiled-callbacks
:::

:::{toctree}
:maxdepth: 1
:caption: Internals

compiled-callbacks-design
:::
