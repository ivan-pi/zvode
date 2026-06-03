# zvode

Python bindings to the classic ZVODE ODE solver.

[![Tests](https://github.com/ivan-pi/zvode/actions/workflows/test.yml/badge.svg)](https://github.com/ivan-pi/zvode/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/zvode)](https://pypi.org/project/zvode/)
[![Python](https://img.shields.io/pypi/pyversions/zvode)](https://pypi.org/project/zvode/)
[![License](https://img.shields.io/github/license/ivan-pi/zvode)](https://github.com/ivan-pi/zvode/blob/main/LICENSE)

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state, written
by G. D. Byrne and A. C. Hindmarsh [[2]](#2). It is part of ODEPACK and uses
a fixed-leading-coefficient Adams or BDF method, selectable by the user.

This package exposes two interfaces to ZVODE:

- **Procedural API** — `solve_complex_ivp(fun, tspan, y0, ...)`: a single-call
  function in the spirit of `scipy.integrate.odeint`. This is the recommended
  starting point and the headline feature of **version 0.2.0**.
- **OdeSolver API** — `ZVODE` / `ZVODE_BDF` / `ZVODE_Adams`: a
  [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html) subclass for use with
  [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

It uses a modified version of the original Fortran package; see
[`extern/README.md`](extern/README.md) for the changes made.

> [!WARNING]
> This integrator is not thread-safe. You cannot have two threads
> using the ZVODE integrator simultaneously.

## Quick start

### Procedural API — `solve_complex_ivp` (new in 0.2.0)

`solve_complex_ivp` is the recommended entry point. Pass the RHS, a time span,
and an initial condition; get back a `ZVODEResult` with `sol.t`, `sol.y`, and
integration statistics (`sol.nfev`, `sol.njev`, …) as attributes.

```python
import numpy as np
from zvode import solve_complex_ivp

def rhs(t, y):
    return np.array([-100j * y[0] + y[1],
                     -1j   * y[1]])

def jac(t, y):
    return np.array([[-100j, 1.0],
                     [ 0.0, -1j]])

sol = solve_complex_ivp(
    fun=rhs,
    tspan=(0.0, 5.0),
    y0=[1.0 + 0j, 0.0 + 1j],
    method='BDF',
    jac=jac,
)

print(sol.t.shape)   # (m,)   — every accepted step
print(sol.y.shape)   # (2, m)
print(sol.nfev)      # RHS evaluations
```

See [`docs/how-to-procedural-api.md`](docs/how-to-procedural-api.md) for output
modes, banded Jacobians, backward integration, and other options.

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
    y0=np.array([1.0 + 0.0j]),
    method=ZVODE_BDF,
    jac=lambda t, y: np.array([[-1j]]),
)
```

More OdeSolver examples are in the [`docs/`](docs/) folder.

## Installation

```bash
pip install zvode          # once available on PyPI
pip install .              # install locally from source
```

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
| `refine` | int | `1` | Insert `refine − 1` interpolated points between each pair of accepted steps using ZVINDY dense output. |
| `max_order` | int | `5` / `12` | Maximum integration order (capped by method). |
| `first_step` | float | auto | Initial step size. |
| `max_step` | float | `np.inf` | Maximum step size. |
| `min_step` | float | `0` | Minimum step size. |
| `max_num_steps` | int | `1 000 000` | Maximum internal steps between output points; raises `RuntimeError` if exceeded. |

### OdeSolver API options

Pass these as keyword arguments to `solve_ivp` (forwarded to the solver
constructor) or directly when constructing `ZVODE` / `ZVODE_BDF` /
`ZVODE_Adams`.

| Option | Type | Default | Description |
|---|---|---|---|
| `lmm` | `'BDF'` or `'Adams'` | `'BDF'` | Linear multistep method. BDF (max order 5) for stiff problems; Adams (max order 12) for non-stiff. Fixed by the `ZVODE_BDF` and `ZVODE_Adams` subclasses. |
| `rtol` | float or array | `1e-3` | Relative error tolerance, per component or global. |
| `atol` | float or array | `1e-6` | Absolute error tolerance, per component or global. |
| `jac` | callable or None | `None` | Jacobian `jac(t, y)`. For a full Jacobian return an `(n, n)` array; for a banded Jacobian return an `(lband + uband + 1, n)` array. Estimated by finite differences if not provided. |
| `lband`, `uband` | int or None | `None` | Lower/upper half-bandwidths of the Jacobian band. Setting either activates the banded solver path; the other defaults to 0. |
| `max_order` | int | `5` / `12` | Maximum integration order (capped by method). |
| `first_step` | float | auto | Initial step size. |
| `max_step` | float | `np.inf` | Maximum step size. |
| `min_step` | float | 0 | Minimum step-size. |
| `jsv` | `1` or `-1` | `1` | `1` saves and reuses the Jacobian; `-1` recomputes every step. |
| `miter` | int | None | Iteration method; normally inferred from  `jac` and `lband`/`uband`. |

> **Note** — For stiff problems, `f` must be analytic (each component must be
> an analytic function of each state variable). For stiff systems where `f` is
> not analytic, use a real-valued solver on the equivalent doubled real system.

## Limitations

- complex floats (fp64) only
- no event-handling/root-finding capabilities
- not thread-safe (ZVODE uses global Fortran COMMON blocks)
- no solution back-tracking available
- only dense or banded Jacobians

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
| ODEPACK | <https://computing.llnl.gov/projects/odepack> |
| Netlib mirror | <https://netlib.org/ode/zvode.f> |
| Netlib mirror (Sandia) | <https://netlib.sandia.gov/ode/zvode.f> |
| SUNDIALS | <https://computing.llnl.gov/projects/sundials> |

### Python / R ecosystem

| Resource | URL |
|---|---|
| `scipy.integrate.OdeSolver` (base class) | <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html> |
| R wrappers — deSolve `zvode` | <https://www.rdocumentation.org/packages/deSolve/versions/1.42/topics/zvode> |

## Building from source

Building requires a C compiler and a Fortran compiler with Fortran 2003 support.
The code has been tested with **gfortran** (passing `-std=f2003`).

A BLAS library is required at link time (located by CMake's
`find_package(BLAS)`). On Linux, [OpenBLAS](https://www.openblas.net/)
or a vendor BLAS (MKL, BLIS, …) should all work. On macOS, the system
Accelerate framework is picked up automatically.

```bash
# Example: Ubuntu / Debian
sudo apt update
sudo apt install gfortran libopenblas-dev
pip install -v ".[test]"
```

```bash
# Example: macOS (Homebrew)
brew update
brew install gfortran
pip install -v ".[test]"
```

To control which BLAS library is used, add the option,
```
pip install ... \
  -C "cmake.args=-DBLA_VENDOR=<blas_vendor>"
```
The list of BLAS/LAPACK vendors can be found [here](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors)

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
