# zvode

Python bindings to the classic ZVODE ODE solver.

[![Tests](https://github.com/ivan-pi/zvode/actions/workflows/test.yml/badge.svg)](https://github.com/ivan-pi/zvode/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/zvode)](https://pypi.org/project/zvode/)
[![Python](https://img.shields.io/pypi/pyversions/zvode)](https://pypi.org/project/zvode/)
[![License](https://img.shields.io/github/license/ivan-pi/zvode)](https://github.com/ivan-pi/zvode/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-ivan--pi.github.io%2Fzvode-blue)](https://ivan-pi.github.io/zvode/)

ZVODE is a variable-coefficient ODE solver for stiff and non-stiff systems of
first-order ordinary differential equations with complex-valued state, written
by P. N. Brown, G. D. Byrne, and A. C. Hindmarsh [[2]](#2). It is part of ODEPACK and uses
a fixed-leading-coefficient Adams or BDF method, selectable by the user.

This package exposes two interfaces to ZVODE:

- **Procedural API** — `solve_complex_ivp(fun, tspan, y0, ...)`: a single-call
  function in the spirit of `scipy.integrate.odeint`. This is the recommended
  starting point.
- **OdeSolver API** — `ZVODE` / `ZVODE_BDF` / `ZVODE_Adams`: a
  [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html) subclass for use with
  [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

The underlying Fortran source has been modified; [`extern/README.md`](https://github.com/ivan-pi/zvode/blob/main/extern/README.md) documents the changes.

> **Warning:** This integrator is currently not thread-safe. You cannot have two threads
> using the ZVODE integrator simultaneously.

**Table of Contents:**

- [Installation](#installation)
- [Quick start](#quick-start)
- [Limitations](#limitations)
- [Links](#links)
- [Building from source](#building-from-source)
- [Changelog](#changelog)
- [License](#license)
- [Contributing](#contributing)
- [References](#references)

## Installation

```bash
pip install zvode          # procedural API only (numpy only)
pip install zvode[scipy]   # also enables ZVODE / ZVODE_BDF / ZVODE_Adams (requires SciPy)
```

The OdeSolver classes (`ZVODE`, `ZVODE_BDF`, `ZVODE_Adams`) are a SciPy
add-on: they subclass `scipy.integrate.OdeSolver` so they can be passed
as the `method` argument to `scipy.integrate.solve_ivp`.  If your code
only uses `solve_complex_ivp` you do not need SciPy.

To install locally from source:

```bash
pip install .              # procedural API only
pip install ".[scipy]"     # also install SciPy
pip install ".[test]"      # run the test suite (includes SciPy)
```

## Quick start

### Procedural API — `solve_complex_ivp`

`solve_complex_ivp` is the recommended entry point. Pass the RHS function, a time span,
and an initial condition; get back a result object with `sol.t`, `sol.y`, and
integration statistics (`sol.nfev`, `sol.njev`, …) as attributes.

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

See the [documentation](https://ivan-pi.github.io/zvode/) for output modes,
banded Jacobians, backward integration, compiled callbacks, and other options.

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

More OdeSolver examples are in the [`docs/`](https://github.com/ivan-pi/zvode/tree/main/docs) folder on GitHub.

## Limitations

- complex floats (fp64) only
- no event-handling/root-finding capabilities
- not thread-safe (ZVODE uses global Fortran COMMON blocks)
- no solution back-tracking available
- only dense or banded Jacobians
- no built-in mass matrix support (a constant mass matrix can be handled by pre-factoring with LU decomposition)

## Links

### ODEPACK & SUNDIALS

- [ODEPACK](https://computing.llnl.gov/projects/odepack)
- [Netlib](https://netlib.org/ode/zvode.f) ([Sandia mirror](https://netlib.sandia.gov/ode/zvode.f))
- [SUNDIALS](https://computing.llnl.gov/projects/sundials)

### Python / R ecosystem

- [`scipy.integrate.OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html) — base class used by the OdeSolver API
- [`scipy.integrate.ode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) — legacy stateful ZVODE wrapper
- [deSolve `zvode`](https://www.rdocumentation.org/packages/deSolve/versions/1.42/topics/zvode) — R wrapper

SciPy has historically provided a ZVODE wrapper through
[`scipy.integrate.ode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html),
a stateful, class-based interface (`integrator='zvode'`). As of SciPy 1.17,
the underlying Fortran source was [replaced](https://github.com/scipy/scipy/pull/23963) with a C translation of ZVODE
that is thread-safe.

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

### Debug build

Passing `-DZVODE_DEBUG` (equivalent to `-DZVODE_DEBUG=1`) enables extra
assertions and diagnostic output in the C extension:

```bash
pip install -v ".[test]" \
  -C "cmake.args=-DCMAKE_C_FLAGS=-DZVODE_DEBUG"
```

## Changelog

See [CHANGELOG.md](https://github.com/ivan-pi/zvode/blob/main/CHANGELOG.md) for version history.

## License

`zvode` is distributed under the BSD license. See [LICENSE](https://github.com/ivan-pi/zvode/blob/main/LICENSE) for details.

## Contributing

Bug reports and suggestions are welcome via the [issue tracker](https://github.com/ivan-pi/zvode/issues).
The most useful reports are:

- **Documentation errors** — typos, incorrect parameter descriptions, or misleading examples.
- **Integration failures** — cases where the solver returns a wrong result, fails to converge,
  or raises an unexpected error. A minimal reproducer (ODE, initial condition, tolerances) is
  very helpful.
- **Feature requests** — even if a feature is not planned, requests help track what practitioners
  actually need.

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
