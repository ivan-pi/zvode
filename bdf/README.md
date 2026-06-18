# pybdf — a small BDF integrator (experimental)

A compact Fortran port of [`scipy.integrate.BDF`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html):
the variable-order (1–5) NDF-enhanced backward differentiation formulas of
Shampine & Reichelt, for stiff systems of first-order ODEs.

This lives alongside the `zvode` project as an experiment in generating a
Fortran integrator from a SciPy reference, organised as a three-level stack:

```
src/fortran/bdf.f90    Fortran core   — stateful bdf_solver, owns its workspace
src/fortran/c_bdf.f90  C ABI           — opaque-handle wrapper (see bdf.h)
src/_bdf.c             CPython binding — marshals Python callbacks
src/pybdf/             Python package  — friendly NumPy interface
```

## Scope

Deliberately narrow, matching the task it was built for:

* real, double-precision state only;
* **dense or banded** Jacobians;
* **user-supplied or finite-difference** Jacobians;
* **LAPACK** (`dgetrf`/`dgetrs`, `dgbtrf`/`dgbtrs`) for the linear algebra;
* **Jacobian saving** — the Jacobian is cached and reused between steps,
  refactoring only the `I − c·J` matrix (toggle with `reuse_jac`);
* no dense output / interpolation — output times are reached by stepping
  exactly onto them.

The integrator is a **stateful object** whose workspace is allocated once,
dynamically, from the problem size and reused for the whole run.

## Quick start

```python
import numpy as np
from pybdf import solve_bdf

# Robertson stiff kinetics.
def f(t, y):
    return [-0.04*y[0] + 1e4*y[1]*y[2],
             0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]**2,
             3e7*y[1]**2]

def jac(t, y):
    return np.array([[-0.04,  1e4*y[2],             1e4*y[1]],
                     [ 0.04, -1e4*y[2] - 6e7*y[1], -1e4*y[1]],
                     [ 0.0,   6e7*y[1],             0.0]])

sol = solve_bdf(f, (0, 40), [1, 0, 0], t_eval=[40],
                rtol=1e-6, atol=[1e-8, 1e-10, 1e-8], jac=jac)
print(sol.y[:, -1])          # ~ [0.7158, 9.19e-6, 0.2842]
print(sol.nfev, sol.njev, sol.nlu, sol.nsteps)
```

A **banded** Jacobian is selected with `band=(ml, mu)`; pass `jac=None`
(the default) for finite differences, a callable for an analytic Jacobian,
or an array for a constant one:

```python
sol = solve_bdf(rhs, (0, 1), y0, band=(1, 1), rtol=1e-8, atol=1e-10)
```

For step-by-step control use the `BDF` class directly:

```python
from pybdf import BDF
s = BDF(f, t0=0.0, y0=[1, 0, 0], t_bound=40.0, jac=jac)
while s.status == "running":
    s.step()
print(s.t, s.y)
```

## Building and testing

The Python package builds with scikit-build-core:

```bash
pip install .            # build the extension and install pybdf
pip install ".[test]"    # also pull in pytest + scipy
pytest test/test_pybdf.py
```

The Fortran core can be built and tested on its own (no Python required):

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## References

1. L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", *SIAM J. Sci.
   Comput.*, 18(1), 1997.
2. E. Hairer, G. Wanner, *Solving Ordinary Differential Equations I*.
