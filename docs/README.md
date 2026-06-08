# Docs

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and
published to <https://ivan-pi.github.io/zvode/>.

To build locally:

```bash
pip install ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## How-to guides

The guides are written in reStructuredText and use Sphinx cross-references
to link to the API reference.

- **[`how-to-procedural-api.rst`](how-to-procedural-api.rst)** — output modes,
  method selection, dense and banded Jacobians, backward integration.
- **[`banded_jacobian.rst`](banded_jacobian.rst)** — compact column-oriented
  storage layout with worked example and memory-saving comparison table.
- **[`how-to-compiled-callbacks.rst`](how-to-compiled-callbacks.rst)** —
  compiled RHS and Jacobian callbacks via numba `@cfunc` or a shared library,
  including the `ctx` parameter for passing parameters without globals.

## API reference

[`api.rst`](api.rst) is the Sphinx autodoc entry point.  It pulls docstrings
from the installed package, so the package must be installed (e.g.
`pip install -e .`) before building the docs.

## Examples (OdeSolver API)

The scripts below are self-contained examples that show how to use
`zvode` with [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
Each script can be run directly:

```bash
python docs/<script_name>.py
```

They require [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/).

1. **[`demo_minimal.py`](demo_minimal.py)** — Minimal working example.
   Scalar complex decay `y' = -i·y` solved with a single `solve_ivp` call.

2. **[`demo_linear.py`](demo_linear.py)** — Non-autonomous scalar ODE `y' = t·y + 2i`
   with a known analytic solution; demonstrates accuracy checking against `scipy.special.erf`.

3. **[`demo_scipy_ode_example.py`](demo_scipy_ode_example.py)** — Two coupled ODEs.
   Reproduces the legacy `scipy.integrate.ode` step-by-step integration loop and
   cross-checks it against `solve_ivp` via dense output.

4. **[`demo_zvode_source.py`](demo_zvode_source.py)** — Two-ODE complex system from the
   original ZVODE Fortran source. Uses a user-supplied Jacobian and demonstrates the
   `args` parameter of `solve_ivp`.

5. **[`demo_complex_linear_system.py`](demo_complex_linear_system.py)** — Complex linear
   system `y' = A·y` solved both as a 3-component vector IVP and as a flattened 3×3
   matrix IVP. Illustrates the Kronecker-product Jacobian trick and validation via
   `scipy.linalg.expm`.

6. **[`demo_qme.py`](demo_qme.py)** — Lindblad master equation for a driven, dissipative
   two-level quantum system (qubit) undergoing Rabi oscillations with spontaneous emission.
   The 2×2 density matrix is vectorized to a 4-component complex array to interface with
   `solve_ivp`.

---

Have you solved an interesting ODE system with `zvode` and are willing to share it?
Contributions are welcome — open an issue or pull request with your example script.
A short note on what makes the problem interesting (stiffness, physics context, unusual
structure) helps others learn from it.
