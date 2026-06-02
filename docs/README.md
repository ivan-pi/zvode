# Examples

The scripts in this folder are self-contained examples that show how to use
`zvode` with [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
Each script can be run directly:

```bash
python docs/<script_name>.py
```

1. **[`demo_minimal.py`](demo_minimal.py)** — Minimal working example.
   Scalar complex decay `y' = -i·y` solved with a single `solve_ivp` call.

2. **[`demo_linear.py`](demo_linear.py)** — Non-autonomous scalar ODE `y' = t·y + 2i`
   with a known analytic solution; demonstrates accuracy checking against `scipy.special.erf`.
   Problem taken from the [MATLAB ODE solver documentation](https://www.mathworks.com/help/matlab/math/choose-an-ode-solver.html#bu8f_6x).

3. **[`demo_scipy_ode_example.py`](demo_scipy_ode_example.py)** — Two coupled ODEs taken from
   the [`scipy.integrate.ode` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html).
   Reproduces the legacy step-by-step integration loop and cross-checks it against
   `solve_ivp` via dense output.

4. **[`demo_zvode_source.py`](demo_zvode_source.py)** — Two-ODE complex system taken directly
   from the original ZVODE Fortran source (`zvode.f`). Uses a user-supplied Jacobian and
   demonstrates the `args` parameter of `solve_ivp`.

5. **[`demo_complex_linear_system.py`](demo_complex_linear_system.py)** — Complex linear
   system `y' = A·y` adapted from the [`scipy.integrate.solve_ivp` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
   solved both as a 3-component vector IVP and as a flattened 3×3 matrix IVP.
   Illustrates the Kronecker-product Jacobian trick and validation via `scipy.linalg.expm`.

6. **[`demo_qme.py`](demo_qme.py)** — Lindblad master equation for a driven, dissipative
   two-level quantum system (qubit) undergoing Rabi oscillations with spontaneous emission.
   The 2×2 density matrix is vectorized to a 4-component complex array to interface with
   `solve_ivp`.

---

Have you solved an interesting ODE system with `zvode` and are willing to share it?
Contributions are welcome — open an issue or pull request with your example script.
A short note on what makes the problem interesting (stiffness, physics context, unusual
structure) helps others learn from it.
