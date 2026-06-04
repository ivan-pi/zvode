# How to use `solve_complex_ivp`

`solve_complex_ivp` is the procedural interface to ZVODE. Pass the RHS, a time
span, and an initial condition; get back a `ZVODEResult` object with `sol.t`,
`sol.y`, and integration statistics as attributes (also accessible as dict
keys).

```python
import numpy as np
from zvode import solve_complex_ivp

sol = solve_complex_ivp(fun, tspan, y0, **options)
# sol.t    — 1-D float array of output times, shape (m,)
# sol.y    — complex array, shape (n, m)
# sol.nfev — RHS evaluations (int)
# sol.njev — Jacobian evaluations (int)
# sol.nlu  — LU decompositions (int)
```

The full parameter list is in the `solve_complex_ivp` docstring (`help(solve_complex_ivp)`).

---

## Output modes

**Collect every accepted step** (default) — pass a 2-element `tspan`:

```python
sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j])
# sol.t[0] == 0.0, sol.t[-1] == 10.0; number of points depends on tolerances
```

**Output at specific times** — pass three or more values as `tspan`:

```python
t_out = np.linspace(0.0, 10.0, 101)
sol = solve_complex_ivp(rhs, tspan=t_out, y0=[1.0 + 0j])
# sol.t is t_out, sol.y.shape == (1, 101)
```

**Endpoint only** — `save_steps=False` returns a scalar `sol.t` and a 1-D `sol.y`:

```python
sol = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0j], save_steps=False)
# float(sol.t), sol.y.shape == (1,)
```

---

## Choosing a method

`method='BDF'` (default) is for stiff problems; `method='Adams'` for non-stiff.

> **Analyticity note** — BDF uses complex Newton iteration, which requires every
> component of `f(t, y)` to be an analytic function of the complex state. If
> your RHS uses `abs`, `conj`, or real/imaginary-part splitting, use a real-valued
> solver on the equivalent doubled real system instead.

---

## Providing a Jacobian

Supplying the Jacobian avoids finite differences and reduces RHS evaluations for
stiff problems.

**Dense** — return an `(n, n)` array from `jac(t, y)`:

```python
def rhs(t, y):
    return np.array([-100j * y[0] + y[1], -1j * y[1]])

def jac(t, y):
    return np.array([[-100j, 1.0], [0.0, -1j]])

sol = solve_complex_ivp(rhs, tspan=(0.0, 5.0),
                        y0=[1.0 + 0j, 0.0 + 1j],
                        method='BDF', jac=jac)
```

**Banded** — pass `lband` and `uband`; return a `(lband + uband + 1, n)` array
where entry `[uband + i - j, j]` holds `df[i]/dy[j]`:

```python
sol = solve_complex_ivp(rhs, tspan=(0.0, 5.0), y0=y0,
                        method='BDF', jac=jac_banded,
                        lband=1, uband=1)
```

---

## Backward integration

Set `tspan` in decreasing order; for knot mode pass times strictly decreasing:

```python
sol = solve_complex_ivp(rhs, tspan=(10.0, 0.0), y0=y_at_t10)
```

---

## In-place callbacks

Set `in_place=True` to avoid a NumPy allocation per step. The RHS signature
becomes `fun(t, y, dy)` (fills `dy` in place); the Jacobian signature becomes
`jac(t, y, pd)` for dense or `jac(t, y, pd, ml, mu)` for banded, where `pd`,
the array of partial derivatives, is modified in place.

