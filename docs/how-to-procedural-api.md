# How to use `solve_complex_ivp`

`solve_complex_ivp` is the procedural interface to ZVODE — pass your RHS, a
time span, and an initial condition; get back arrays of times and states. It
works like `scipy.integrate.odeint` but targets complex-valued ODEs and gives
direct access to ZVODE-specific options such as banded Jacobians and step-size
controls.

This guide covers the most common usage patterns. All examples assume:

```python
import numpy as np
from zvode import solve_complex_ivp
```

---

## Return values

`solve_complex_ivp` always returns at least two values:

```python
t, y = solve_complex_ivp(fun, tspan, y0, ...)
```

- `t` — 1-D float array of output times, shape `(m,)`
- `y` — complex array, shape `(n, m)`, where `n = len(y0)`

The number of output points `m` and which times are returned depend on how
`tspan` is specified (see [Output modes](#output-modes) below).

---

## Minimal example

```python
def rhs(t, y):
    return -1j * y          # dy/dt = -i·y  →  y(t) = exp(-i·t)

t, y = solve_complex_ivp(rhs, tspan=(0.0, 2 * np.pi), y0=[1.0 + 0.0j])

print(t.shape)   # (m,)    — every accepted step
print(y.shape)   # (1, m)
print(abs(y[0, -1]))  # ≈ 1.0  (stays on the unit circle)
```

---

## Output modes

### Collect all accepted steps (default)

Pass a 2-element `tspan`; the solver returns every step it takes internally.
The number of output points depends on the problem and tolerances.

```python
t, y = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0.0j])
# len(t) varies; t[0] == 0.0, t[-1] == 10.0
```

### Output at specific times

Pass three or more values as `tspan`. The solver advances to each requested
time using ZVODE's internal interpolation (ITASK=1), so the output lands
exactly on the requested grid regardless of the solver's internal step size.

```python
t_out = np.linspace(0.0, 10.0, 101)
t, y = solve_complex_ivp(rhs, tspan=t_out, y0=[1.0 + 0.0j])
# t is t_out, y.shape == (1, 101)
```

This is usually the most convenient mode when downstream code needs a regular
grid (plotting, error analysis, comparison with a reference solution).

### Endpoint only

Set `save_steps=False` to integrate silently from `t0` to `tf` and return only
the final state. The return values are a scalar `t` and a 1-D `y` rather than
arrays, which is convenient when only the terminal value matters.

```python
t_final, y_final = solve_complex_ivp(
    rhs,
    tspan=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    save_steps=False,
)
# t_final is a float, y_final.shape == (1,)
```

### Dense interpolation with `refine`

Set `refine=k` to insert `k − 1` interpolated points between every pair of
consecutive accepted steps. The solver evaluates ZVINDY (the built-in
Nordsieck interpolant) to fill in the extra points at no extra RHS cost.

```python
t, y = solve_complex_ivp(
    rhs,
    tspan=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    refine=5,               # 4 extra points between each pair of steps
)
```

`refine` applies only in all-steps mode (2-element `tspan`). For a fixed
output grid, use the 3+-element `tspan` form instead.

---

## Choosing a method

| Method | Flag | Best for |
|--------|------|----------|
| BDF (default) | `method='BDF'` | Stiff systems; max order 5 |
| Adams | `method='Adams'` | Non-stiff systems; max order 12 |

Use BDF when your system has widely separated eigenvalues (stiff) or when
an Adams run requires an unexpectedly large number of steps.

```python
t, y = solve_complex_ivp(rhs, tspan=(0.0, 10.0), y0=[1.0 + 0.0j],
                         method='Adams')
```

> **Analyticity requirement** — The BDF path uses complex Newton iteration.
> For this to work, every component of `f(t, y)` must be an analytic function
> of the complex state variables. If your RHS involves `abs`, `conj`, or
> separately manipulates real and imaginary parts, the Jacobian is not
> complex-analytic and BDF will not converge correctly. In that case, rewrite
> the problem over the doubled real system and use a real-valued solver.

---

## Controlling tolerances

`rtol` and `atol` can be scalar (shared across all components) or 1-D arrays
of length `n` (per-component).

```python
t, y = solve_complex_ivp(
    rhs,
    tspan=(0.0, 10.0),
    y0=np.array([1.0 + 0.0j, 0.5 - 0.5j]),
    rtol=1e-8,
    atol=np.array([1e-10, 1e-12]),  # tighter tolerance on component 1
)
```

The local error target per component is approximately
`rtol * |y[i]| + atol[i]`.

---

## Providing a Jacobian

Supplying the Jacobian avoids finite-difference approximations and can
substantially reduce the number of RHS evaluations for stiff problems.

### Dense Jacobian

Return an `(n, n)` complex array from `jac(t, y)`:

```python
def rhs(t, y):
    return np.array([
        -100j * y[0] + y[1],
        -1j * y[1],
    ])

def jac(t, y):
    return np.array([
        [-100j, 1.0],
        [0.0,  -1j],
    ])

t, y = solve_complex_ivp(
    rhs,
    tspan=(0.0, 5.0),
    y0=[1.0 + 0j, 0.0 + 1j],
    method='BDF',
    jac=jac,
)
```

### Banded Jacobian

For large sparse systems whose non-zeros lie in a band, pass `lband` and
`uband` (lower and upper half-bandwidths) and return a
`(lband + uband + 1, n)` array packed in LAPACK banded format:
`pd[uband + i - j, j] = J[i, j]`.

```python
# 3-component tridiagonal system  (lband=1, uband=1)
n = 3
lband, uband = 1, 1

def rhs(t, y):
    dy = np.zeros(n, dtype=complex)
    dy[0] = -2j * y[0] + 1j * y[1]
    for k in range(1, n - 1):
        dy[k] = 1j * y[k - 1] - 2j * y[k] + 1j * y[k + 1]
    dy[-1] = 1j * y[-2] - 2j * y[-1]
    return dy

def jac_banded(t, y):
    pd = np.zeros((lband + uband + 1, n), dtype=complex)
    for j in range(n):
        pd[uband, j] = -2j                    # main diagonal: J[j,j]
        if j + 1 < n:
            pd[uband - 1, j + 1] = 1j        # super-diagonal: J[j, j+1]
        if j - 1 >= 0:
            pd[uband + 1, j - 1] = 1j        # sub-diagonal: J[j, j-1]
    return pd

y0 = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j])
t, y = solve_complex_ivp(
    rhs,
    tspan=(0.0, 5.0),
    y0=y0,
    method='BDF',
    jac=jac_banded,
    lband=lband,
    uband=uband,
)
```

Setting `lband` or `uband` tells the solver to allocate a banded workspace and
use the banded LU factorisation path, which is `O(n)` per step instead of
`O(n²)` for dense.

---

## Integration statistics

Pass `ret_stats=True` to get a `ZVODEStats` object as the third return value:

```python
t, y, stats = solve_complex_ivp(
    rhs,
    tspan=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    ret_stats=True,
)
print(stats)
# ZVODEStats(nsteps=42, nfev=63, njev=0, nlu=0)

print(stats.nsteps)    # attribute access
print(stats['nfev'])   # dict access
```

| Field | Meaning |
|-------|---------|
| `nsteps` | Total accepted steps |
| `nfev` | RHS evaluations |
| `njev` | Jacobian evaluations |
| `nlu` | LU decompositions |

`njev > 0` and `nlu > 0` only when BDF is used with a user-supplied or
finite-difference Jacobian.

---

## Backward integration

Set `tspan` so that `t0 > tf`. The solver integrates backward in time.

```python
t, y = solve_complex_ivp(
    rhs,
    tspan=(10.0, 0.0),   # backward: 10 → 0
    y0=np.exp(-1j * 10.0) * np.array([1.0 + 0j]),
)
# y[:, -1] ≈ [1+0j]  (the initial condition at t=0)
```

For knot mode, pass the times in strictly decreasing order.

---

## In-place callback convention

By default, `fun(t, y)` returns a new array (SciPy style). Set `in_place=True`
to use the signature `fun(t, y, dy)`, which fills `dy` in place and avoids an
allocation per step.

```python
def rhs_ip(t, y, dy):
    dy[0] = -1j * y[0]

t, y = solve_complex_ivp(
    rhs_ip,
    tspan=(0.0, 10.0),
    y0=[1.0 + 0.0j],
    in_place=True,
)
```

The Jacobian callback also gains a different signature when `in_place=True`:

- Dense: `jac(t, y, pd)` fills `pd` (shape `(n, n)`) in place.
- Banded: `jac(t, y, pd, ml, mu)` fills the banded array `pd` in place;
  `ml` and `mu` are the half-bandwidths passed for reference.

---

## Step-size controls

| Parameter | Default | Effect |
|-----------|---------|--------|
| `first_step` | auto | Override the initial step size. |
| `max_step` | `np.inf` | Cap the step size (useful when the solution has sharp but non-stiff features). |
| `min_step` | `0` | Raise an error rather than take a step smaller than this. |
| `max_num_steps` | `1 000 000` | Raise `RuntimeError` if this many steps are taken between two output points. |

```python
t, y = solve_complex_ivp(
    rhs,
    tspan=(0.0, 100.0),
    y0=[1.0 + 0j],
    max_step=0.5,        # no step longer than 0.5
    max_num_steps=50000,
)
```

---

## Common pitfalls

**Real initial condition** — `y0` is always cast to `complex128`. Passing a
real array triggers a `UserWarning`; silence it by writing `y0` with an
explicit imaginary part (`1.0 + 0j`) or calling `np.asarray(y0, dtype=complex)`
before passing it.

**Thread safety** — ZVODE stores its state in Fortran COMMON blocks that are
global to the process. `solve_complex_ivp` acquires a process-wide lock for the
duration of the integration, so multiple threads calling it concurrently will
run serially, not in parallel. Use multiprocessing for true parallelism.

**Non-analytic RHS with BDF** — see the [Analyticity requirement](#choosing-a-method)
note above.

---

## Parameter reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fun` | callable | — | `f(t, y) → array` or `f(t, y, dy)` (in-place). |
| `tspan` | array-like | — | Integration interval or output knots. |
| `y0` | array-like | — | Initial state; cast to `complex128`. |
| `method` | `'BDF'` or `'Adams'` | `'BDF'` | Integration method. |
| `rtol` | float or array | `1e-3` | Relative tolerance. |
| `atol` | float or array | `1e-6` | Absolute tolerance. |
| `jac` | callable or None | `None` | Jacobian callback. |
| `lband` | int or None | `None` | Lower half-bandwidth of banded Jacobian. |
| `uband` | int or None | `None` | Upper half-bandwidth of banded Jacobian. |
| `in_place` | bool | `False` | Use in-place callback convention. |
| `save_steps` | bool | `True` | Collect every step vs endpoint only. |
| `ret_stats` | bool | `False` | Return `ZVODEStats` as third value. |
| `refine` | int | `1` | Interpolated points per step (all-steps mode only). |
| `allow_overshoot` | bool | `False` | Allow solver to step past `tf`. |
| `first_step` | float | auto | Initial step size. |
| `max_step` | float | `np.inf` | Maximum step size. |
| `min_step` | float | `0` | Minimum step size. |
| `max_num_steps` | int | `1 000 000` | Step limit between output points. |
| `max_order` | int | `5`/`12` | Maximum integration order. |
| `miter` | int or None | auto | Iteration method override (0–5). |
| `save_jac` | bool | `True` | Cache and reuse Jacobian between steps. |
