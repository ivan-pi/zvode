# Banded Jacobians

When the Jacobian `J[i,j] = dF_i/dy_j` is zero for all `|i-j| > max(lband, uband)`,
ZVODE can exploit the band structure to save both memory and factorisation work.
For a system of size `n` the dense path allocates and factors an `n × n` matrix; the
banded path works with an `(lband + uband + 1) × n` strip — a significant saving for
large, narrowly banded systems.

Activate the banded path by passing `lband` and/or `uband` to `solve_complex_ivp`.
The other bandwidth defaults to 0 if omitted.

## What "lower/upper half-bandwidth" means

- **`lband`** — number of non-zero subdiagonals *below* the main diagonal.
- **`uband`** — number of non-zero superdiagonals *above* the main diagonal.

A tridiagonal matrix has `lband = uband = 1`.
A matrix that couples each row to the two rows below it and one row above has
`lband = 2`, `uband = 1`.

## Storage layout

Return a NumPy array `pd` (short for *partial derivatives*) of shape
`(lband + uband + 1, n)` from your `jac` callable.
The mapping between `pd` and the logical Jacobian `J` is:

```
pd[i - j + uband, j]  =  J[i, j]
```

Each **column** of `pd` holds the in-band entries of the corresponding column of
`J`.  Positions that fall outside the band are unused; the convention is to leave
them as zero.

### Worked example — `lband = 2`, `uband = 1`, `n = 5`

Consider a system whose Jacobian has the following sparsity pattern
(using single letters as stand-ins for non-zero values):

```
a  d  .  .  .
b  a  d  .  .
c  b  a  d  .
.  c  b  a  d
.  .  c  b  a
```

where `a` is on the main diagonal, `d` on the superdiagonal, `b` and `c` on the
first and second subdiagonals respectively, and `.` denotes zero.

The banded storage array `pd` has shape `(lband + uband + 1, n) = (4, 5)`.
Reading the formula `pd[i - j + uband, j] = J[i, j]` row by row:

```
*  d  d  d  d       row 0  (i - j = -1):  J[j-1, j] = d   superdiagonal
a  a  a  a  a       row 1  (i - j =  0):  J[j,   j] = a   main diagonal
b  b  b  b  *       row 2  (i - j =  1):  J[j+1, j] = b   first subdiagonal
c  c  c  *  *       row 3  (i - j =  2):  J[j+2, j] = c   second subdiagonal
```

`*` marks unused positions (fixed at zero by convention).

Notice how the columns of `pd` mirror the *columns* of `J` compacted downward:
column `j` of `pd` contains the entries `J[j-uband, j]` through
`J[j+lband, j]`, clamped to valid row indices.

## Writing the `jac` callback

The callback signature is the same as for a dense Jacobian — `jac(t, y)` —
but returns the compact `pd` array instead of a square matrix.

For the example above, with constant coefficients `alpha`, `beta`, `gamma`,
`delta`:

```python
lband = 2
uband = 1

def jac_banded(t, y):
    pd = np.zeros((lband + uband + 1, n), dtype=complex)
    pd[0, 1:]  = delta    # J[j-1, j] = delta,  j = 1 .. n-1  (superdiag)
    pd[1, :]   = alpha    # J[j,   j] = alpha,  j = 0 .. n-1  (main diag)
    pd[2, :-1] = beta     # J[j+1, j] = beta,   j = 0 .. n-2  (subdiag 1)
    pd[3, :-2] = gamma    # J[j+2, j] = gamma,  j = 0 .. n-3  (subdiag 2)
    return pd
```

For a state-dependent Jacobian the same indexing applies; just compute the
values from `y` before filling `pd`.

## Calling `solve_complex_ivp`

Pass `lband`, `uband`, and `jac` as keyword arguments:

```python
from zvode import solve_complex_ivp

sol = solve_complex_ivp(
    fun, [t0, tf], y0,
    jac=jac_banded,
    lband=2,
    uband=1,
)
t, y = sol.t, sol.y
```

To let ZVODE estimate the banded Jacobian by finite differences instead,
omit `jac` while still providing `lband` and `uband`.

Output modes and integration statistics are covered in
[`how-to-procedural-api.md`](how-to-procedural-api.md).

## Memory saving at a glance

| `n`   | `lband` | `uband` | Dense `n²` | Banded `(l+u+1)·n` | Memory reduction |
|------:|--------:|--------:|-----------:|-------------------:|-----------------:|
| 100   | 2       | 1       | 10 000     | 400                | 25×              |
| 1 000 | 2       | 1       | 1 000 000  | 4 000              | 250×             |
| 1 000 | 5       | 3       | 1 000 000  | 9 000              | 111×             |

Each entry is the number of complex numbers in the Jacobian workspace (`Dense / Banded`
gives the memory reduction factor).  The banded path also avoids fill-in during
LU factorisation, so the work-per-step saving is comparable.

## Complete example

[`demo_banded_jacobian.py`](demo_banded_jacobian.py) solves `y' = Ay` for
a 5-component chain (`lband=2`, `uband=1`) with complex coefficients and
verifies the result against `scipy.linalg.expm`.
