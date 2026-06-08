# Compiled Callbacks Design Specification

## Goals

- Allow users to pass compiled C function pointers (ctypes, numba) as the
  right-hand side and Jacobian of `solve_complex_ivp`, bypassing Python
  interpreter overhead on every RHS/Jacobian evaluation.
- Support an optional shared user-data pointer (`ctx`) passed to both
  callbacks, enabling parameterised problems without global variables.
- Keep numba an optional dependency — no import at library level.
- Keep the public API simple: no wrapper classes, no flags.

---

## Python Public API

```python
solve_complex_ivp(
    fun,           # Python callable  OR  ctypes._CFuncPtr
    tspan,
    y0,
    *,
    jac  = None,   # Python callable  OR  ctypes._CFuncPtr  OR  None
    ctx  = None,   # ctypes.c_void_p  OR  None
    ...            # rtol, atol, method, lband, uband, etc. — unchanged
)
```

The `in_place` parameter is **removed**.

### `fun` and `jac`

Two kinds are accepted, detected by type:

| Kind | Type | How address is obtained |
|------|------|-------------------------|
| Python callable | any `callable` | n/a — called via Python |
| Compiled callback | `ctypes._CFuncPtr` | `ctypes.cast(fun, ctypes.c_void_p).value` |

Numba `@cfunc` objects expose a `.ctypes` property that is a
`ctypes._CFuncPtr`; numba users pass that explicitly:

```python
@cfunc(zvode.zvode_fun_sig)
def my_rhs(neq, t, y, dy, ctx): ...

solve_complex_ivp(my_rhs.ctypes, tspan, y0)
```

ctypes users pass the decorated function directly:

```python
@ZVODE_FUN_CTYPE
def my_rhs(neq, t, y_ptr, dy_ptr, ctx): ...

solve_complex_ivp(my_rhs, tspan, y0)
```

Mixed cases are supported: `fun` may be a Python callable while `jac` is a
compiled callback, or vice versa.  This allows gradual porting — a user can
lower the performance-critical callback to compiled code first and leave the
other as a plain Python function until needed.

### `ctx`

An optional shared `ctypes.c_void_p` passed as the `ctx` argument to **both**
compiled callbacks on every invocation.  Named `ctx` for consistency with the
C callback signatures.

- `None` (default) — NULL is passed as `ctx`.
- `ctypes.c_void_p` — its `.value` is passed as `ctx`.
- Anything else — `TypeError` is raised immediately.

The caller is responsible for keeping the referent alive for the duration of
the integration.

**Pattern 1 — numpy array of parameters:**

```python
params = np.array([lam1, lam2, coupling], dtype=np.complex128)
ctx = ctypes.cast(params.ctypes.data, ctypes.c_void_p)

solve_complex_ivp(my_rhs.ctypes, tspan, y0, ctx=ctx)
```

**Pattern 2 — ctypes Structure:**

```python
class Params(ctypes.Structure):
    _fields_ = [("lam1", ctypes.c_double * 2),
                ("lam2", ctypes.c_double * 2)]

p = Params(...)
ctx = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

solve_complex_ivp(my_rhs.ctypes, tspan, y0, ctx=ctx)
```

**Pattern 3 — compiled closure (no `ctx` needed):**

When parameters are known at compile time, numba can capture them from the
enclosing Python scope directly, avoiding the need for `ctx` entirely:

```python
def make_rhs(lam1, lam2, coupling):
    @cfunc(zvode.zvode_fun_sig)
    def rhs(neq, t, y, dy, ctx):
        # lam1, lam2, coupling captured as compile-time constants;
        # ctx is ignored
        dy[0] = lam1 * y[0] + coupling * y[1]
        dy[1] = lam2 * y[1]
    return rhs

my_rhs = make_rhs(-1.0 + 2.0j, -2.0 + 1.0j, 0.5j)
solve_complex_ivp(my_rhs.ctypes, tspan, y0)
```

Each call to `make_rhs` triggers a fresh numba compilation; cache the result
if the same parameters are reused.

If `fun` is a plain Python callable, `ctx` is silently ignored and a warning
is issued if `ctx is not None`.

---

## Compiled Callback Signatures

### RHS

```c
void fun(int neq, double t,
         const double complex *y,
         double complex       *dy,
         void                 *ctx);
```

### Jacobian

```c
void jac(int neq, double t,
         const double complex *y,
         int ml, int mu,
         double complex       *pd,
         int nrowpd,
         void                 *ctx);
```

`pd` is column-major (Fortran order).  For a dense Jacobian,
`J[i,j] = df_i/dy_j` goes to `pd[i + j*nrowpd]`.  For a banded Jacobian
the banded-storage convention applies: `pd[mu + i - j + j*nrowpd]`.

Inside a numba `@cfunc`, `numba.farray` creates a 2-D Fortran-order view
over the flat `pd` pointer, which is often more readable than manual index
arithmetic:

```python
import numba as nb

@cfunc(zvode.zvode_jac_sig)
def my_jac(neq, t, y, ml, mu, pd, nrowpd, ctx):
    J = nb.farray(pd, (nrowpd, neq))   # 2-D view, column-major
    J[0, 0] = lam1    # df[0]/dy[0]
    J[0, 1] = c       # df[0]/dy[1]
    J[1, 1] = lam2    # df[1]/dy[1]
```

---

## Exported Names

| Name | Kind | Purpose |
|------|------|---------|
| `solve_complex_ivp` | function | main entry point |
| `ZVODE_FUN_CTYPE` | ctypes type | canonical `CFUNCTYPE` for the RHS |
| `ZVODE_JAC_CTYPE` | ctypes type | canonical `CFUNCTYPE` for the Jacobian |
| `zvode_fun_sig` | numba type (lazy) | `@cfunc` signature for the RHS |
| `zvode_jac_sig` | numba type (lazy) | `@cfunc` signature for the Jacobian |

`zvode_fun_sig` and `zvode_jac_sig` are constructed on first access (module
`__getattr__`) so that numba remains an optional dependency.

### Removed from 0.2.x

- `in_place` parameter — gone; callback kind is detected by type.
- `check_cfunc_signature` — gone; ctypes validates at decoration time,
  numba at compilation time.
- `ZvodeCallback` wrapper class — not needed without per-callback user data.

---

## Address Extraction (Python layer)

```python
def _get_cfunc_address(fun):
    """Return integer address if fun is a compiled callback, else None."""
    if isinstance(fun, ctypes._CFuncPtr):
        return ctypes.cast(fun, ctypes.c_void_p).value
    return None
```

Called once per `solve_complex_ivp` invocation for `fun` and `jac`.
The result (integer or `None`) is what gets passed to the C extension.

---

## C Extension Layer

### Callback struct

```c
typedef enum { CB_PYTHON = 0, CB_CFUNC = 1 } cb_kind_t;

typedef struct {
    /* RHS */
    cb_kind_t fun_kind;
    union {
        PyObject *pyobj;   /* CB_PYTHON */
        zvode_fun cfunc;   /* CB_CFUNC  */
    } fun_u;

    /* Jacobian */
    cb_kind_t jac_kind;
    union {
        PyObject *pyobj;   /* CB_PYTHON */
        zvode_jac cfunc;   /* CB_CFUNC  */
    } jac_u;

    void *ctx;       /* shared user data; NULL when data=None */
    int   is_banded; /* 1 when miter == 4 */
    int   error;     /* set to 1 by adaptor on Python exception */
} zvode_cb_t;
```

### Argument parsing

The Python layer normalises before calling C:

- Compiled callback → pass the integer address as a Python `int`.
- Python callable → pass the callable `PyObject *` unchanged.
- `ctx` → pass `ctypes.c_void_p.value` (an integer) or `0`.

In C, `fun_obj` and `jac_obj` are parsed with `O` (generic `PyObject *`).
The kind is then determined by type inspection:

```c
if (PyLong_Check(fun_obj)) {
    cb.fun_kind     = CB_CFUNC;
    cb.fun_u.cfunc  = (zvode_fun)(uintptr_t)PyLong_AsSsize_t(fun_obj);
} else {
    cb.fun_kind     = CB_PYTHON;
    cb.fun_u.pyobj  = fun_obj;
}
cb.ctx = (void *)(uintptr_t)ctx_addr;   /* 0 when data=None */
```

### Function pointer selection (branch outside)

The adaptor is chosen **once** before the integration loop, not on every
evaluation.  `ZVODE_LOCK` ensures only one integration runs at a time, so
a static `current_cb` pointer is safe to use:

```c
static zvode_cb_t *current_cb = NULL;  /* protected by ZVODE_LOCK */

/* ... populate cb ... */

current_cb = &cb;

zvode_fun fun_fptr = (cb.fun_kind == CB_CFUNC)
                   ? cb.fun_u.cfunc       /* passed directly — zero overhead */
                   : python_fun_adaptor;

zvode_jac jac_fptr = (cb.jac_kind == CB_CFUNC)
                   ? cb.jac_u.cfunc
                   : python_jac_adaptor;

c_zvode(..., fun_fptr, ..., jac_fptr, mf, cb.ctx);

current_cb = NULL;
```

`cb.ctx` is passed as the `data` argument to `c_zvode`, so compiled callbacks
receive it directly with no extra indirection.  Python adaptors ignore the
`data` argument and read `current_cb` instead.

### Adaptors (Python path)

```c
static void python_fun_adaptor(int *neq, double *t,
                                const double complex *y,
                                double complex *dy, void *data)
{
    zvode_cb_t *cb = current_cb;  /* data == cb->ctx; ignored here */
    /* ... build numpy arrays, call cb->fun_u.pyobj, copy result ... */
    /* set cb->error = 1 on exception */
}
```

No redirectors are needed.  Compiled callbacks are passed directly to
`c_zvode` and receive `cb->ctx` as their `void *ctx` argument.

### Single C entry point

The experimental `drive_cfunc_knots` / `drive_cfunc_adaptive` entry points
(compiled-only, separate from the Python-callable `drive_knots`) are replaced
by a single `drive_knots` and `drive_adaptive` that each accept a
`zvode_cb_t` and handle both Python and compiled callbacks through the
selection mechanism above.
