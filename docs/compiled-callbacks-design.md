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
    data = None,   # ctypes.c_void_p  OR  None
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
compiled callback, or vice versa.

### `data`

An optional shared `ctypes.c_void_p` passed as the `ctx` argument to **both**
compiled callbacks on every invocation.

- `None` (default) — NULL is passed as `ctx`.
- `ctypes.c_void_p` — its `.value` is passed as `ctx`.
- Anything else — `TypeError` is raised immediately.

The caller is responsible for keeping the referent alive for the duration of
the integration.  The typical pattern is a numpy array or a ctypes Structure:

```python
params = np.array([lam1, lam2, coupling], dtype=np.complex128)
data = ctypes.cast(params.ctypes.data, ctypes.c_void_p)

solve_complex_ivp(my_rhs.ctypes, tspan, y0, data=data)
```

If `fun` is a plain Python callable, `data` is silently ignored and a warning
is issued if `data is not None`.

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
- `data` → pass `ctypes.c_void_p.value` (an integer) or `0`.

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

The adaptor or redirector is chosen **once** before the integration loop,
not inside the callback on every evaluation:

```c
zvode_fun fun_fptr = (cb.fun_kind == CB_CFUNC)
                   ? cfunc_fun_redirector
                   : python_fun_adaptor;

zvode_jac jac_fptr = (cb.jac_kind == CB_CFUNC)
                   ? cfunc_jac_redirector
                   : python_jac_adaptor;

c_zvode(..., fun_fptr, ..., jac_fptr, mf, &cb);
```

`&cb` is the single `data` pointer passed to `c_zvode`; all four functions
receive it as their `void *data` argument.

### Adaptors (Python path)

Call the Python object and handle exceptions:

```c
static void python_fun_adaptor(int *neq, double *t,
                                const double complex *y,
                                double complex *dy, void *data)
{
    zvode_cb_t *cb = data;
    /* ... build numpy arrays, call cb->fun_u.pyobj, copy result ... */
    /* set cb->error = 1 on exception */
}
```

### Redirectors (compiled path)

Thin wrappers that forward `cb->ctx` to the compiled function.
Necessary because `c_zvode` passes `&cb` as `data`, not `cb->ctx` directly:

```c
static void cfunc_fun_redirector(int *neq, double *t,
                                  const double complex *y,
                                  double complex *dy, void *data)
{
    zvode_cb_t *cb = data;
    cb->fun_u.cfunc(*neq, *t, y, dy, cb->ctx);
}

static void cfunc_jac_redirector(int *neq, double *t,
                                  const double complex *y,
                                  int *ml, int *mu,
                                  double complex *pd, int *nrowpd,
                                  void *data)
{
    zvode_cb_t *cb = data;
    cb->jac_u.cfunc(*neq, *t, y, *ml, *mu, pd, *nrowpd, cb->ctx);
}
```

The redirector adds one extra function call and one pointer dereference per
evaluation — negligible against any real computation in the callback.

### Single C entry point

The current `drive_knots` / `drive_cfunc_knots` / `drive_cfunc_adaptive`
entry points collapse into a single `drive_knots` and `drive_adaptive` that
each accept a `zvode_cb_t` and handle both Python and compiled callbacks
through the selection mechanism above.
