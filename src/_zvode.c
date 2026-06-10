#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <complex.h>


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>

#include "zvode.h"

/* ------------------------------------------------------------------ */
/* Debug helpers — compile with -DZVODE_DEBUG (or -DZVODE_DEBUG=1)   */
/* ------------------------------------------------------------------ */

#ifndef ZVODE_DEBUG
#define ZVODE_DEBUG 0
#endif

static const char *dtype_name(int typenum) {
    switch (typenum) {
        case NPY_FLOAT32:    return "float32";
        case NPY_FLOAT64:    return "float64";
        case NPY_COMPLEX64:  return "complex64";
        case NPY_COMPLEX128: return "complex128";
        case NPY_INT32:      return "int32";
        case NPY_INT64:      return "int64";
        default:             return "?";
    }
}

static void dump_repr(const char *label, PyObject *obj) {
    if (obj == NULL) {
        fprintf(stderr, "  %-8s = <NULL>\n", label);
        return;
    }
    PyObject *r = PyObject_Repr(obj);
    const char *s = r ? PyUnicode_AsUTF8(r) : NULL;
    fprintf(stderr, "  %-8s = %s\n", label, s ? s : "<repr failed>");
    Py_XDECREF(r);
}

static void dump_array(const char *label, PyArrayObject *arr) {
    if (arr == NULL) {
        fprintf(stderr, "  %-8s = <NULL>\n", label);
        return;
    }
    fprintf(stderr, "  %-8s = ndarray(shape=(", label);
    int nd = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    for (int i = 0; i < nd; ++i) {
        fprintf(stderr, "%lld%s",
                (long long) dims[i], (i + 1 < nd) ? ", " : "");
    }
    fprintf(stderr, "), dtype=%s, size=%lld, C=%d F=%d W=%d A=%d, data=%p",
            dtype_name(PyArray_TYPE(arr)),
            (long long) PyArray_SIZE(arr),
            PyArray_IS_C_CONTIGUOUS(arr) ? 1 : 0,
            PyArray_IS_F_CONTIGUOUS(arr) ? 1 : 0,
            PyArray_ISWRITEABLE(arr)     ? 1 : 0,
            PyArray_ISALIGNED(arr)       ? 1 : 0,
            PyArray_DATA(arr));

    /* Small value preview — first up to 3 elements. */
    npy_intp n = PyArray_SIZE(arr);
    npy_intp k = n < 3 ? n : 3;
    if (k > 0) {
        fprintf(stderr, ", head=[");
        switch (PyArray_TYPE(arr)) {
        case NPY_FLOAT64: {
            const double *p = PyArray_DATA(arr);
            for (npy_intp i = 0; i < k; ++i)
                fprintf(stderr, "%.6g%s", p[i], (i + 1 < k) ? ", " : "");
            break;
        }
        case NPY_COMPLEX128: {
            const double complex *p = PyArray_DATA(arr);
            for (npy_intp i = 0; i < k; ++i)
                fprintf(stderr, "(%.6g%+.6gj)%s",
                        creal(p[i]), cimag(p[i]),
                        (i + 1 < k) ? ", " : "");
            break;
        }
        case NPY_INT32: {
            const int32_t *p = PyArray_DATA(arr);
            for (npy_intp i = 0; i < k; ++i)
                fprintf(stderr, "%d%s", (int) p[i], (i + 1 < k) ? ", " : "");
            break;
        }
        default:
            fprintf(stderr, "<unprinted>");
        }
        fprintf(stderr, "%s]", (n > k) ? ", ..." : "");
    }
    fprintf(stderr, ")\n");
}

static void dump_zvode_args(
        PyObject *fun, PyArrayObject *ap_y,
        double t, double tout, int itol,
        PyArrayObject *ap_rtol, PyArrayObject *ap_atol,
        int itask, int istate, int iopt,
        PyArrayObject *ap_zwork, PyArrayObject *ap_rwork,
        PyArrayObject *ap_iwork,
        PyObject *jac, int mf) {

    fprintf(stderr, "---- zvode args ----\n");
    dump_repr ("fun",   fun);
    dump_array("y",     ap_y);
    fprintf(stderr, "  t        = %.17g\n", t);
    fprintf(stderr, "  tout     = %.17g\n", tout);
    fprintf(stderr, "  itol     = %d\n",    itol);
    dump_array("rtol",  ap_rtol);
    dump_array("atol",  ap_atol);
    fprintf(stderr, "  itask    = %d\n",    itask);
    fprintf(stderr, "  istate   = %d\n",    istate);
    fprintf(stderr, "  iopt     = %d\n",    iopt);
    dump_array("zwork", ap_zwork);
    dump_array("rwork", ap_rwork);
    dump_array("iwork", ap_iwork);
    dump_repr ("jac",   jac);
    fprintf(stderr, "  mf       = %d\n",    mf);
    fprintf(stderr, "--------------------\n");
    fflush(stderr);
}

/* ZVODE's IWORK is Fortran default INTEGER, declared as `int` in the C
 * header.  We expose it to Python as int32, so the two must agree. */
_Static_assert(sizeof(int) == 4, "iwork bridging assumes a 32-bit C int");

/* ------------------------------------------------------------------ */
/* Array argument validators                                          */
/* ------------------------------------------------------------------ */

/* Returns 1 (ok) or 0 (failure, exception set).
 * ndim     : required number of dimensions
 * typenum  : required NumPy type (e.g. NPY_COMPLEX128)
 * order    : 'C' = require C-contiguous, 'F' = require Fortran-contiguous,
 *            0   = no contiguity check */
static inline int
check_array(PyArrayObject *ap, const char *name, int ndim, int typenum, char order)
{
    if (PyArray_NDIM(ap) != ndim) {
        PyErr_Format(PyExc_ValueError,
            "zvode: %s must be %d-D (got %d-D)", name, ndim, PyArray_NDIM(ap));
        return 0;
    }
    if (PyArray_TYPE(ap) != typenum) {
        PyErr_Format(PyExc_TypeError,
            "zvode: %s must have dtype %s", name, dtype_name(typenum));
        return 0;
    }
    if (order == 'C' && !PyArray_IS_C_CONTIGUOUS(ap)) {
        PyErr_Format(PyExc_ValueError,
            "zvode: %s must be C-contiguous", name);
        return 0;
    }
    if (order == 'F' && !PyArray_IS_F_CONTIGUOUS(ap)) {
        PyErr_Format(PyExc_ValueError,
            "zvode: %s must be Fortran-contiguous", name);
        return 0;
    }
    return 1;
}

/* Returns 1 (ok) or 0 (failure, exception set). */
static inline int
check_array_1d(PyArrayObject *ap, const char *name, int typenum) {
    return check_array(ap, name, 1, typenum, 'C');
}

static inline int
check_writable(PyArrayObject *ap, const char *name) {
    if (!PyArray_ISWRITEABLE(ap)) {
        PyErr_Format(PyExc_ValueError, "zvode: %s must be writable", name);
        return 0;
    }
    return 1;
}

static inline int
check_array_scalar_or_1d(PyArrayObject *ap, const char *name, int typenum)
{
    if (PyArray_NDIM(ap) > 1) {
        PyErr_Format(PyExc_ValueError,
            "zvode: %s must be a scalar or 1-D array (got %d-D)",
            name, PyArray_NDIM(ap));
        return 0;
    }
    if (PyArray_TYPE(ap) != typenum) {
        PyErr_Format(PyExc_TypeError,
            "zvode: %s must have dtype %s", name, dtype_name(typenum));
        return 0;
    }
    if (PyArray_NDIM(ap) == 1 && !PyArray_IS_C_CONTIGUOUS(ap)) {
        PyErr_Format(PyExc_ValueError,
            "zvode: %s must be C-contiguous", name);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Callback plumbing                                                  */
/* ------------------------------------------------------------------ */

/* Compiled-callback dispatch tag.
 *   CB_PYTHON = Python callable
 *   CB_CFUNC  = raw C function pointer extracted by the Python layer
 *   CB_NONE   = no callback provided (jac only; ZVODE must not call it) */
typedef enum { CB_PYTHON = 0, CB_CFUNC = 1, CB_NONE = 2 } cb_kind_t;

struct zvode_callbacks {
    /* RHS */
    cb_kind_t fun_kind;
    union {
        PyObject  *pyobj;   /* CB_PYTHON: the callable (never NULL) */
        zvode_fun  cfunc;   /* CB_CFUNC:  raw function pointer       */
    } fun_u;

    /* Jacobian — jac_kind == CB_NONE when no Jacobian is provided. */
    cb_kind_t jac_kind;
    union {
        PyObject  *pyobj;   /* CB_PYTHON: the callable (never NULL) */
        zvode_jac  cfunc;   /* CB_CFUNC:  raw function pointer      */
    } jac_u;

    void *ctx;         /* shared user data for compiled callbacks; NULL = none */
    int   jac_is_banded; /* 1 when MITER=4 (abs(mf)%10 == 4), 0 otherwise */
    int   error;         /* set to 1 by Python adaptor on exception */
    /* TODO: add zewset and zwnorm in the future */
};

static void fun_adaptor(
        int neq,
        double t,
        const double complex y[],
        double complex dy[],
        void *data) {

    struct zvode_callbacks *cb = (struct zvode_callbacks *) data;
    assert(cb != NULL);
    assert(neq > 0);

    if (cb->fun_kind == CB_CFUNC) {
        /* Compiled path: call the C function pointer directly, no Python overhead. */
        assert(cb->fun_u.cfunc != NULL);
        cb->fun_u.cfunc(neq, t, y, dy, cb->ctx);
        return;
    }

    /* CB_PYTHON path */
    assert(cb->fun_u.pyobj != NULL);

    /* A previous callback already raised a Python exception; short-circuit so
     * the original exception is preserved rather than clobbered by calling
     * back into Python with an error already pending. */
    if (cb->error) {
        return;
    }

    const npy_intp dims[1] = { neq };

    /* Wrap the solver-owned y buffer as a read-only NumPy view (no copy). */
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_COMPLEX128, (void *) y);
    if (ap_y == NULL) {
        cb->error = 1;
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    /* SciPy-style call: fun(t, y) returns the derivative array.  Use the
     * vectorcall protocol with a small C stack to skip the format-string
     * parsing and the intermediate args tuple PyObject_CallFunction builds. */
    PyObject *t_obj = PyFloat_FromDouble(t);
    if (t_obj == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }
    PyObject *stack[2] = { t_obj, (PyObject *) ap_y };
    PyObject *result = PyObject_Vectorcall(cb->fun_u.pyobj, stack, 2, NULL);
    Py_DECREF(t_obj);
    if (result == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    /* Coerce the result to a contiguous complex128 array.  FORCECAST accepts
     * Python lists and real arrays, matching the old `dy[:] = fun(t, y)`. */
    PyArrayObject *arr = (PyArrayObject *) PyArray_FROM_OTF(
        result, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (arr == NULL) {
        Py_DECREF(result);
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    if (PyArray_SIZE(arr) != (npy_intp) neq) {
        PyErr_Format(PyExc_ValueError,
            "fun(t, y) must return an array of size neq=%d; got size %zd.",
            neq, (Py_ssize_t) PyArray_SIZE(arr));
        Py_DECREF(arr);
        Py_DECREF(result);
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    memcpy(dy, PyArray_DATA(arr), (size_t) neq * sizeof(double complex));

    Py_DECREF(arr);
    Py_DECREF(result);
    Py_DECREF(ap_y);
}

static void jac_adaptor(
        int neq,
        double t,
        const double complex y[],
        int ml, int mu,
        double complex pd[],
        int nrowpd,
        void *data) {

    struct zvode_callbacks *cb = (struct zvode_callbacks *) data;
    assert(cb != NULL);
    assert(neq > 0);
    assert(ml >= 0 && mu >= 0);
    assert(nrowpd >= (cb->jac_is_banded ? ml+mu+1 : neq));

    if (cb->jac_kind == CB_CFUNC) {
        /* Compiled path: call the C function pointer directly. */
        assert(cb->jac_u.cfunc != NULL);
        cb->jac_u.cfunc(neq, t, y, ml, mu, pd, nrowpd, cb->ctx);
        return;
    }

    /* CB_PYTHON path — jac_adaptor is only invoked by ZVODE when miter
     * requires a user Jacobian (miter=1 or 4).  CB_NONE reaching here
     * would indicate an mf/miter mismatch. */
    assert(cb->jac_kind == CB_PYTHON);
    assert(cb->jac_u.pyobj != NULL);

    /* A previous callback already raised a Python exception; short-circuit so
     * the original exception is preserved rather than clobbered by calling
     * back into Python with an error already pending. */
    if (cb->error) {
        return;
    }

    const npy_intp dims_y[1] = { (npy_intp) neq };
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_y, NPY_COMPLEX128, (void *) y);
    if (ap_y == NULL) {
        cb->error = 1;
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    /* SciPy-style call: jac(t, y) returns the Jacobian array.
     *   dense  (miter=1): shape (neq, neq), J[i,j] = df_i/dy_j
     *   banded (miter=4): shape (ml + mu + 1, neq) */
    PyObject *t_obj = PyFloat_FromDouble(t);
    if (t_obj == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }
    PyObject *stack[2] = { t_obj, (PyObject *) ap_y };
    PyObject *result = PyObject_Vectorcall(cb->jac_u.pyobj, stack, 2, NULL);
    Py_DECREF(t_obj);
    if (result == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    /* Coerce to F-contiguous complex128 so each column is laid out
     * contiguously, ready to copy into PD's column-major buffer. */
    PyArrayObject *arr = (PyArrayObject *) PyArray_FROM_OTF(
        result, NPY_COMPLEX128, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (arr == NULL) {
        Py_DECREF(result);
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    const npy_intp rows = cb->jac_is_banded ? (npy_intp) (ml + mu + 1) : (npy_intp) neq;
    if (PyArray_NDIM(arr) != 2 ||
        PyArray_DIM(arr, 0) != rows ||
        PyArray_DIM(arr, 1) != (npy_intp) neq) {
        if (PyArray_NDIM(arr) != 2) {
            PyErr_Format(PyExc_ValueError,
                "jac(t, y) must return a 2-D array of shape (%zd, %d); "
                "got an array with ndim=%d.",
                (Py_ssize_t) rows, neq, PyArray_NDIM(arr));
        } else {
            PyErr_Format(PyExc_ValueError,
                "jac(t, y) must return an array of shape (%zd, %d); "
                "got shape (%zd, %zd).",
                (Py_ssize_t) rows, neq,
                (Py_ssize_t) PyArray_DIM(arr, 0),
                (Py_ssize_t) PyArray_DIM(arr, 1));
        }
        Py_DECREF(arr);
        Py_DECREF(result);
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    /* PD is column-major with leading dimension NROWPD.  Copy each column of
     * the user Jacobian into the first `rows` entries of PD's column; the
     * remaining nrowpd - rows rows are LAPACK fill workspace, left untouched. */
    const double complex *src = (const double complex *) PyArray_DATA(arr);
    for (npy_intp j = 0; j < (npy_intp) neq; ++j) {
        memcpy(&pd[(size_t) j * (size_t) nrowpd],
               &src[(size_t) j * (size_t) rows],
               (size_t) rows * sizeof(double complex));
    }

    Py_DECREF(arr);
    Py_DECREF(result);
    Py_DECREF(ap_y);
}

/* ------------------------------------------------------------------ */
/* zvode                                                              */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(zvode_doc,
"zvode(fun, y, t, tout, itol, rtol, atol, itask, istate, iopt,\n"
"      zwork, rwork, iwork, jac, mf) -> (t, istate)\n"
"\n"
"Advance a complex ODE system with a single ZVODE call.\n"
"\n"
"`y`, `zwork`, `rwork`, `iwork` are modified in place and must be\n"
"contiguous arrays of dtype ``complex128``, ``complex128``, ``float64`` and ``int32``.\n"
"`fun` is called as ``fun(t, y) -> array`` of shape ``(neq,)``; `jac` (or None)\n"
"is called as ``jac(t, y) -> array`` of shape ``(neq, neq)`` (dense) or\n"
"``(ml + mu + 1, neq)`` (banded).  Compiled cfunc callbacks instead use the\n"
"in-place C ABI.  The `y` passed to a Python callback is a read-only view onto\n"
"solver-owned memory, valid only for that call; copy it to retain the state.\n"
"Returns the advanced time and the ZVODE istate.\n");

static PyObject* zvode_py(PyObject* Py_UNUSED(self), PyObject *args) {

    PyArrayObject *ap_y = NULL, *ap_rtol = NULL, *ap_atol = NULL;
    PyArrayObject *ap_zwork = NULL, *ap_rwork = NULL, *ap_iwork = NULL;

    double t, tout;
    int itol, itask, istate, iopt, mf;

    /* zvode_py is the low-level single-step entry point used only from the
     * Python-level integration loop (_zvode_adaptive/_zvode_knots).  It
     * always receives Python callables — never compiled cfuncs. */
    PyObject *fun_obj = NULL, *jac_obj = NULL;

    if (!PyArg_ParseTuple(args,"OO!ddiO!O!iiiO!O!O!Oi:zvode",
       &fun_obj,
       &PyArray_Type, &ap_y,
       &t, &tout, &itol,
       &PyArray_Type, &ap_rtol,
       &PyArray_Type, &ap_atol,
       &itask, &istate, &iopt,
       &PyArray_Type, &ap_zwork,
       &PyArray_Type, &ap_rwork,
       &PyArray_Type, &ap_iwork,
       &jac_obj, &mf)) {
        return NULL;
    }

    assert(PyCallable_Check(fun_obj));
    assert(jac_obj == Py_None || PyCallable_Check(jac_obj));

    /* zvode_py always uses Python callbacks; cfuncs go through drive_knots/
     * drive_adaptive which handle the dispatch internally. */
    struct zvode_callbacks cb = {
        .fun_kind      = CB_PYTHON,
        .fun_u         = { .pyobj = fun_obj },
        .jac_kind      = (jac_obj == Py_None) ? CB_NONE : CB_PYTHON,
        .jac_u         = { .pyobj = (jac_obj == Py_None) ? NULL : jac_obj },
        .ctx           = NULL,
        .jac_is_banded = (abs(mf) % 10 == 4),
    };

    if (ZVODE_DEBUG) {
        dump_zvode_args(fun_obj, ap_y, t, tout, itol, ap_rtol, ap_atol,
                        itask, istate, iopt, ap_zwork, ap_rwork, ap_iwork,
                        jac_obj, mf);
    }

    const int neq = (int) PyArray_DIM(ap_y, 0);
    assert(neq > 0);          /* Python guarantees y0 is non-empty */
    assert(itask >= 1 && itask <= 5);  /* Python manages itask internally */

    assert(abs(mf) % 10 <= 5);
    assert(abs(mf)/10 == 1 || abs(mf)/10 == 2); /* method */

    /* Python validates all of the following before the first call and the
     * arrays are not supposed to change between repeated calls.  Keep the
     * checks compiled in but only run them when ZVODE_DEBUG is set so they
     * can be re-enabled during development without rebuilding from scratch. */
    if (ZVODE_DEBUG && istate == 1) {

        if (!PyCallable_Check(fun_obj)) {
            PyErr_SetString(PyExc_TypeError, "zvode: fun must be callable");
            return NULL;
        }

        if (jac_obj != Py_None && !PyCallable_Check(jac_obj)) {
            PyErr_SetString(PyExc_TypeError, "zvode: jac must be callable or None");
            return NULL;
        }

        /* Validate dtype, dimensionality, contiguity, and writability for every
         * array argument.*/
        if (!check_array_1d(ap_y,     "y",     NPY_COMPLEX128) || !check_writable(ap_y,     "y"))     return NULL;
        if (!check_array_1d(ap_zwork, "zwork", NPY_COMPLEX128) || !check_writable(ap_zwork, "zwork")) return NULL;
        if (!check_array_1d(ap_rwork, "rwork", NPY_FLOAT64)    || !check_writable(ap_rwork, "rwork")) return NULL;
        if (!check_array_1d(ap_iwork, "iwork", NPY_INT32)      || !check_writable(ap_iwork, "iwork")) return NULL;

        if (!check_array_scalar_or_1d(ap_rtol, "rtol", NPY_FLOAT64)) return NULL;
        if (!check_array_scalar_or_1d(ap_atol, "atol", NPY_FLOAT64)) return NULL;

        /* itol controls whether rtol/atol are scalar (length 1) or per-component
         * (length neq).  ZVODE convention: bit 0 set → rtol is array, bit 1 set →
         * atol is array.
         *   itol=1: rtol scalar, atol scalar
         *   itol=2: rtol scalar, atol array
         *   itol=3: rtol array,  atol scalar
         *   itol=4: rtol array,  atol array  */
        {
            int rtol_scalar = (itol == 1 || itol == 2);
            int atol_scalar = (itol == 1 || itol == 3);
            npy_intp rtol_expected = rtol_scalar ? 1 : (npy_intp) neq;
            npy_intp atol_expected = atol_scalar ? 1 : (npy_intp) neq;
            if (PyArray_SIZE(ap_rtol) != rtol_expected) {
                PyErr_Format(PyExc_ValueError,
                    "zvode: rtol must have length %d for itol=%d (got %d)",
                    (int) rtol_expected, itol, (int) PyArray_SIZE(ap_rtol));
                return NULL;
            }
            if (PyArray_SIZE(ap_atol) != atol_expected) {
                PyErr_Format(PyExc_ValueError,
                    "zvode: atol must have length %d for itol=%d (got %d)",
                    (int) atol_expected, itol, (int) PyArray_SIZE(ap_atol));
                return NULL;
            }
        }
    }

    const int lzw = (int) PyArray_SIZE(ap_zwork);
    const int lrw = (int) PyArray_SIZE(ap_rwork);
    const int liw = (int) PyArray_SIZE(ap_iwork);

    double complex *y     = (double complex *) PyArray_DATA(ap_y);
    double complex *zwork = (double complex *) PyArray_DATA(ap_zwork);
    double         *rwork = (double *)         PyArray_DATA(ap_rwork);
    int            *iwork = (int *)            PyArray_DATA(ap_iwork);
    const double   *rtol  = (const double *)   PyArray_DATA(ap_rtol);
    const double   *atol  = (const double *)   PyArray_DATA(ap_atol);

    // Call the actual "C" integrator
    // N.b.: argument y is modified in place
    c_zvode(
        &fun_adaptor,
        neq, y, &t, tout,
        itol, rtol, atol,
        itask, &istate,
        iopt, zwork, lzw, rwork, lrw, iwork, liw,
        &jac_adaptor,
        mf,
        &cb
    );

    /* If a callback raised a Python exception, cb.error is set and the
     * exception is already active — return NULL to propagate it. */
    if (cb.error) {
        assert(PyErr_Occurred());
        return NULL;
    }

    // Return the (t, istate) tuple
    return Py_BuildValue("di",t,istate);
}

/* ------------------------------------------------------------------ */
/* zvindy (interpolation)                                             */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(zvindy_doc,
"zvindy(t, k, yh, h, tn, hu, dky) -> None\n"
"\n"
"Interpolate the `k`-th derivative of `y` at time `t` using the Nordsieck array.\n"
"\n"
"Must be called after at least one successful ZVODE step.  The ZVODE internal\n"
"state (``TN``, ``H``, ``NQ``, ...) is shared via Fortran COMMON blocks, so no explicit\n"
"state argument is needed.\n"
"\n"
"Parameters\n"
"----------\n"
"t   : float  -- interpolation time; must lie in ``[tn - hu, tn]``.\n"
"k   : int    -- derivative order; must satisfy ``0 <= k <= yh.shape[1] - 1``.\n"
"yh  : ``complex128`` ndarray, shape ``(ldyh, nq+1)``, F-contiguous -- Nordsieck array.\n"
"h   : float  -- ``HCUR``, the step size the Nordsieck array is scaled to.\n"
"tn  : float  -- ``TCUR``, the current solver time.\n"
"hu  : float  -- ``HU``, the last successfully used step size.\n"
"dky : ``complex128`` ndarray, 1-D length n, writable -- receives the result.\n"
"\n"
"Raises\n"
"------\n"
"ValueError -- if k is out of range or t is outside [tn - hu, tn].\n");

static PyObject* zvindy_py(PyObject* Py_UNUSED(self), PyObject *args) {

    double t, h, tn, hu;
    PyArrayObject *ap_yh = NULL, *ap_dky = NULL;
    int k;

    if (!PyArg_ParseTuple(args, "diO!dddO!:zvindy",
            &t, &k,
            &PyArray_Type, &ap_yh,
            &h, &tn, &hu,
            &PyArray_Type, &ap_dky)) {
        return NULL;
    }

    /* Python always passes correctly typed, contiguous, writable arrays.
     * Only verify in debug builds — zvindy is called on every interpolated
     * sub-point (refine-1 times per accepted step) so these checks would
     * otherwise run in a tight loop during integration. */
    if (ZVODE_DEBUG) {
        if (!check_array(ap_yh, "yh", 2, NPY_COMPLEX128, 'F')) return NULL;
        if (!check_array(ap_dky, "dky", 1, NPY_COMPLEX128, 'C')) return NULL;
        if (!check_writable(ap_dky, "dky"))                      return NULL;
    }

    const int n    = (int) PyArray_DIM(ap_dky,0);     /* number of equations */
    const int ldyh = (int) PyArray_DIM(ap_yh, 0);     /* leading dimension   */
    const int nq   = (int) PyArray_DIM(ap_yh, 1) - 1; /* current order       */

    assert(ldyh >= n);

    if (k < 0 || k > nq) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: k must satisfy 0 <= k <= %d (got %d)", nq, k);
        return NULL;
    }

    double complex *yh  = (double complex *) PyArray_DATA(ap_yh);
    double complex *dky = (double complex *) PyArray_DATA(ap_dky);

    const int iflag = c_zvindy(n, t, yh, ldyh, k, dky,
        &(struct zvode_step_t){.h = h, .tn = tn, .hu = hu, .nq = nq});

    if (iflag != 0) {
        if (iflag == -1) {
            PyErr_Format(PyExc_ValueError,
                "zvindy: k=%d is out of range [0, nq=%d] (Fortran IFLAG=-1)", k, nq);
            return NULL;
        }
        if (iflag == -2) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                "zvindy: t=%.17g is outside the valid interval [tn-hu, tn] "
                "(Fortran IFLAG=-2)", t);
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        assert(0 && "zvindy: unexpected IFLAG — contract violation");
    }

    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* cb_init_from_pyobjs — shared callback-struct initialiser           */
/* ------------------------------------------------------------------ */

/* Populate *cb from the three Python objects that drive_knots_py and
 * drive_adaptive_py both receive as their first three arguments.
 *
 * fun_obj: Python callable  → CB_PYTHON
 *          Python int        → CB_CFUNC (raw function-pointer address)
 * jac_obj: Py_None           → CB_PYTHON with pyobj = Py_None (no jac)
 *          Python callable   → CB_PYTHON
 *          Python int        → CB_CFUNC
 * ctx_obj: Python int; 0 maps to NULL via PyLong_AsVoidPtr.
 * mf     : ZVODE method flag; used to derive jac_is_banded.
 *
 * Returns 0 on success; sets PyErr and returns -1 on error.
 */
static int
cb_init_from_pyobjs(struct zvode_callbacks *cb,
                    PyObject *fun_obj, PyObject *jac_obj,
                    PyObject *ctx_obj, int mf)
{
    cb->error = 0;

    if (PyCallable_Check(fun_obj)) {
        cb->fun_kind    = CB_PYTHON;
        cb->fun_u.pyobj = fun_obj;
    } else {
        cb->fun_kind    = CB_CFUNC;
        /* PyLong_AsVoidPtr returns void *; converting it to a function
         * pointer is forbidden by ISO C (hence the -Wpedantic warning) but
         * is required to accept a raw ctypes/numba callback address and works
         * on every platform zvode targets. */
        cb->fun_u.cfunc = (zvode_fun) PyLong_AsVoidPtr(fun_obj);
        if (PyErr_Occurred()) return -1;
        assert(cb->fun_u.cfunc != NULL);
    }

    if (jac_obj == Py_None) {
        cb->jac_kind    = CB_NONE;
        cb->jac_u.pyobj = NULL;
    } else if (PyCallable_Check(jac_obj)) {
        cb->jac_kind    = CB_PYTHON;
        cb->jac_u.pyobj = jac_obj;
    } else {
        cb->jac_kind    = CB_CFUNC;
        cb->jac_u.cfunc = (zvode_jac) PyLong_AsVoidPtr(jac_obj);  /* see note above */
        if (PyErr_Occurred()) return -1;
        assert(cb->jac_u.cfunc != NULL);
    }

    cb->ctx = PyLong_AsVoidPtr(ctx_obj);
    if (PyErr_Occurred()) return -1;

    cb->jac_is_banded = (abs(mf) % 10 == 4);

    assert(cb->fun_kind == CB_CFUNC ||
           (cb->fun_u.pyobj != NULL && PyCallable_Check(cb->fun_u.pyobj)));
    assert(cb->jac_kind == CB_NONE ||
           cb->jac_kind == CB_CFUNC ||
           (cb->jac_u.pyobj != NULL && PyCallable_Check(cb->jac_u.pyobj)));
    return 0;
}

/* ------------------------------------------------------------------ */
/* drive_knots                                                        */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(drive_knots_doc,
"drive_knots(fun, jac, ctx, mf, tspan, y, ts_out, ys_out,\n"
"            itol, rtol, atol, iopt, zwork, rwork, iwork) -> (istate, knots_completed)\n"
"\n"
"Integrate a complex ODE system to a sequence of pre-specified output knots.\n"
"\n"
"Accepts both Python callables and compiled C function pointers (ctypes/numba).\n"
"For compiled callbacks, pass the function pointer address as a Python int.\n"
"\n"
"Advances the ODE from ``tspan[0]`` to ``tspan[-1]``, evaluating the solution\n"
"at each requested knot and writing the results into the pre-allocated output\n"
"arrays ``ts_out`` and ``ys_out``.  The initial condition (``tspan[0]``, ``y``)\n"
"is copied into column 0 of the output arrays before the first ZVODE call.\n"
"\n"
"Parameters\n"
"----------\n"
"fun    : callable or int -- RHS.  Python callable: called as ``fun(t, y, dy)``\n"
"         (must fill ``dy`` in place).  Compiled: an int holding the function\n"
"         pointer address; called as ``fun(neq, t, y, dy, ctx)``.\n"
"jac    : callable, int, or None -- Jacobian.  Same kind convention as fun.\n"
"         Python callable: ``jac(t, y, pd)`` or ``jac(t, y, pd, ml, mu)``.\n"
"         Compiled: ``jac(neq, t, y, ml, mu, pd, nrowpd, ctx)``.\n"
"         Pass ``None`` when not used.\n"
"ctx    : int -- shared user-data pointer for compiled callbacks; 0 = NULL.\n"
"         Ignored when the corresponding callback is a Python callable.\n"
"mf     : int -- ZVODE method flag (encodes linear multistep method and miter).\n"
"tspan  : float64 ndarray, 1-D -- output knot times; ``tspan[0]`` is t0.\n"
"         Must have at least 2 elements and be strictly monotone.\n"
"y      : complex128 ndarray, 1-D, writable -- working state vector.\n"
"         Must be initialised to ``y(tspan[0])`` by the caller on entry.\n"
"         On return contains the last successfully reached state.\n"
"ts_out : float64 ndarray, 1-D, writable -- receives the output times;\n"
"         must have length ``len(tspan)``.\n"
"ys_out : complex128 ndarray, shape (neq, len(tspan)), F-contiguous, writable\n"
"         -- receives the solution; column k holds the state at ``ts_out[k]``.\n"
"itol   : int -- tolerance mode flag (1–4); controls scalar vs per-component\n"
"         interpretation of ``rtol`` and ``atol``.\n"
"rtol   : float64 scalar or 1-D ndarray -- relative tolerance.\n"
"atol   : float64 scalar or 1-D ndarray -- absolute tolerance.\n"
"iopt   : int -- optional-input flag: 0 = use ZVODE defaults,\n"
"         1 = read optional inputs from the ``rwork``/``iwork`` slots.\n"
"zwork  : complex128 ndarray, 1-D, writable -- ZVODE complex workspace.\n"
"rwork  : float64 ndarray, 1-D, writable -- ZVODE real workspace.\n"
"iwork  : int32 ndarray, 1-D, writable -- ZVODE integer workspace.\n"
"\n"
"Returns\n"
"-------\n"
"(istate, knots_completed) : (int, int)\n"
"    ``istate`` is the final ZVODE istate (2 = success, negative = failure).\n"
"    ``knots_completed`` is the number of columns written into ``ts_out`` and\n"
"    ``ys_out``, including column 0 (the initial condition).  On success this\n"
"    equals ``len(tspan)``; on failure it equals the number of knots reached\n"
"    before ZVODE gave up, so the caller can truncate the output arrays.\n"
"\n"
"Raises\n"
"------\n"
"Exception\n"
"    If a Python callback (``fun`` or ``jac``) raises an exception, it is\n"
"    propagated immediately; the output arrays may be partially filled.\n");

static PyObject *drive_knots_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *ap_tspan  = NULL;
    PyArrayObject *ap_y      = NULL;
    PyArrayObject *ap_ts_out = NULL, *ap_ys_out = NULL;
    PyArrayObject *ap_rtol   = NULL, *ap_atol   = NULL;
    PyArrayObject *ap_zwork  = NULL, *ap_rwork  = NULL, *ap_iwork = NULL;
    int mf, itol, iopt;
    PyObject *fun_obj = NULL, *jac_obj = NULL, *ctx_obj = NULL;

    if (!PyArg_ParseTuple(args, "OOOiO!O!O!O!iO!O!iO!O!O!:drive_knots",
            &fun_obj,
            &jac_obj,
            &ctx_obj,
            &mf,
            &PyArray_Type, &ap_tspan,
            &PyArray_Type, &ap_y,
            &PyArray_Type, &ap_ts_out,
            &PyArray_Type, &ap_ys_out,
            &itol,
            &PyArray_Type, &ap_rtol,
            &PyArray_Type, &ap_atol,
            &iopt,
            &PyArray_Type, &ap_zwork,
            &PyArray_Type, &ap_rwork,
            &PyArray_Type, &ap_iwork))
        return NULL;

    struct zvode_callbacks cb;
    if (cb_init_from_pyobjs(&cb, fun_obj, jac_obj, ctx_obj, mf) < 0)
        return NULL;

    const int neq    = (int) PyArray_DIM(ap_y,     0);
    const int nknots = (int) PyArray_DIM(ap_tspan,  0);

    assert(neq    >= 1);
    assert(nknots >= 2);
    assert((int) PyArray_DIM(ap_ts_out, 0) == nknots);
    assert((int) PyArray_DIM(ap_ys_out, 0) == neq &&
           (int) PyArray_DIM(ap_ys_out, 1) == nknots);

    /* ---- extract raw pointers ---- */

    const int lzw = (int) PyArray_SIZE(ap_zwork);
    const int lrw = (int) PyArray_SIZE(ap_rwork);
    const int liw = (int) PyArray_SIZE(ap_iwork);

    double complex       *y      = (double complex *) PyArray_DATA(ap_y);
    const double         *tspan  = (const double *)   PyArray_DATA(ap_tspan);
    double               *ts_out = (double *)         PyArray_DATA(ap_ts_out);
    double complex       *ys_out = (double complex *) PyArray_DATA(ap_ys_out);
    double complex       *zwork  = (double complex *) PyArray_DATA(ap_zwork);
    double               *rwork  = (double *)         PyArray_DATA(ap_rwork);
    int                  *iwork  = (int *)            PyArray_DATA(ap_iwork);
    const double         *rtol   = (const double *)   PyArray_DATA(ap_rtol);
    const double         *atol   = (const double *)   PyArray_DATA(ap_atol);

    /* ---- store initial condition in output arrays ---- */

    double t  = tspan[0];
    ts_out[0] = t;
    memcpy(ys_out, y, (size_t) neq * sizeof(double complex));  /* column 0 */

    /* ---- integration loop ---- */

    const int itask  = 1;   /* advance to tout, landing exactly on it */
    int istate       = 1;   /* first call: initialise ZVODE            */
    int knots_completed = 1; /* column 0 (initial condition) always filled */

    for (int knot = 1; knot < nknots; knot++) {

        c_zvode(
            &fun_adaptor, neq, y,
            &t, tspan[knot],
            itol, rtol, atol,
            itask, &istate,
            iopt,
            zwork, lzw,
            rwork, lrw,
            iwork, liw,
            &jac_adaptor,
            mf,
            &cb
        );

        /* A Python callback raised an exception; it is already active. */
        if (cb.error) {
            assert(PyErr_Occurred());
            return NULL;
        }

        /* On ZVODE failure, stop filling output and let the caller decide. */
        if (istate != 2)
            break;

        ts_out[knot] = t;
        memcpy(ys_out + (npy_intp) knot * neq, y,
               (size_t) neq * sizeof(double complex));
        knots_completed++;
    }

    return Py_BuildValue("ii", istate, knots_completed);
}


/* ------------------------------------------------------------------ */
/* StepBuf: growable column buffer for adaptive stepping output       */
/*                                                                    */
/* Lifecycle:                                                         */
/*   stepbuf_init(&buf, neq, cap)        -- allocate; -1 on OOM      */
/*   stepbuf_append(&buf, t, y)          -- add one (t, y[neq]) pair */
/*   stepbuf_finalize(&buf, &ts, &ys)    -- produce output arrays    */
/*   stepbuf_free(&buf)                  -- release backing arrays   */
/*                                                                    */
/* The backing store is plain malloc/realloc memory, NOT NumPy        */
/* arrays.  This keeps init/append/grow/free free of any Python C    */
/* API call, so they may run with the GIL released — a prerequisite  */
/* for releasing the GIL around the whole adaptive loop.  Only        */
/* stepbuf_finalize touches the Python/NumPy C API, and it is called  */
/* exactly once after the loop exits (with the GIL held).             */
/*                                                                    */
/* On allocation failure init/append/grow return -1 without setting   */
/* a Python exception; the caller raises one once the GIL is held.    */
/*                                                                    */
/* ts is float64, length capacity; ys is complex128, length           */
/* capacity*neq, in column-major order: column k occupies             */
/* ys[k*neq .. (k+1)*neq-1].                                          */
/* ------------------------------------------------------------------ */

/* Initial column capacity.  Doubled on each overflow. */
#define STEPBUF_INIT_CAP 10

typedef struct {
    double         *ts;  /* float64, length = capacity                  */
    double complex *ys;  /* complex128, length = capacity * neq         */
                         /* column k occupies ys[k*neq .. (k+1)*neq-1]  */
    int neq;
    int size;            /* columns filled so far                       */
    int capacity;        /* allocated columns                           */
} StepBuf;

/* Allocate backing buffers.  Returns 0 on success, -1 on failure. */
static int
stepbuf_init(StepBuf *buf, int neq, int init_cap)
{
    assert(neq > 0);
    assert(init_cap > 0);

    buf->ts = (double *) malloc((size_t)init_cap * sizeof(double));
    if (!buf->ts) return -1;

    buf->ys = (double complex *)
        malloc((size_t)init_cap * neq * sizeof(double complex));
    if (!buf->ys) { free(buf->ts); buf->ts = NULL; return -1; }

    buf->neq      = neq;
    buf->size     = 0;
    buf->capacity = init_cap;
    return 0;
}

/* Release backing buffers (safe to call even after a partial init). */
static void
stepbuf_free(StepBuf *buf)
{
    free(buf->ts); buf->ts = NULL;
    free(buf->ys); buf->ys = NULL;
}

/* Double capacity via realloc, preserving existing data.
 * Returns 0 on success, -1 on failure (buf left valid and unchanged). */
static int
stepbuf_grow(StepBuf *buf)
{
    assert(buf->ts != NULL && buf->ys != NULL);
    assert(buf->capacity > 0);
    assert(buf->size == buf->capacity);  /* grow is only called when full */

    int new_cap = buf->capacity * 2;

    double *new_ts = (double *)
        realloc(buf->ts, (size_t)new_cap * sizeof(double));
    if (!new_ts) return -1;          /* buf->ts still valid at old capacity */
    buf->ts = new_ts;

    double complex *new_ys = (double complex *)
        realloc(buf->ys, (size_t)new_cap * buf->neq * sizeof(double complex));
    if (!new_ys) return -1;          /* buf->ys still valid; ts merely larger */
    buf->ys = new_ys;

    buf->capacity = new_cap;

    assert(buf->capacity == buf->size * 2);
    return 0;
}

/* Append one (t, y[neq]) pair, growing if needed.
 * Returns 0 on success, -1 on failure (allocation). */
static int
stepbuf_append(StepBuf *buf, double t, const double complex *y)
{
    assert(buf->ts != NULL && buf->ys != NULL);
    assert(buf->size <= buf->capacity);
    assert(y != NULL);
#ifndef NDEBUG
    int old_size = buf->size;
#endif

    if (buf->size == buf->capacity) {
        if (stepbuf_grow(buf) < 0)
            return -1;
    }

    buf->ts[buf->size] = t;
    memcpy(buf->ys + (size_t)buf->size * buf->neq, y,
           (size_t)buf->neq * sizeof(double complex));
    buf->size++;

    assert(buf->size == old_size + 1);
    assert(buf->size <= buf->capacity);
    return 0;
}

/* Build the output arrays from the filled portion of the raw buffer.
 *   *ts_out : shape (size,)       float64
 *   *ys_out : shape (neq, size)   complex128, F-contiguous
 * Returns 0 on success, -1 on failure (exception set).
 *
 * Allocates fresh NumPy arrays and copies the buffered data into them.
 * Uses the Python/NumPy C API, so it must be called with the GIL held;
 * buf is left unchanged (still owned by the caller) in all cases. */
static int
stepbuf_finalize(StepBuf *buf,
                 PyArrayObject **ts_out,
                 PyArrayObject **ys_out)
{
    assert(buf->ts != NULL && buf->ys != NULL);
    assert(buf->size > 0);           /* nothing to export from an empty buffer */
    assert(buf->size <= buf->capacity);

    npy_intp ts_shape[1] = { buf->size };
    PyArrayObject *ts = (PyArrayObject *)
        PyArray_EMPTY(1, ts_shape, NPY_FLOAT64, 0);
    if (!ts) return -1;
    memcpy(PyArray_DATA(ts), buf->ts, (size_t)buf->size * sizeof(double));

    /* F-contiguous (neq, size): column k at offset k*neq matches the
     * column-major layout in the raw buffer, so a flat memcpy suffices. */
    npy_intp ys_shape[2] = { buf->neq, buf->size };
    PyArrayObject *ys = (PyArrayObject *)
        PyArray_EMPTY(2, ys_shape, NPY_COMPLEX128, 1 /* fortran order */);
    if (!ys) { Py_DECREF(ts); return -1; }
    memcpy(PyArray_DATA(ys), buf->ys,
           (size_t)buf->size * buf->neq * sizeof(double complex));

    *ts_out = ts;
    *ys_out = ys;
    return 0;
}

/* ------------------------------------------------------------------ */
/* drive_adaptive                                                     */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(drive_adaptive_doc,
"drive_adaptive(fun, jac, ctx, mf, t0, t_bound, y,\n"
"               itol, rtol, atol, iopt, zwork, rwork, iwork,\n"
"               refine, allow_overshoot) -> (ts, ys, istate)\n"
"\n"
"Integrate a complex ODE system in single-step mode, collecting every\n"
"accepted step into growable output buffers.\n"
"\n"
"Accepts both Python callables and compiled C function pointers (ctypes/numba).\n"
"For compiled callbacks, pass the function pointer address as a Python int.\n"
"\n"
"Uses ITASK=5 by default (solver may not overshoot t_bound = rwork[0] =\n"
"TCRIT).  When allow_overshoot=1 uses ITASK=2 instead.\n"
"\n"
"When refine > 1, inserts (refine - 1) interpolated points inside each\n"
"accepted step via ZVINDY before appending the step endpoint.\n"
"\n"
"Parameters\n"
"----------\n"
"fun           : callable or int -- RHS.  Python: fun(t, y, dy).\n"
"               Compiled: int address, called as fun(neq, t, y, dy, ctx).\n"
"jac           : callable, int, or None -- Jacobian.  Same kind convention.\n"
"ctx           : int -- shared user-data pointer for compiled callbacks; 0 = NULL.\n"
"mf            : int -- ZVODE method flag.\n"
"t0, t_bound   : float -- start and end times.\n"
"y             : complex128 ndarray, 1-D, writable -- working state;\n"
"               must be initialised to y(t0) by the caller.\n"
"itol          : int -- tolerance mode flag (1-4).\n"
"rtol, atol    : float64 scalar or 1-D ndarray -- tolerances.\n"
"iopt          : int -- optional-input flag (0 or 1).\n"
"zwork         : complex128 ndarray, 1-D, writable -- complex workspace.\n"
"rwork         : float64 ndarray, 1-D, writable -- real workspace;\n"
"               rwork[0] must be set to t_bound (TCRIT) by the caller.\n"
"iwork         : int32 ndarray, 1-D, writable -- integer workspace.\n"
"refine        : int >= 1 -- interpolated sub-points per accepted step.\n"
"allow_overshoot : int (0 or 1) -- 0: ITASK=5 (default), 1: ITASK=2.\n"
"\n"
"Returns\n"
"-------\n"
"(ts, ys, istate) : (float64 ndarray shape (m,),\n"
"                    complex128 ndarray shape (neq, m) F-contiguous,\n"
"                    int)\n"
"    ts and ys include the initial condition at index 0.  On solver\n"
"    failure istate < 0 and the arrays contain all points up to and\n"
"    including the last successful step.\n"
"\n"
"Raises\n"
"------\n"
"Exception\n"
"    Propagated immediately if fun or jac raises inside a callback.\n"
"RuntimeError\n"
"    If ZVINDY fails during refinement interpolation.\n");

static PyObject *
drive_adaptive_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *ap_y     = NULL;
    PyArrayObject *ap_rtol  = NULL, *ap_atol  = NULL;
    PyArrayObject *ap_zwork = NULL, *ap_rwork = NULL, *ap_iwork = NULL;
    double t0, t_bound;
    int mf, itol, iopt, refine, allow_overshoot;
    PyObject *fun_obj = NULL, *jac_obj = NULL, *ctx_obj = NULL;

    if (!PyArg_ParseTuple(args, "OOOiddO!iO!O!iO!O!O!ii:drive_adaptive",
            &fun_obj,
            &jac_obj,
            &ctx_obj,
            &mf,
            &t0, &t_bound,
            &PyArray_Type, &ap_y,
            &itol,
            &PyArray_Type, &ap_rtol,
            &PyArray_Type, &ap_atol,
            &iopt,
            &PyArray_Type, &ap_zwork,
            &PyArray_Type, &ap_rwork,
            &PyArray_Type, &ap_iwork,
            &refine,
            &allow_overshoot))
        return NULL;

    struct zvode_callbacks cb;
    if (cb_init_from_pyobjs(&cb, fun_obj, jac_obj, ctx_obj, mf) < 0)
        return NULL;

    assert(refine >= 1);

    const int neq = (int) PyArray_DIM(ap_y, 0);
    const int lzw = (int) PyArray_SIZE(ap_zwork);
    const int lrw = (int) PyArray_SIZE(ap_rwork);
    const int liw = (int) PyArray_SIZE(ap_iwork);

    double complex       *y     = (double complex *) PyArray_DATA(ap_y);
    double complex       *zwork = (double complex *) PyArray_DATA(ap_zwork);
    double               *rwork = (double *)         PyArray_DATA(ap_rwork);
    int                  *iwork = (int *)            PyArray_DATA(ap_iwork);
    const double         *rtol  = (const double *)   PyArray_DATA(ap_rtol);
    const double         *atol  = (const double *)   PyArray_DATA(ap_atol);

    /* Scratch buffer for ZVINDY interpolated output; allocated once if
     * refine > 1, managed by Python's allocator. */
    double complex *dky = NULL;
    if (refine > 1) {
        dky = PyMem_New(double complex, neq);
        if (!dky) { PyErr_NoMemory(); return NULL; }
    }

    StepBuf buf = {0};
    if (stepbuf_init(&buf, neq, STEPBUF_INIT_CAP) < 0) {
        PyErr_NoMemory();
        goto cleanup;
    }

    /* Store initial condition. */
    double t = t0;
    if (stepbuf_append(&buf, t, y) < 0) {
        PyErr_NoMemory();
        goto cleanup;
    }

    const int itask   = allow_overshoot ? 2 : 5;
    int       istate  = 1;
    assert(t_bound != t0);  /* Python layer guarantees strict monotonicity of tspan */
    double    direction = (t_bound > t0) ? 1.0 : -1.0;

    /* Deferred error state.  The loop below performs no Python C API calls
     * (with compiled callbacks); any failure records its cause here and
     * leaves the loop, and the exception is raised afterwards once the GIL
     * is guaranteed held.  This is what makes the loop safe to run with the
     * GIL released in a later step. */
    int    alloc_failed = 0;     /* StepBuf ran out of memory               */
    int    zvindy_iflag = 0;     /* nonzero => ZVINDY interpolation failure  */
    double zvindy_t     = 0.0;   /* t at which ZVINDY failed (for message)   */

    while (direction * (t_bound - t) > 0.0) {
        double t_old = t;

        c_zvode(
            &fun_adaptor, neq, y,
            &t, t_bound,
            itol, rtol, atol,
            itask, &istate,
            iopt,
            zwork, lzw,
            rwork, lrw,
            iwork, liw,
            &jac_adaptor,
            mf,
            &cb
        );

        if (cb.error)
            break;  /* callback raised — exception already pending */

        if (istate < 0)
            break;  /* solver error — return what we have so far */

        if (refine > 1) {
            /* Nordsieck array yh lives at zwork[0..neq*(nq+1)-1], ldyh = neq.
             * h == hu immediately after an accepted step. */
            int    nq = iwork[13];    /* NQU: IWORK(14), order last used   */
            double hu = rwork[10];    /* HU:  RWORK(11), step size last used */
            struct zvode_step_t step = { .h = hu, .tn = t, .hu = hu, .nq = nq };

            for (int i = 1; i < refine; i++) {
                double t_i = t_old + (double)i * (t - t_old) / refine;
                int iflag = c_zvindy(neq, t_i, zwork, neq, 0, dky, &step);
                if (iflag != 0) {
                    zvindy_iflag = iflag;
                    zvindy_t     = t_i;
                    break;
                }
                if (stepbuf_append(&buf, t_i, dky) < 0) {
                    alloc_failed = 1;
                    break;
                }
            }
            if (zvindy_iflag != 0 || alloc_failed)
                break;  /* refinement failed — escape the step loop */
        }

        if (stepbuf_append(&buf, t, y) < 0) {
            alloc_failed = 1;
            break;
        }
    }

    /* Raise any deferred error now (GIL held). */
    if (cb.error) {
        assert(PyErr_Occurred());
        goto cleanup;
    }
    if (alloc_failed) {
        PyErr_NoMemory();
        goto cleanup;
    }
    if (zvindy_iflag != 0) {
        PyErr_Format(PyExc_RuntimeError,
            "ZVINDY failed (iflag=%d) interpolating at t=%.17g",
            zvindy_iflag, zvindy_t);
        goto cleanup;
    }

    /* Build the final output arrays from the filled portion of the buffer. */
    PyArrayObject *ts_out = NULL, *ys_out = NULL;
    if (stepbuf_finalize(&buf, &ts_out, &ys_out) < 0)
        goto cleanup;

    PyMem_Free(dky);
    stepbuf_free(&buf);

    /* "N" steals the references — no explicit Py_DECREF needed. */
    return Py_BuildValue("(NNi)", ts_out, ys_out, istate);

cleanup:
    PyMem_Free(dky);
    stepbuf_free(&buf);
    return NULL;
}


static struct PyMethodDef zvode_module_methods[] = {
    {"zvode",          zvode_py,          METH_VARARGS, zvode_doc},
    {"zvindy",         zvindy_py,         METH_VARARGS, zvindy_doc},
    {"drive_knots",    drive_knots_py,    METH_VARARGS, drive_knots_doc},
    {"drive_adaptive", drive_adaptive_py, METH_VARARGS, drive_adaptive_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_zvode",
    "ZVODE - Complex ODE Solver",
    -1,
    zvode_module_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__zvode(void) {

    import_array();   /* NumPy C-API; expands to `return NULL;` on failure */

    PyObject *m;
    if (!(m = PyModule_Create(&module_def))) {
        return NULL;
    }
    return m;
}
