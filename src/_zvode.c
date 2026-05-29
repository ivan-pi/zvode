#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <complex.h>

#ifndef ZVODE_DEBUG
#define ZVODE_DEBUG 0
#endif

#include <stdio.h> // For debugging only
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "zvode.h"

/* ------------------------------------------------------------------ */
/* Debug helpers (compile with -DZVODE_DEBUG to enable)               */
/* ------------------------------------------------------------------ */

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

/* Convenience wrappers for the common cases. */
static inline int
check_array_1d(PyArrayObject *ap, const char *name, int typenum)
{
    return check_array(ap, name, 1, typenum, 'C');
}

static inline int
check_writable(PyArrayObject *ap, const char *name)
{
    if (!PyArray_ISWRITEABLE(ap)) {
        PyErr_Format(PyExc_ValueError, "zvode: %s must be writable", name);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Callback plumbing                                                  */
/* ------------------------------------------------------------------ */

struct zvode_callbacks {
    PyObject *fun;
    PyObject *jac;
    int jac_is_banded; /* 1 when MITER=4 (abs(mf)%10 == 4), 0 otherwise */
    int error;
    // TODO: add zewset and zwnorm in the future
};

static void fun_adaptor(
        int neq,
        double t,
        const double complex y[],
        double complex dy[],
        void *ctx) {

    struct zvode_callbacks *cb = (struct zvode_callbacks *) ctx;
    assert(cb != NULL);
    assert(cb->fun != NULL);

    const npy_intp dims[1] = { neq };

    /* Wrap the solver-owned buffers as NumPy views (no copy). */
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_COMPLEX128, (void *) y);
    if (ap_y == NULL) {
        cb->error = 1;
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    const npy_intp dims_dy[1] = { neq };

    PyArrayObject *ap_dy =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_dy, NPY_COMPLEX128, dy);
    if (ap_dy == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }

    /* fun(t, y, dy): Python writes the derivative into dy in place. */
    PyObject *res = PyObject_CallFunction(cb->fun, "dOO", t,
        (PyObject *) ap_y,
        (PyObject *) ap_dy);

    Py_DECREF(ap_y);
    Py_DECREF(ap_dy);
    if (res == NULL) {
        cb->error = 1;
        return;
    }
    Py_DECREF(res);
}

static void jac_adaptor(
        int neq,
        double t,
        const double complex y[],
        int ml, int mu,
        double complex pd[],
        int nrowpd,
        void *ctx) {

    struct zvode_callbacks *cb = (struct zvode_callbacks *) ctx;
    assert(cb != NULL);
    assert(cb->jac != NULL && cb->jac != Py_None);
    assert(nrowpd >= (cb->jac_is_banded ? ml+mu+1 : neq));

    const npy_intp dims_y[1] = { (npy_intp) neq };
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_y, NPY_COMPLEX128, (void *) y);
    if (ap_y == NULL) {
        cb->error = 1;
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    /* PD is column-major with leading dimension NROWPD, exactly the layout
     * ZVODE/LAPACK expect.  Expose it as an F-contiguous (nrowpd, neq) view
     * so that pd[i, j] in Python is PD(i+1, j+1) in Fortran. */

    const npy_intp dims_pd[2] = { (npy_intp) nrowpd, (npy_intp) neq };

    /* Explicitly define strides to achieve Fortran contiguity */
    const npy_intp strides_pd[2] = {
        sizeof(double complex),
        (npy_intp) ((size_t) nrowpd * sizeof(double complex))
    };

    PyArrayObject *ap_pd = (PyArrayObject *) PyArray_New(
        &PyArray_Type, 2, dims_pd, NPY_COMPLEX128,
        strides_pd, (void *)pd, 0, NPY_ARRAY_WRITEABLE, NULL
    );
    if (ap_pd == NULL) {
        Py_DECREF(ap_y);
        cb->error = 1;
        return;
    }


    PyObject *res;
    if (cb->jac_is_banded) {
        /* jac(t, y, pd, ml, mu): Python writes the Jacobian into pd in place. */
        res = PyObject_CallFunction(cb->jac, "dOOii", t,
            (PyObject *) ap_y, (PyObject *) ap_pd, ml, mu);
    } else {
        /* jac(t, y, pd): Python writes the Jacobian into pd in place. */
        res = PyObject_CallFunction(cb->jac, "dOO", t,
            (PyObject *) ap_y, (PyObject *) ap_pd);
    }
    Py_DECREF(ap_y);
    Py_DECREF(ap_pd);
    if (res == NULL) {
        cb->error = 1;
        return;
    }
    Py_DECREF(res);
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
"contiguous arrays of dtype complex128, complex128, float64 and int32.\n"
"`fun` is called as fun(t, y, dy) and must fill `dy`; `jac` (or None) is\n"
"called as jac(t, y, pd).  Returns the advanced time and the ZVODE istate.\n");

static PyObject* zvode_py(PyObject* self, PyObject *args) {

    PyArrayObject *ap_y = NULL, *ap_rtol = NULL, *ap_atol = NULL;
    PyArrayObject *ap_zwork = NULL, *ap_rwork = NULL, *ap_iwork = NULL;

    double t, tout;
    int itol, itask, istate, iopt, mf;

    // Container for the actual Python callbacks
    struct zvode_callbacks cb = { .fun = NULL, .jac = NULL,
        .jac_is_banded = 0, .error = 0, };

    if (!PyArg_ParseTuple(args,"OO!ddiO!O!iiiO!O!O!Oi:zvode",
       &cb.fun,
       &PyArray_Type, &ap_y,
       &t, &tout, &itol,
       &PyArray_Type, &ap_rtol,
       &PyArray_Type, &ap_atol,
       &itask, &istate, &iopt,
       &PyArray_Type, &ap_zwork,
       &PyArray_Type, &ap_rwork,
       &PyArray_Type, &ap_iwork,
       &cb.jac, &mf)) {
        return NULL;
    }

    assert(ap_y);
    assert(ap_rtol);
    assert(ap_atol);
    assert(ap_zwork);
    assert(ap_rwork);
    assert(ap_iwork);
    assert(cb.fun);
    assert(cb.jac); // should be Py_None or a callable
    assert(itask > 0);
    assert(mf > 0);

    if (ZVODE_DEBUG) {
        dump_zvode_args(cb.fun, ap_y, t, tout, itol, ap_rtol, ap_atol,
                        itask, istate, iopt, ap_zwork, ap_rwork, ap_iwork,
                        cb.jac, mf);
    }

    const int neq = (int) PyArray_DIM(ap_y, 0);
    if (neq <= 0) {
        PyErr_SetString(PyExc_ValueError, "zvode: y must be non-empty");
        return NULL;
    }

    const int miter = abs(mf) % 10;
    cb.jac_is_banded = (miter == 4);

    // Upon initialization of ZVODE, do stringent type checks, but skip
    // them otherwise, because they are expensive.
    // The caller should not change any of the arrays when the integration
    // is active.

    if (istate == 1) {

        if (!PyCallable_Check(cb.fun)) {
            PyErr_SetString(PyExc_TypeError, "zvode: fun must be callable");
            return NULL;
        }

        if (cb.jac != Py_None && !PyCallable_Check(cb.jac)) {
            PyErr_SetString(PyExc_TypeError, "zvode: jac must be callable or None");
            return NULL;
        }

        /* Validate dtype, dimensionality, contiguity, and writability for every
         * array argument.*/
        if (!check_array_1d(ap_y,     "y",     NPY_COMPLEX128) || !check_writable(ap_y,     "y"))     return NULL;
        if (!check_array_1d(ap_zwork, "zwork", NPY_COMPLEX128) || !check_writable(ap_zwork, "zwork")) return NULL;
        if (!check_array_1d(ap_rwork, "rwork", NPY_FLOAT64)    || !check_writable(ap_rwork, "rwork")) return NULL;
        if (!check_array_1d(ap_iwork, "iwork", NPY_INT32)      || !check_writable(ap_iwork, "iwork")) return NULL;
        if (!check_array_1d(ap_rtol,  "rtol",  NPY_FLOAT64))  return NULL;
        if (!check_array_1d(ap_atol,  "atol",  NPY_FLOAT64))  return NULL;

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
    PyObject *res;
    if (!(res = Py_BuildValue("di",t,istate))) {
        return NULL;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/* zvindy (interpolation)                                             */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(zvindy_doc,
"zvindy(t, k, yh, h, tn, hu, dky) -> iflag\n"
"\n"
"Interpolate the K-th derivative of y at time T using the Nordsieck array.\n"
"\n"
"Parameters\n"
"----------\n"
"t   : float  -- interpolation time; must lie in [tn - hu, tn].\n"
"k   : int    -- derivative order; must satisfy 0 <= k <= yh.shape[1] - 1.\n"
"yh  : complex128 ndarray, shape (n, nq+1), F-contiguous -- Nordsieck array.\n"
"h   : float  -- HCUR, the step size the Nordsieck array is scaled to.\n"
"tn  : float  -- TCUR, the current solver time.\n"
"hu  : float  -- HU, the last successfully used step size.\n"
"dky : complex128 ndarray, 1-D length n, writable -- receives the result.\n"
"\n"
"Raises\n"
"------\n"
"ValueError -- if k is out of range or t is outside [tn - hu, tn].\n");

static PyObject* zvindy_py(PyObject* self, PyObject *args) {

    PyArrayObject *ap_yh = NULL, *ap_dky = NULL;
    double t, h, tn, hu;
    int k;

    if (!PyArg_ParseTuple(args, "diO!dddO!:zvindy",
            &t, &k,
            &PyArray_Type, &ap_yh,
            &h, &tn, &hu,
            &PyArray_Type, &ap_dky)) {
        return NULL;
    }

    if (!check_array(ap_yh,  "yh",  2, NPY_COMPLEX128, 'F')) return NULL;

    if (!check_array_1d(ap_dky, "dky", NPY_COMPLEX128)) return NULL;
    if (!check_writable(ap_dky, "dky"))                 return NULL;

    const int n    = (int) PyArray_DIM(ap_dky, 0);      /* number of equations */
    const int ldyh = (int) PyArray_DIM(ap_yh, 0);       /* leading dimension (>= n) */
    const int nq   = (int) PyArray_DIM(ap_yh, 1) - 1;   /* current order     */

    if (ldyh < n) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: yh leading dimension (%d) must be >= len(dky) (%d)",
            ldyh, n);
        return NULL;
    }
    if (k < 0 || k > nq) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: k must satisfy 0 <= k <= %d (got %d)", nq, k);
        return NULL;
    }

    double complex *yh  = (double complex *) PyArray_DATA(ap_yh);
    double complex *dky = (double complex *) PyArray_DATA(ap_dky);

    int iflag = 0;
    zvindy(t, k, yh, ldyh, dky, &iflag);

    if (iflag == -1) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: k=%d is out of range [0, nq=%d] (Fortran IFLAG=-1)", k, nq);
        return NULL;
    }
    if (iflag == -2) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: t=%.17g is outside the valid interval [tn-hu, tn] "
            "(Fortran IFLAG=-2)", t);
        return NULL;
    }

    Py_RETURN_NONE;
}

static struct PyMethodDef zvode_module_methods[] = {
    {"zvode", zvode_py, METH_VARARGS, zvode_doc},
    {"zvindy", zvindy_py, METH_VARARGS, zvindy_doc},
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
