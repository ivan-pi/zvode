#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <complex.h>

#define ZVODE_DEBUG
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
/* Callback plumbing                                                  */
/* ------------------------------------------------------------------ */

struct zvode_callbacks {
    PyObject *fun;
    PyObject *jac;
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

#if 0
    fprintf(stderr, "fun_adaptor: y=%p  dy=%p  neq=%d\n",
            (void*)y, (void*)dy, neq);
    fflush(stderr);
#endif
    const npy_intp dims[1] = { neq };

    /* Wrap the solver-owned buffers as NumPy views (no copy). */
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_COMPLEX128, (double complex *) y);
    if (ap_y == NULL) {
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    const npy_intp dims_dy[1] = { neq };

    PyArrayObject *ap_dy =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_dy, NPY_COMPLEX128, dy);
    if (ap_dy == NULL) {
        return;
    }

#if 0
    printf("calling fun at t = %f\n", t);
    fprintf(stderr, "DEBUG: ap_y=%p (rc=%ld)  ap_dy=%p (rc=%ld)\n",
            (void*)ap_y, Py_REFCNT(ap_y),
            (void*)ap_dy, Py_REFCNT(ap_dy));
    fflush(stderr);

    /* 1. Check if the function pointer matches the original */
    void *expected_fun = (void *) cb->fun;

    /* 2. Read the raw CPU Stack Pointer */
    void *sp = __builtin_frame_address(0);
    int is_aligned = ((uintptr_t)sp % 16 == 0);

    fprintf(stderr, ">>> DIAGNOSTIC: cb->fun = %p | SP = %p | ALIGNED = %s\n",
            expected_fun, sp, is_aligned ? "YES" : "NO");
    fflush(stderr);
#endif

    /* fun(t, y, dy): Python writes the derivative into dy in place. */
    assert(cb->fun && ap_y && ap_dy);
    PyObject *res = PyObject_CallFunction(cb->fun, "dOO", t, (PyObject *)ap_y, (PyObject *)ap_dy);

#if 0
    fprintf(stderr, "res = %p, exception set = %d\n",
            (void*)res, PyErr_Occurred() != NULL);
    fflush(stderr);
    printf("called fun at t = %f\n", t);
#endif

    Py_DECREF(ap_y);     // missing: ap_y is leaking every call
    Py_DECREF(ap_dy);    // missing: ap_dy is leaking every call
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
    assert(cb->jac != NULL);

    printf("In Jacobian func.\n");

    const npy_intp dims_y[1] = { (npy_intp) neq };
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_y, NPY_COMPLEX128, y);
    assert(ap_y);
    if (ap_y == NULL) {
        return;
    }
    PyArray_CLEARFLAGS(ap_y,NPY_ARRAY_WRITEABLE);

    /* PD is column-major with leading dimension NROWPD, exactly the layout
     * ZVODE/LAPACK expect.  Expose it as an F-contiguous (nrowpd, neq) view
     * so that pd[i, j] in Python is PD(i+1, j+1) in Fortran. */

    PyArrayObject *ap_pd = NULL;

    // TODO: build numpy compatible array objects for y and pd
    // the arrays pd has dimension nrowpd by neq, but it might represent
    // either a dense or a banded array (including padding)

    // TODO: use ml and mu in the callback
    PyObject *res = PyObject_CallFunction(cb->jac, "dOO", t, ap_y, ap_pd);

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
    struct zvode_callbacks cb = { .fun = NULL, .jac = NULL };

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

    assert(tout >= t);
    assert(ap_y);
    assert(ap_rtol);
    assert(ap_atol);
    assert(ap_zwork);
    assert(ap_rwork);
    assert(ap_iwork);
    assert(cb.fun);
    assert(cb.jac); // should be Py_None or a callable
    assert(istate > 0);
    assert(itask > 0);
    assert(mf > 0);

#ifdef ZVODE_DEBUG
    dump_zvode_args(cb.fun, ap_y, t, tout, itol, ap_rtol, ap_atol,
                    itask, istate, iopt, ap_zwork, ap_rwork, ap_iwork,
                    cb.jac, mf);
#endif

    if (istate == 1) {
        // Initialization of ZVODE
        // TODO: validate arguments for type and contiguity
        //   on future calls we assume that everything is okay
    }

   const int neq = (int) PyArray_DIM(ap_y, 0); assert(neq > 0);
   const int lzw = (int) PyArray_SIZE(ap_zwork); assert(lzw > 0);
   const int lrw = (int) PyArray_SIZE(ap_rwork); assert(lrw > 0);
   const int liw = (int) PyArray_SIZE(ap_iwork); assert(liw > 0);

    double complex *y     = (double complex *) PyArray_DATA(ap_y);
    double complex *zwork = (double complex *) PyArray_DATA(ap_zwork);
    double         *rwork = (double *)         PyArray_DATA(ap_rwork);
    int            *iwork = (int *)            PyArray_DATA(ap_iwork);
    const double   *rtol  = (const double *)   PyArray_DATA(ap_rtol);
    const double   *atol  = (const double *)   PyArray_DATA(ap_atol);

    assert(y);
    assert(zwork);
    assert(rwork);
    assert(iwork);
    assert(rtol);
    assert(atol);

    // Call the actual "C" integrator
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

    PyObject *res;
    if (!(res = Py_BuildValue("di",t,istate))) {
        return NULL;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/* zvindy (interpolation) - not implemented yet                       */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(zvindy_doc,
"zvindy(...) -> (not implemented)\n");

static PyObject* zvindy_py(PyObject* self, PyObject *args) {
    PyErr_SetString(PyExc_NotImplementedError,
        "zvindy (dense-output interpolation) is not implemented yet.");
    return NULL;
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
