#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <complex.h>

#include <stdio.h> // For debugging only

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "zvode.h"

/* ZVODE's IWORK is Fortran default INTEGER, declared as `int` in the C
 * header.  We expose it to Python as int32, so the two must agree. */
_Static_assert(sizeof(int) == 4, "iwork bridging assumes a 32-bit C int");

/* ------------------------------------------------------------------ */
/* Callback plumbing                                                  */
/* ------------------------------------------------------------------ */

struct zvode_callbacks {
    PyObject *fun;
    PyObject *jac;
    // TODO: add zewset and zwnorm in the future
};

static void fun_adaptor(
        int neq,
        double t,
        double complex y[],
        double complex dy[],
        void *ctx) {

    struct zvode_callbacks *cb = (struct zvode_callbacks *) ctx;
    assert(cb != NULL);
    assert(cb->fun != NULL);

    printf("In fun_adaptor.\n");
    fprintf(stderr, "fun_adaptor: y=%p  dy=%p  neq=%d\n",
            (void*)y, (void*)dy, neq);
    fflush(stderr);

    const npy_intp dims[1] = { neq };

    /* Wrap the solver-owned buffers as NumPy views (no copy). */
    PyArrayObject *ap_y =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_COMPLEX128, y);
    assert(ap_y);
    if (ap_y == NULL) {
        return;
    }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    printf("y is ready.\n");


    const npy_intp dims_dy[1] = { neq };

    PyArrayObject *ap_dy =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, dims_dy, NPY_COMPLEX128, dy);
    assert(ap_dy);
    if (ap_dy == NULL) {
        return;
    }

    printf("dy is ready.\n");
    assert(ap_y);
    assert(ap_dy);
    printf("calling fun at t = %f\n", t);

    fprintf(stderr, "DEBUG: ap_y=%p (rc=%ld)  ap_dy=%p (rc=%ld)\n",
            (void*)ap_y, Py_REFCNT(ap_y),
            (void*)ap_dy, Py_REFCNT(ap_dy));
    fflush(stderr);
    /* fun(t, y, dy): Python writes the derivative into dy in place. */
    PyObject *res = PyObject_CallFunction(cb->fun, "dOO", t, ap_y, ap_dy);
    fprintf(stderr, "res = %p, exception set = %d\n",
            (void*)res, PyErr_Occurred() != NULL);
    fflush(stderr);
    printf("called fun at t = %f\n", t);

    Py_DECREF(ap_y);     // missing: ap_y is leaking every call
    Py_DECREF(ap_dy);    // missing: ap_dy is leaking every call
}

static void jac_adaptor(
        int neq,
        double t,
        double complex y[],
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

    PyArrayObject *ap_pd;

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

    printf("About to parse args.\n");

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
    assert(cb.jac); // could be Python None

    printf("Args are parsed.\n");

    if (istate == 1) {
        // Initialization of ZVODE
        // TODO: validate arguments for type and contiguity
        //   on future calls we assume that everything is okay
    }

   const int neq = (int) PyArray_DIM(ap_y, 0); assert(neq > 0);
   const int lzw = (int) PyArray_SIZE(ap_zwork);
   const int lrw = (int) PyArray_SIZE(ap_rwork);
   const int liw = (int) PyArray_SIZE(ap_iwork);

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
    assert(lzw > 0);
    assert(lrw > 0);
    assert(liw > 0);

    assert(t != tout);
    assert(istate > 0);
    assert(itask > 0);
    assert(mf > 0);

    printf("All pointers are ready.\n");

//      SUBROUTINE ZVODE (F, NEQ, Y, T, TOUT, ITOL, RTOL, ATOL, ITASK,
//     1            ISTATE, IOPT, ZWORK, LZW, RWORK, LRW, IWORK, LIW,
//     2            JAC, MF, CTX) BIND(C,name="zvode")

    // Call the Fortran integrator
    zvode(
        &fun_adaptor,
        neq, y, &t, tout,
        itol, rtol, atol,
        itask, &istate,
        iopt, zwork, lzw, rwork, lrw, iwork, liw,
        &jac_adaptor,
        mf,
        (void *) &cb
    );

    printf("Returned from ZVODE.\n");

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
