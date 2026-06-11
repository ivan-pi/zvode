/* _bdf.c -- CPython extension wrapping the Fortran BDF integrator.
 *
 * Exposes a single extension type, `_bdf.Solver`, that owns an opaque Fortran
 * handle and marshals Python `fun(t, y)` / `jac(t, y)` callbacks across the C
 * ABI declared in bdf.h.  The friendly, NumPy-flavoured API lives one level up
 * in the `pybdf` package; this module stays deliberately thin.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>

#include "bdf.h"

typedef struct {
    PyObject_HEAD
    void     *handle;   /* Fortran solver handle               */
    PyObject *fun;      /* Python rhs callable                 */
    PyObject *jac;      /* Python Jacobian callable or NULL    */
    int       n;        /* number of equations                 */
    int       error;    /* set when a callback raised          */
} SolverObject;

/* ------------------------------------------------------------------ */
/* Callback trampolines (invoked by the Fortran core through bdf.h).   */
/* ------------------------------------------------------------------ */

/* Wrap a raw buffer as a read-only 1-D float64 array. */
static PyObject *view_1d(const double *data, int n) {
    npy_intp dims[1] = {n};
    PyObject *a = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *)data);
    if (a) PyArray_CLEARFLAGS((PyArrayObject *)a, NPY_ARRAY_WRITEABLE);
    return a;
}

static void rhs_trampoline(int n, double t, const double *y,
                           double *f, void *ctx) {
    SolverObject *s = (SolverObject *)ctx;
    int i;
    if (s->error) { for (i = 0; i < n; ++i) f[i] = 0.0; return; }

    PyObject *ya = view_1d(y, n);
    PyObject *res = ya ? PyObject_CallFunction(s->fun, "dO", t, ya) : NULL;
    Py_XDECREF(ya);

    PyArrayObject *ra = NULL;
    if (res) ra = (PyArrayObject *)PyArray_FROM_OTF(res, NPY_DOUBLE,
                                                    NPY_ARRAY_IN_ARRAY);
    if (ra && PyArray_SIZE(ra) == n) {
        const double *p = (const double *)PyArray_DATA(ra);
        for (i = 0; i < n; ++i) f[i] = p[i];
    } else {
        if (ra && !PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError,
                            "rhs callback returned wrong-sized array");
        s->error = 1;
        /* NaN forces the integrator to treat the step as non-finite. */
        for (i = 0; i < n; ++i) f[i] = Py_NAN;
    }
    Py_XDECREF(ra);
    Py_XDECREF(res);
}

static void jac_trampoline(int n, double t, const double *y,
                           int ml, int mu, double *pd, int ldpd, void *ctx) {
    SolverObject *s = (SolverObject *)ctx;
    int i, j, r;
    if (s->error) return;

    PyObject *ya = view_1d(y, n);
    PyObject *res = ya ? PyObject_CallFunction(s->jac, "dO", t, ya) : NULL;
    Py_XDECREF(ya);

    PyArrayObject *ja = NULL;
    if (res) ja = (PyArrayObject *)PyArray_FROM_OTF(res, NPY_DOUBLE,
                                                    NPY_ARRAY_IN_ARRAY);
    if (!ja) { s->error = 1; Py_XDECREF(res); return; }

    const double *p = (const double *)PyArray_DATA(ja);
    if (ml < 0) {
        /* Dense: Python J is C-order (n, n); pd is column-major (n, n). */
        if (PyArray_SIZE(ja) == (npy_intp)n * n) {
            for (j = 0; j < n; ++j)
                for (i = 0; i < n; ++i)
                    pd[i + (npy_intp)j * ldpd] = p[(npy_intp)i * n + j];
        } else { s->error = 1; }
    } else {
        /* Banded: Python J is C-order (ml+mu+1, n); pd column-major same. */
        int ldj = ml + mu + 1;
        if (PyArray_SIZE(ja) == (npy_intp)ldj * n) {
            for (j = 0; j < n; ++j)
                for (r = 0; r < ldj; ++r)
                    pd[r + (npy_intp)j * ldpd] = p[(npy_intp)r * n + j];
        } else { s->error = 1; }
    }
    if (s->error && !PyErr_Occurred())
        PyErr_SetString(PyExc_ValueError, "jac callback returned wrong shape");
    Py_XDECREF(ja);
    Py_XDECREF(res);
}

/* ------------------------------------------------------------------ */
/* Type lifecycle                                                      */
/* ------------------------------------------------------------------ */

static PyObject *Solver_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    (void)args; (void)kw;
    SolverObject *self = (SolverObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->handle = bdf_create();
    if (!self->handle) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    self->fun = self->jac = NULL;
    self->n = 0;
    self->error = 0;
    return (PyObject *)self;
}

static void Solver_dealloc(SolverObject *self) {
    if (self->handle) bdf_destroy(self->handle);
    Py_XDECREF(self->fun);
    Py_XDECREF(self->jac);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* setup(fun, t0, y0, t_bound, rtol, atol, jac_mode, jac,
 *       ml, mu, max_step, first_step, reuse_jac) */
static PyObject *Solver_setup(SolverObject *self, PyObject *args) {
    PyObject *fun, *y0_obj, *atol_obj, *jac_obj;
    double t0, t_bound, rtol, max_step, first_step;
    int jac_mode, ml, mu, reuse_jac;

    if (!PyArg_ParseTuple(args, "OdOddOiOiiddi",
                          &fun, &t0, &y0_obj, &t_bound, &rtol, &atol_obj,
                          &jac_mode, &jac_obj, &ml, &mu,
                          &max_step, &first_step, &reuse_jac))
        return NULL;

    PyArrayObject *y0 = (PyArrayObject *)PyArray_FROM_OTF(
        y0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *atol = (PyArrayObject *)PyArray_FROM_OTF(
        atol_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!y0 || !atol) { Py_XDECREF(y0); Py_XDECREF(atol); return NULL; }

    int n = (int)PyArray_SIZE(y0);
    if ((int)PyArray_SIZE(atol) != n) {
        PyErr_SetString(PyExc_ValueError, "atol must have the same length as y0");
        Py_DECREF(y0); Py_DECREF(atol);
        return NULL;
    }

    Py_INCREF(fun);
    Py_XSETREF(self->fun, fun);
    if (jac_obj != Py_None) {
        Py_INCREF(jac_obj);
        Py_XSETREF(self->jac, jac_obj);
    } else {
        Py_CLEAR(self->jac);
    }
    self->n = n;
    self->error = 0;

    bdf_jac_fn jcb = (self->jac != NULL) ? jac_trampoline : NULL;
    bdf_init(self->handle, n, t0, (const double *)PyArray_DATA(y0), t_bound,
             rhs_trampoline, rtol, (const double *)PyArray_DATA(atol),
             jac_mode, jcb, ml, mu, max_step, first_step, reuse_jac, self);

    Py_DECREF(y0);
    Py_DECREF(atol);

    if (self->error) return NULL;  /* a callback raised during init's first eval */
    Py_RETURN_NONE;
}

static PyObject *Solver_step(SolverObject *self, PyObject *Py_UNUSED(ig)) {
    int code;
    self->error = 0;
    bdf_step(self->handle, &code);
    if (self->error) return NULL;
    return PyLong_FromLong(code);
}

static PyObject *Solver_integrate(SolverObject *self, PyObject *Py_UNUSED(ig)) {
    int code;
    self->error = 0;
    bdf_integrate(self->handle, &code);
    if (self->error) return NULL;
    return PyLong_FromLong(code);
}

static PyObject *Solver_set_t_bound(SolverObject *self, PyObject *arg) {
    double tb = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) return NULL;
    bdf_set_t_bound(self->handle, tb);
    Py_RETURN_NONE;
}

static PyObject *Solver_get_t(SolverObject *self, PyObject *Py_UNUSED(ig)) {
    return PyFloat_FromDouble(bdf_get_t(self->handle));
}

static PyObject *Solver_get_y(SolverObject *self, PyObject *Py_UNUSED(ig)) {
    npy_intp dims[1] = {self->n};
    PyObject *y = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!y) return NULL;
    bdf_get_y(self->handle, self->n, (double *)PyArray_DATA((PyArrayObject *)y));
    return y;
}

static PyObject *Solver_get_stats(SolverObject *self, PyObject *Py_UNUSED(ig)) {
    int nfev, njev, nlu, nsteps;
    bdf_get_stats(self->handle, &nfev, &njev, &nlu, &nsteps);
    return Py_BuildValue("{s:i,s:i,s:i,s:i}",
                         "nfev", nfev, "njev", njev,
                         "nlu", nlu, "nsteps", nsteps);
}

static PyMethodDef Solver_methods[] = {
    {"setup",       (PyCFunction)Solver_setup,       METH_VARARGS, "Configure the solver."},
    {"step",        (PyCFunction)Solver_step,        METH_NOARGS,  "Take one internal step."},
    {"integrate",   (PyCFunction)Solver_integrate,   METH_NOARGS,  "Integrate to t_bound."},
    {"set_t_bound", (PyCFunction)Solver_set_t_bound, METH_O,       "Set the integration boundary."},
    {"get_t",       (PyCFunction)Solver_get_t,       METH_NOARGS,  "Current time."},
    {"get_y",       (PyCFunction)Solver_get_y,       METH_NOARGS,  "Current state (copy)."},
    {"get_stats",   (PyCFunction)Solver_get_stats,   METH_NOARGS,  "Evaluation counters."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SolverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_bdf.Solver",
    .tp_basicsize = sizeof(SolverObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Stateful BDF integrator (thin wrapper over the Fortran core).",
    .tp_new = Solver_new,
    .tp_dealloc = (destructor)Solver_dealloc,
    .tp_methods = Solver_methods,
};

static PyModuleDef bdfmodule = {
    PyModuleDef_HEAD_INIT, "_bdf",
    "CPython bindings for the Fortran BDF integrator.", -1, NULL,
};

PyMODINIT_FUNC PyInit__bdf(void) {
    import_array();
    if (PyType_Ready(&SolverType) < 0) return NULL;

    PyObject *m = PyModule_Create(&bdfmodule);
    if (!m) return NULL;

    Py_INCREF(&SolverType);
    if (PyModule_AddObject(m, "Solver", (PyObject *)&SolverType) < 0) {
        Py_DECREF(&SolverType);
        Py_DECREF(m);
        return NULL;
    }

    PyModule_AddIntConstant(m, "JAC_FD", BDF_JAC_FD);
    PyModule_AddIntConstant(m, "JAC_USER", BDF_JAC_USER);
    PyModule_AddIntConstant(m, "JAC_CONSTANT", BDF_JAC_CONSTANT);
    PyModule_AddIntConstant(m, "OK", BDF_OK);
    PyModule_AddIntConstant(m, "FINISHED", BDF_FINISHED);
    PyModule_AddIntConstant(m, "TOO_SMALL_STEP", BDF_TOO_SMALL_STEP);
    PyModule_AddIntConstant(m, "TOO_MANY_STEPS", BDF_TOO_MANY_STEPS);
    return m;
}
