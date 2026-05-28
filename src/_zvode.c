#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "zvode.h"

struct zvode_callbacks {
	PyObject *fun;
	PyObject *jac;
	// TODO: add zewset and zwnorm in the future
};

static void fun_adaptor(
		int neqn,
		double t,
		double complex y[],
		double complex dy[],
		void *ctx) {

	struct zvode_callbacks cb = ctx;
	assert(cb->fun != NULL);

	// TODO: use complex vectors here

	// 1. Create Numpy vectors
	const npy_intp dims_y[1] = {neqn};
	PyArrayObject *ap_y = PyArray_SimpleNewFromData(1, dims_y, NPY_FLOAT64, y);
	if (!ap_y) {}

	const npy_intp dims_dy[1] = {neqn};
	PyArrayObject *ap_dy = PyArray_SimpleNewFromData(1, dims_dy, NPY_FLOAT64, dy);
	if (!ap_dy) {}

	// 2. Invoke the callback function, fun(t,y,dy) -> None
	PyObject_CallFunction(
		cb->fun,"dOO", t, ap_y, ap_dy)
}

static void jac_adaptor(
		int neq
		double t
		double complex y[],
		int ml, int mu,
		double complex pd[],
		int nrowpd,
		void *ctx) {

	struct zvode_callbacks cb = ctx;
	assert(cb->jac != NULL);

	// TODO: build numpy compatible array objects for y and pd
	// the arrays pd has dimension nrowpd by neq, but it might represent
	// either a dense or a banded array (including padding)

	// TODO: use ml and mu in the callback
	PyObject_CallFunction(
		cb->fun,"dOO", t, ap_y, ap_dy)
}

PyDocSTR(zvode_doc, /* TODO */)
static PyObject* zvode_py(PyObject* self, PyObject *args) {

	int itask, istate;

	PyArrayObject *ap_y, *ap_atol, *ap_rtol;
	PyArrayObject *ap_zwork, *ap_rwork, *ap_iwork;

	struct zvode_callbacks cb = {.fun=NULL, .jac=Py_None};

	// t, istate = zvode_step(
	// 		)

	if (!(PyArg_ParseTuple(args,":zvode_step",
		))) { return NULL; }


	// Call the Fortran integrator
	zvode(
		&fun_adaptor,
		neqn, y, t, tout,
		itol, rtol, atol,
		itask, istate,
		iopt, zwork, lzw, rwork, lrw, iwork, liw
		&jac_adaptor,
		mf,
		(void *) &cb
	);

	PyObject *res
	if (!(res = Py_BuildValue("di",t,istate))) {
		return NULL;
	}
	return res;
}

PyDocSTR(zvindy_doc, /* TODO */)
static PyObject* zvindy_py(PyObject* self, PyObject *args) {
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

	import_array(); // NumPy

    PyObject *m;
    if (!(m = PyModule_Create(&module_def))) {
        return NULL;
    }
    return m;
}
