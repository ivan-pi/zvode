#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <complex.h>


#include <stdint.h>
#include <stdio.h>

#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>

#include "zvode.h"

/* ------------------------------------------------------------------ */
/* Debug helpers (compile with -DZVODE_DEBUG to enable)               */
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
/* Callback struct and adaptors                                       */
/* ------------------------------------------------------------------ */

typedef enum { CB_PYTHON = 0, CB_CFUNC = 1 } cb_kind_t;

typedef struct {
    /* RHS */
    cb_kind_t fun_kind;
    union {
        PyObject  *pyobj;   /* CB_PYTHON */
        zvode_fun  cfunc;   /* CB_CFUNC  */
    } fun_u;

    /* Jacobian */
    cb_kind_t jac_kind;
    union {
        PyObject  *pyobj;   /* CB_PYTHON; Py_None when jac=None */
        zvode_jac  cfunc;   /* CB_CFUNC  */
    } jac_u;

    void *ctx;       /* shared user data; NULL when ctx=None */
    int   is_banded; /* 1 when miter == 4 */
    int   error;     /* set to 1 by adaptor on Python exception */
} zvode_cb_t;

static void fun_adaptor(
        int neq,
        double t,
        const double complex y[],
        double complex dy[],
        void *data) {

    zvode_cb_t *cb = (zvode_cb_t *) data;
    assert(cb != NULL);
    assert(neq > 0);

    if (cb->fun_kind == CB_CFUNC) {
        cb->fun_u.cfunc(neq, t, y, dy, cb->ctx);
        return;
    }

    /* Python path: fun(t, y) -> array; copy result into dy. */
    assert(cb->fun_u.pyobj != NULL);

    const npy_intp dims[1] = { (npy_intp) neq };
    PyArrayObject *ap_y = (PyArrayObject *) PyArray_SimpleNewFromData(
        1, dims, NPY_COMPLEX128, (void *) y);
    if (!ap_y) { cb->error = 1; return; }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    PyObject *res = PyObject_CallFunction(cb->fun_u.pyobj, "dO", t, (PyObject *) ap_y);
    Py_DECREF(ap_y);
    if (!res) { cb->error = 1; return; }

    PyArrayObject *ap_res = (PyArrayObject *) PyArray_FROM_OTF(
        res, NPY_COMPLEX128, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    Py_DECREF(res);
    if (!ap_res) { cb->error = 1; return; }

    memcpy(dy, PyArray_DATA(ap_res), (size_t) neq * sizeof(double complex));
    Py_DECREF(ap_res);
}

static void jac_adaptor(
        int neq,
        double t,
        const double complex y[],
        int ml, int mu,
        double complex pd[],
        int nrowpd,
        void *data) {

    zvode_cb_t *cb = (zvode_cb_t *) data;
    assert(cb != NULL);
    assert(neq > 0);
    assert(ml >= 0 && mu >= 0);

    if (cb->jac_kind == CB_CFUNC) {
        cb->jac_u.cfunc(neq, t, y, ml, mu, pd, nrowpd, cb->ctx);
        return;
    }

    /* No user Jacobian — ZVODE does not call us when miter ∉ {1,4},
     * but guard defensively. */
    if (cb->jac_u.pyobj == NULL || cb->jac_u.pyobj == Py_None)
        return;

    /* Python path: jac(t, y) -> array; copy result into pd (F-order).
     * Dense: result shape (neq, neq); banded: (ml+mu+1, neq). */
    const npy_intp dims_y[1] = { (npy_intp) neq };
    PyArrayObject *ap_y = (PyArrayObject *) PyArray_SimpleNewFromData(
        1, dims_y, NPY_COMPLEX128, (void *) y);
    if (!ap_y) { cb->error = 1; return; }
    PyArray_CLEARFLAGS(ap_y, NPY_ARRAY_WRITEABLE);

    PyObject *res = PyObject_CallFunction(cb->jac_u.pyobj, "dO", t, (PyObject *) ap_y);
    Py_DECREF(ap_y);
    if (!res) { cb->error = 1; return; }

    /* Require F-contiguous so column j starts at offset j * leading_dim. */
    PyArrayObject *ap_res = (PyArrayObject *) PyArray_FROM_OTF(
        res, NPY_COMPLEX128, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    Py_DECREF(res);
    if (!ap_res) { cb->error = 1; return; }

    const int rows        = cb->is_banded ? (ml + mu + 1) : neq;
    const npy_intp ldim_r = PyArray_DIM(ap_res, 0);
    const double complex *src = (const double complex *) PyArray_DATA(ap_res);
    for (int j = 0; j < neq; j++) {
        memcpy(pd + (npy_intp) j * nrowpd,
               src + (npy_intp) j * ldim_r,
               (size_t) rows * sizeof(double complex));
    }
    Py_DECREF(ap_res);
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
"`fun` is called as ``fun(t, y) -> array`` and must return the derivative;\n"
"`jac` (or None) is called as ``jac(t, y) -> array``.\n"
"Returns the advanced time and the ZVODE istate.\n");

static PyObject* zvode_py(PyObject* Py_UNUSED(self), PyObject *args) {

    PyArrayObject *ap_y = NULL, *ap_rtol = NULL, *ap_atol = NULL;
    PyArrayObject *ap_zwork = NULL, *ap_rwork = NULL, *ap_iwork = NULL;

    double t, tout;
    int itol, itask, istate, iopt, mf;

    PyObject *fun_obj = NULL, *jac_obj = NULL;
    zvode_cb_t cb;
    memset(&cb, 0, sizeof(cb));

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

    /* zvode_py is Python-only; compiled callbacks go through drive_knots/drive_adaptive. */
    cb.fun_kind = CB_PYTHON;
    cb.fun_u.pyobj = fun_obj;
    cb.jac_kind = CB_PYTHON;
    cb.jac_u.pyobj = jac_obj;
    cb.ctx = NULL;

    assert(ap_y);
    assert(ap_rtol);
    assert(ap_atol);
    assert(ap_zwork);
    assert(ap_rwork);
    assert(ap_iwork);
    assert(fun_obj);
    assert(jac_obj); // should be Py_None or a callable

    if (ZVODE_DEBUG) {
        dump_zvode_args(fun_obj, ap_y, t, tout, itol, ap_rtol, ap_atol,
                        itask, istate, iopt, ap_zwork, ap_rwork, ap_iwork,
                        jac_obj, mf);
    }

    const int neq = (int) PyArray_DIM(ap_y, 0);
    assert(neq > 0);          /* Python guarantees y0 is non-empty */
    assert(itask >= 1 && itask <= 5);  /* Python manages itask internally */

    const int miter = abs(mf) % 10;
    assert(miter <= 5);
    assert(abs(mf)/10 == 1 || abs(mf)/10 == 2); /* method */

    cb.is_banded = (miter == 4);

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

    if (!check_array(ap_yh, "yh", 2, NPY_COMPLEX128, 'F')) return NULL;
    if (!check_array(ap_dky, "dky", 1, NPY_COMPLEX128, 'C')) return NULL;
    if (!check_writable(ap_dky, "dky"))                 return NULL;

    const int n    = (int) PyArray_DIM(ap_dky,0);     /* number of equations */
    const int ldyh = (int) PyArray_DIM(ap_yh, 0);     /* leading dimension   */
    const int nq   = (int) PyArray_DIM(ap_yh, 1) - 1; /* current order       */

    assert(ldyh >= n);

    if ((int) PyArray_SIZE(ap_dky) < n) {
        PyErr_Format(PyExc_ValueError,
            "zvindy: dky must have length >= %d (got %d)",
            n, (int) PyArray_SIZE(ap_dky));
        return NULL;
    }

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
/* drive_knots                                                        */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(drive_knots_doc,
"drive_knots(fun, jac, ctx, mf, tspan, y, ts_out, ys_out,\n"
"            itol, rtol, atol, iopt, zwork, rwork, iwork) -> (istate, knots_completed)\n"
"\n"
"Integrate a complex ODE system to a sequence of pre-specified output knots.\n"
"\n"
"``fun`` and ``jac`` may each be a Python callable (return-value convention:\n"
"``fun(t, y) -> array``) or a Python int holding the address of a compiled\n"
"C function matching the ``zvode_fun``/``zvode_jac`` signature.\n"
"``ctx`` is an integer user-data pointer passed to compiled callbacks (0 = NULL).\n"
"\n"
"Advances the ODE from ``tspan[0]`` to ``tspan[-1]``, evaluating the solution\n"
"at each requested knot and writing the results into the pre-allocated output\n"
"arrays ``ts_out`` and ``ys_out``.\n"
"\n"
"Returns\n"
"-------\n"
"(istate, knots_completed) : (int, int)\n"
"    ``istate`` is the final ZVODE istate (2 = success, negative = failure).\n"
"    ``knots_completed`` includes column 0 (the initial condition).\n"
"\n"
"Raises\n"
"------\n"
"Exception\n"
"    If a Python callback raises an exception, it is propagated immediately.\n");

static PyObject *drive_knots_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *ap_tspan  = NULL;
    PyArrayObject *ap_y      = NULL;
    PyArrayObject *ap_ts_out = NULL, *ap_ys_out = NULL;
    PyArrayObject *ap_rtol   = NULL, *ap_atol   = NULL;
    PyArrayObject *ap_zwork  = NULL, *ap_rwork  = NULL, *ap_iwork = NULL;
    int mf, itol, iopt;

    PyObject *fun_obj, *jac_obj, *ctx_obj;
    zvode_cb_t cb;
    memset(&cb, 0, sizeof(cb));

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

    /* Detect fun kind: callable → CB_PYTHON; otherwise integer address → CB_CFUNC. */
    if (PyCallable_Check(fun_obj)) {
        cb.fun_kind    = CB_PYTHON;
        cb.fun_u.pyobj = fun_obj;
    } else {
        cb.fun_kind    = CB_CFUNC;
        cb.fun_u.cfunc = (zvode_fun) PyLong_AsVoidPtr(fun_obj);
        if (cb.fun_u.cfunc == NULL && PyErr_Occurred()) return NULL;
        if (cb.fun_u.cfunc == NULL) {
            PyErr_SetString(PyExc_ValueError, "drive_knots: fun address must be non-zero");
            return NULL;
        }
    }

    /* Detect jac kind: None or callable → CB_PYTHON; integer → CB_CFUNC. */
    if (jac_obj == Py_None || PyCallable_Check(jac_obj)) {
        cb.jac_kind    = CB_PYTHON;
        cb.jac_u.pyobj = jac_obj;
    } else {
        cb.jac_kind    = CB_CFUNC;
        cb.jac_u.cfunc = (zvode_jac) PyLong_AsVoidPtr(jac_obj);
        if (cb.jac_u.cfunc == NULL && PyErr_Occurred()) return NULL;
        if (cb.jac_u.cfunc == NULL) {
            PyErr_SetString(PyExc_ValueError, "drive_knots: jac address must be non-zero");
            return NULL;
        }
    }

    /* ctx is always passed as a Python int (0 for no context). */
    cb.ctx = PyLong_AsVoidPtr(ctx_obj);
    if (cb.ctx == NULL && PyErr_Occurred()) return NULL;

    cb.is_banded = (abs(mf) % 10 == 4);

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
/* drive_adaptive                                                     */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(drive_adaptive_doc,
"drive_adaptive(fun, jac, ctx, y, rtol, atol,\n"
"               t0, t_bound, itol, iopt, mf,\n"
"               zwork, rwork, iwork,\n"
"               refine, allow_overshoot)\n"
"    -> (ts, ys, istate)\n"
"\n"
"Drive ZVODE in single-step mode, collecting every accepted step.\n"
"Returns (ts, ys, istate) where ts is a 1-D float64 array and ys is a\n"
"(neq, m) complex128 F-order array.\n"
"\n"
"``fun`` and ``jac`` may be Python callables (return-value convention) or\n"
"Python ints holding compiled C function pointer addresses.\n"
"``ctx`` is an integer user-data pointer passed to compiled callbacks (0 = NULL).\n");

#define ADAPTIVE_INIT_CAP 1024

static PyObject *drive_adaptive_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *ap_y     = NULL;
    PyArrayObject *ap_rtol  = NULL, *ap_atol  = NULL;
    PyArrayObject *ap_zwork = NULL, *ap_rwork = NULL, *ap_iwork = NULL;
    double t0, t_bound;
    int mf, itol, iopt, refine, allow_overshoot;

    PyObject *fun_obj, *jac_obj, *ctx_obj;
    zvode_cb_t cb;
    memset(&cb, 0, sizeof(cb));

    if (!PyArg_ParseTuple(args, "OOOO!O!O!ddiiiO!O!O!ii:drive_adaptive",
            &fun_obj,
            &jac_obj,
            &ctx_obj,
            &PyArray_Type, &ap_y,
            &PyArray_Type, &ap_rtol,
            &PyArray_Type, &ap_atol,
            &t0, &t_bound,
            &itol, &iopt, &mf,
            &PyArray_Type, &ap_zwork,
            &PyArray_Type, &ap_rwork,
            &PyArray_Type, &ap_iwork,
            &refine, &allow_overshoot))
        return NULL;

    /* Detect fun kind */
    if (PyCallable_Check(fun_obj)) {
        cb.fun_kind    = CB_PYTHON;
        cb.fun_u.pyobj = fun_obj;
    } else {
        cb.fun_kind    = CB_CFUNC;
        cb.fun_u.cfunc = (zvode_fun) PyLong_AsVoidPtr(fun_obj);
        if (cb.fun_u.cfunc == NULL && PyErr_Occurred()) return NULL;
        if (cb.fun_u.cfunc == NULL) {
            PyErr_SetString(PyExc_ValueError, "drive_adaptive: fun address must be non-zero");
            return NULL;
        }
    }

    /* Detect jac kind */
    if (jac_obj == Py_None || PyCallable_Check(jac_obj)) {
        cb.jac_kind    = CB_PYTHON;
        cb.jac_u.pyobj = jac_obj;
    } else {
        cb.jac_kind    = CB_CFUNC;
        cb.jac_u.cfunc = (zvode_jac) PyLong_AsVoidPtr(jac_obj);
        if (cb.jac_u.cfunc == NULL && PyErr_Occurred()) return NULL;
        if (cb.jac_u.cfunc == NULL) {
            PyErr_SetString(PyExc_ValueError, "drive_adaptive: jac address must be non-zero");
            return NULL;
        }
    }

    cb.ctx       = PyLong_AsVoidPtr(ctx_obj);
    if (cb.ctx == NULL && PyErr_Occurred()) return NULL;
    cb.is_banded = (abs(mf) % 10 == 4);

    const int ITASK = allow_overshoot ? 2 : 5;
    const int neq   = (int) PyArray_DIM(ap_y, 0);

    const int lzw = (int) PyArray_SIZE(ap_zwork);
    const int lrw = (int) PyArray_SIZE(ap_rwork);
    const int liw = (int) PyArray_SIZE(ap_iwork);

    double complex *zwork = (double complex *) PyArray_DATA(ap_zwork);
    double         *rwork = (double *)         PyArray_DATA(ap_rwork);
    int            *iwork = (int *)            PyArray_DATA(ap_iwork);
    const double   *rtol  = (const double *)   PyArray_DATA(ap_rtol);
    const double   *atol  = (const double *)   PyArray_DATA(ap_atol);

    /* Work on a copy of y so the caller's array is not modified. */
    double complex *ytmp = (double complex *) malloc((size_t) neq * sizeof(double complex));
    if (!ytmp) return PyErr_NoMemory();
    memcpy(ytmp, PyArray_DATA(ap_y), (size_t) neq * sizeof(double complex));

    /* Dynamic output buffers; grow by doubling when full. */
    npy_intp cap = ADAPTIVE_INIT_CAP;
    double         *ts_buf = (double *)        malloc((size_t) cap * sizeof(double));
    double complex *ys_buf = (double complex *) malloc((size_t) neq * (size_t) cap
                                                       * sizeof(double complex));
    if (!ts_buf || !ys_buf) {
        free(ytmp); free(ts_buf); free(ys_buf);
        return PyErr_NoMemory();
    }

    double t         = t0;
    double direction = (t_bound > t0) ? 1.0 : -1.0;
    int istate       = 1;
    npy_intp count   = 0;

    /* Store initial condition (column 0). */
    ts_buf[count] = t;
    memcpy(ys_buf + (size_t) count * (size_t) neq, ytmp,
           (size_t) neq * sizeof(double complex));
    count = 1;

    while (direction * (t_bound - t) > 0.0) {
        double t_old = t;

        c_zvode(
            &fun_adaptor, neq, ytmp,
            &t, t_bound,
            itol, rtol, atol,
            ITASK, &istate,
            iopt,
            zwork, lzw,
            rwork, lrw,
            iwork, liw,
            &jac_adaptor,
            mf,
            &cb
        );

        if (cb.error) {
            free(ytmp); free(ts_buf); free(ys_buf);
            assert(PyErr_Occurred());
            return NULL;
        }

        if (istate < 0) break;

        /* Ensure capacity for up to refine new columns. */
        npy_intp need = count + (npy_intp) (refine > 1 ? refine : 1);
        if (need > cap) {
            npy_intp new_cap = cap;
            while (new_cap < need) new_cap *= 2;

            double *new_ts = (double *) realloc(ts_buf,
                (size_t) new_cap * sizeof(double));
            if (!new_ts) { free(ytmp); free(ts_buf); free(ys_buf);
                           return PyErr_NoMemory(); }
            ts_buf = new_ts;

            double complex *new_ys = (double complex *) realloc(ys_buf,
                (size_t) neq * (size_t) new_cap * sizeof(double complex));
            if (!new_ys) { free(ytmp); free(ts_buf); free(ys_buf);
                           return PyErr_NoMemory(); }
            ys_buf = new_ys;
            cap    = new_cap;
        }

        /* Optionally insert interpolated points between t_old and t. */
        if (refine > 1) {
            const int nq = iwork[13];     /* NQU: order last used */
            const double hu = rwork[10];  /* HU:  step size last used */
            struct zvode_step_t step = { .h = hu, .tn = t, .hu = hu, .nq = nq };

            for (int k = 1; k < refine; k++) {
                double t_k = t_old + (double) k * (t - t_old) / (double) refine;
                double complex *dky = ys_buf + (size_t) count * (size_t) neq;
                const int iflag = c_zvindy(neq, t_k, zwork, neq, 0, dky, &step);
                if (iflag != 0) {
                    free(ytmp); free(ts_buf); free(ys_buf);
                    PyErr_Format(PyExc_ValueError,
                        "drive_adaptive: ZVINDY failed (iflag=%d) at "
                        "interpolation time t=%.17g", iflag, t_k);
                    return NULL;
                }
                ts_buf[count] = t_k;
                count++;
            }
        }

        /* Store step endpoint. */
        ts_buf[count] = t;
        memcpy(ys_buf + (size_t) count * (size_t) neq, ytmp,
               (size_t) neq * sizeof(double complex));
        count++;
    }

    /* Build output numpy arrays from the C buffers. */
    PyArrayObject *ts_arr = NULL, *ys_arr = NULL;

    {
        npy_intp dims_ts[1] = { count };
        ts_arr = (PyArrayObject *) PyArray_SimpleNew(1, dims_ts, NPY_FLOAT64);
        if (!ts_arr) goto oom;
        memcpy(PyArray_DATA(ts_arr), ts_buf, (size_t) count * sizeof(double));
    }

    {
        /* ys_buf is stored column-by-column: ys_buf[k*neq + i] = ys[i, k].
         * This matches F-order (neq, count): element [i, k] at offset i + k*neq. */
        npy_intp dims_ys[2] = { neq, count };
        ys_arr = (PyArrayObject *) PyArray_EMPTY(2, dims_ys, NPY_COMPLEX128, 1 /* F-order */);
        if (!ys_arr) { Py_DECREF(ts_arr); goto oom; }
        memcpy(PyArray_DATA(ys_arr), ys_buf,
               (size_t) neq * (size_t) count * sizeof(double complex));
    }

    free(ytmp); free(ts_buf); free(ys_buf);

    PyObject *result = Py_BuildValue("(OOi)",
        (PyObject *) ts_arr, (PyObject *) ys_arr, istate);
    Py_DECREF(ts_arr);
    Py_DECREF(ys_arr);
    return result;

oom:
    free(ytmp); free(ts_buf); free(ys_buf);
    return PyErr_NoMemory();
}


static struct PyMethodDef zvode_module_methods[] = {
    {"zvode",         zvode_py,         METH_VARARGS, zvode_doc},
    {"zvindy",        zvindy_py,        METH_VARARGS, zvindy_doc},
    {"drive_knots",   drive_knots_py,   METH_VARARGS, drive_knots_doc},
    {"drive_adaptive",drive_adaptive_py,METH_VARARGS, drive_adaptive_doc},
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
