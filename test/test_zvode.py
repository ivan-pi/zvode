"""Tests of the _zvode extension module

These smoke-tests exercise _zvode.zvode(...) directly, bypassing any
higher-level wrapper.  The goal is to verify that the C extension and the
underlying Fortran solver are wired together correctly before debugging the
ODE-solver class that crashes the interpreter.

Conventions (from the ZVODE documentation)
-------------------------------------------
istate  1  – first call (input)
        2  – successful return (output)
        negative – error (output)

itask   1  – integrate to TOUT, interpolating at TOUT

itol    1  – scalar rtol *and* scalar atol (both length-1 arrays)

mf = 10   Adams / non-stiff, functional iteration (no Jacobian needed)
mf = 22   BDF   / stiff,     internally-generated dense Jacobian

Work-array sizes (from the ZVODE docs, using default MAXORD)
-------------------------------------------------------------
MF = 10:  LZW = 15*NEQ,            LRW = 20+NEQ,  LIW = 30
MF = 22:  LZW = 8*NEQ + 2*NEQ**2,  LRW = 20+NEQ,  LIW = 30+NEQ

Optional output stored by ZVODE on a successful call (0-based Python indices)
  rwork[10]  = HU   – step size last used
  rwork[12]  = TCUR – current internal time reached by the solver
  iwork[10]  = NST  – number of steps taken
  iwork[11]  = NFE  – number of f evaluations
"""

import numpy as np
from numpy.testing import assert_allclose

import pytest

from zvode import _zvode

import ctypes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

print("Hello from test_zvode.py")

def _make_workspaces(neq, mf):
    """Allocate ZVODE work arrays for the given NEQ and MF.

    Uses the size formulas from the ZVODE documentation (default MAXORD).
    """
    miter = mf % 10

    if mf == 10:
        lzw = 15 * neq
    elif mf in (11, 12):
        lzw = 15 * neq + 2 * neq**2
    elif mf == 13:
        lzw = 16 * neq
    elif mf == 20:
        lzw = 8 * neq
    elif mf in (21, 22):
        lzw = 8 * neq + 2 * neq**2
    elif mf == 23:
        lzw = 9 * neq
    else:
        raise ValueError(f"Unsupported MF = {mf} in test helper")

    lrw = 20 + neq
    liw = 30 if miter in (0, 3) else 30 + neq

    zwork = np.zeros(lzw, dtype=np.complex128)
    rwork = np.zeros(lrw, dtype=np.float64)
    iwork = np.zeros(liw, dtype=np.int32)

    return zwork, rwork, iwork


def _call_zvode(fun, y, t, tout, zwork, rwork, iwork, *,
                mf=10, itol=1, rtol=1e-7, atol=1e-9,
                itask=1, istate=1, iopt=0, jac=None):
    """Thin wrapper that converts scalar tolerances to 1-element arrays."""
    rtol_arr = np.array([rtol], dtype=np.float64)
    atol_arr = np.array([atol], dtype=np.float64)
    return _zvode.zvode(
        fun, y, t, tout,
        itol, rtol_arr, atol_arr,
        itask, istate, iopt,
        zwork, rwork, iwork,
        jac, mf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zvode_scalar_real_decay():
    """
    dy/dt = -y,  y(0) = 1   =>  y(t) = exp(-t)

    The simplest possible call: one equation, Adams method (MF=10),
    no Jacobian, real-valued solution carried in a complex array.
    Checks that the solver returns istate == 2 and reaches TOUT.
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        print("ABC: In fun with 1 eq")
        print(y.dtype, dy.dtype, y.shape, dy.shape)
        dy[:] = -y[:]

    y     = np.array([1.0 + 0j], dtype=np.complex128)
    t     = 0.0
    tout  = 10.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)
    iopt = 0

    print(f"lzw = {zwork.shape}")
    print(f"lrw = {rwork.shape}")
    print(f"liw = {iwork.shape}")

    base = zwork.ctypes.data
    print(f"zwork: [{hex(base)}, {hex(base + zwork.nbytes)})")

    itol = 1
    rtol_arr = np.array([1e-6], dtype=np.float64)
    atol_arr = np.array([1e-8], dtype=np.float64)

    itask = 2
    istate = 1

    t_new, istate_new = _zvode.zvode(
        fun, y, t, tout,
        itol, rtol_arr, atol_arr,
        itask, istate, iopt,
        zwork, rwork, iwork,
        None, mf)


    assert istate_new == 2,  f"ZVODE failed with istate = {istate_new}"
    assert t_new     == tout, "ZVODE did not reach TOUT"

    assert_allclose(y[0].real, np.exp(-tout), rtol=1e-4, atol=1e-8)
    assert abs(y[0].imag) < 1e-12, "Imaginary part should remain zero"


def test_zvode_complex_rotation():
    """
    dy/dt = i*y,  y(0) = 1   =>  y(t) = exp(i*t)

    Tests the complex arithmetic path.  After a quarter-turn the solution
    should be purely imaginary: y(pi/2) ≈ i.
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        dy[0] = 1j * y[0]

    y     = np.array([1.0 + 0j], dtype=np.complex128)
    t     = 0.0
    tout  = np.pi / 2.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork, mf=mf)

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    expected = np.exp(1j * tout)           # = 0 + 1j (exactly at pi/2)
    assert_allclose(y[0], expected, rtol=1e-5, atol=1e-8)


def test_zvode_multistep_continuation():
    """
    Advance through several TOUT values by calling zvode repeatedly,
    reusing the same work arrays and threading istate through.

    Uses dy/dt = i*y so that y(t) = exp(i*t), which gives exact
    checkpoints every quarter-turn of the unit circle.
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        dy[0] = 1j * y[0]

    y      = np.array([1.0 + 0j], dtype=np.complex128)
    t      = 0.0
    rtol   = np.array([1e-7], dtype=np.float64)
    atol   = np.array([1e-9], dtype=np.float64)
    istate = 1
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    checkpoints = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

    for tout in checkpoints:
        t, istate = _zvode.zvode(
            fun, y, t, tout,
            1, rtol, atol,
            1, istate, 0,
            zwork, rwork, iwork,
            None, mf)

        assert istate == 2, \
            f"ZVODE failed at tout={tout:.4f} with istate={istate}"

        expected = np.exp(1j * tout)
        assert_allclose(y[0], expected, rtol=1e-5, atol=1e-8,
                        err_msg=f"Solution mismatch at tout = {tout:.4f}")


def test_zvode_two_component_system():
    """
    Two decoupled equations:
        dy[0]/dt = -0.1*y[0]
        dy[1]/dt = -2.0*y[1]
    Exact solutions: y[k](t) = exp(-rate[k]*t)

    Tests multi-equation support with MF=10.
    """
    neq = 2
    mf = 10

    def fun(t, y, dy):
        dy[0] = -0.1 * y[0]
        dy[1] = -2.0 * y[1]

    y     = np.array([1.0+0j, 1.0+0j], dtype=np.complex128)
    t     = 0.0
    tout  = 5.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork,
                                    mf=mf, rtol=1e-6, atol=1e-8)

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    expected = np.array([np.exp(-0.1*tout), np.exp(-2.0*tout)]) + 0j
    assert_allclose(y, expected, rtol=1e-4)


def test_zvode_bdf_method():
    """
    Test the BDF stiff solver (MF=22: BDF + internally-generated dense Jacobian).

    Problem: dy/dt = -y, y(0) = 1.  Not intrinsically stiff, but exercises
    the BDF code path and the larger complex workspace (8*NEQ + 2*NEQ**2).
    """
    neq = 1
    mf = 22

    def fun(t, y, dy):
        dy[0] = -y[0]

    y     = np.array([1.0 + 0j], dtype=np.complex128)
    t     = 0.0
    tout  = 5.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork,
                                    mf=mf, rtol=1e-6, atol=1e-8)

    assert istate_new == 2, f"BDF ZVODE failed with istate = {istate_new}"
    assert_allclose(y[0].real, np.exp(-tout), rtol=1e-4)


def test_zvode_optional_output_populated():
    """
    After a successful call the solver writes diagnostic counters into
    rwork and iwork.  Verify that at least some of them are non-zero,
    which confirms the Fortran output path was reached:

        rwork[10] = HU    (last step size used)
        rwork[12] = TCUR  (internal time reached)
        iwork[10] = NST   (number of steps)
        iwork[11] = NFE   (number of f evaluations)
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        dy[0] = -y[0]

    y     = np.array([1.0 + 0j], dtype=np.complex128)
    t     = 0.0
    tout  = 2.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork, mf=mf)

    assert istate_new == 2

    hu   = rwork[10]   # last step size used
    tcur = rwork[12]   # current internal time
    nst  = iwork[10]   # number of steps
    nfe  = iwork[11]   # number of f evaluations

    assert hu   > 0,    f"HU should be positive, got {hu}"
    assert tcur >= tout, f"TCUR ({tcur}) should be >= TOUT ({tout})"
    assert nst  > 0,    f"NST should be positive, got {nst}"
    assert nfe  > 0,    f"NFE should be positive, got {nfe}"


def test_zvode_wrong_array_type():
    """
    Passing a plain Python list where a numpy ndarray is required should
    raise a TypeError, not crash the interpreter.
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        dy[0] = -y[0]

    rtol  = np.array([1e-6], dtype=np.float64)
    atol  = np.array([1e-8], dtype=np.float64)
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    # y is a list instead of a numpy array
    with pytest.raises(TypeError):
        _zvode.zvode(
            fun, [1.0 + 0j], 0.0, 1.0,
            1, rtol, atol, 1, 1, 0,
            zwork, rwork, iwork,
            None, mf)

    # zwork is a list instead of a numpy array
    y = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(TypeError):
        _zvode.zvode(
            fun, y, 0.0, 1.0,
            1, rtol, atol, 1, 1, 0,
            list(zwork), rwork, iwork,
            None, mf)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_zvode_scalar_real_decay()
    test_zvode_complex_rotation()
    test_zvode_multistep_continuation()
    test_zvode_two_component_system()
    test_zvode_bdf_method()
    test_zvode_optional_output_populated()
    test_zvode_wrong_array_type()
    print("All tests passed.")
