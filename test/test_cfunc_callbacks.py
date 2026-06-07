"""Tests for compiled (ctypes CFUNCTYPE / numba @cfunc) callbacks in solve_complex_ivp.

The test ODE is the same 2-component coupled system used in
test_solve_complex_ivp.py so that analytic solutions are readily available
and numerical results can be cross-checked.

    dy[0]/dt = LAM1*y[0] + C*y[1]
    dy[1]/dt = LAM2*y[1]

Analytic solution:
    y[1](t) = Y0[1] * exp(LAM2*t)
    y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t)
    B = C*Y0[1]/(LAM2 - LAM1),  A = Y0[0] - B

ctypes callbacks use a Python body executed through a C thunk; they test the
full compiled-callback code path (function-pointer extraction, C-level dispatch)
without requiring numba.  Numba tests are guarded by pytest.importorskip and are
skipped when numba is not installed.

All ctypes callback functions are defined at module scope so that ctypes does
not garbage-collect their underlying C thunk during testing.
"""

import ctypes

import numpy as np
import pytest

from zvode import solve_complex_ivp, ZVODE_FUN_CTYPE, ZVODE_JAC_CTYPE, check_cfunc_signature

# ---------------------------------------------------------------------------
# Problem parameters  (same values as test_solve_complex_ivp.py)
# ---------------------------------------------------------------------------

LAM1 = -1 + 2j
LAM2 = -2 + 1j
C = 0.5j
Y0 = np.array([1.0 + 0j, 0.0 + 1j])
T0 = 0.0
TF = 2.0
LBAND = 0
UBAND = 1

_B = C * Y0[1] / (LAM2 - LAM1)
_A = Y0[0] - _B

RTOL = 1e-8
ATOL = 1e-10


def exact(t):
    """Analytic solution at time(s) t, returned as shape (2, ...) array."""
    t = np.asarray(t, dtype=float)
    y0c = _A * np.exp(LAM1 * t) + _B * np.exp(LAM2 * t)
    y1c = Y0[1] * np.exp(LAM2 * t)
    return np.array([y0c, y1c])


def _check(t_arr, y_arr, sol_rtol=1e-5):
    ref = exact(t_arr)
    assert np.allclose(y_arr, ref, rtol=sol_rtol), (
        f"max err={np.max(np.abs(y_arr - ref)):.2e}"
    )


# ---------------------------------------------------------------------------
# ctypes callback prototype helpers
#
# double complex in C has the same memory layout as two consecutive doubles
# (real part first), so we treat each complex element as a pair of doubles.
# All callback objects are kept alive at module scope.
# ---------------------------------------------------------------------------

# void fun(int neq, double t, double* y, double* dy, void* ctx)
# y and dy are treated as arrays of 2*neq doubles (real/imag interleaved)
FUN_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,                    # neq
    ctypes.c_double,                 # t
    ctypes.POINTER(ctypes.c_double), # y  (complex128[] as double pairs)
    ctypes.POINTER(ctypes.c_double), # dy (complex128[] as double pairs)
    ctypes.c_void_p,                 # ctx
)

# void jac(int neq, double t, double* y, int ml, int mu, double* pd, int nrowpd, void* ctx)
JAC_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,                    # neq
    ctypes.c_double,                 # t
    ctypes.POINTER(ctypes.c_double), # y
    ctypes.c_int,                    # ml
    ctypes.c_int,                    # mu
    ctypes.POINTER(ctypes.c_double), # pd (complex128[], F-order, leading dim nrowpd)
    ctypes.c_int,                    # nrowpd
    ctypes.c_void_p,                 # ctx
)


def _c128_get(ptr, i):
    """Read complex128 element i from a double* pointer."""
    return complex(ptr[2 * i], ptr[2 * i + 1])


def _c128_set(ptr, i, val):
    """Write complex value val to complex128 element i via a double* pointer."""
    ptr[2 * i] = val.real
    ptr[2 * i + 1] = val.imag


@FUN_CTYPE
def _fun_ctypes(neq, t, y, dy, ctx):
    """RHS for the coupled ODE via ctypes callback."""
    y0c = _c128_get(y, 0)
    y1c = _c128_get(y, 1)
    _c128_set(dy, 0, LAM1 * y0c + C * y1c)
    _c128_set(dy, 1, LAM2 * y1c)


@JAC_CTYPE
def _jac_dense_ctypes(neq, t, y, ml, mu, pd, nrowpd, ctx):
    """Dense Jacobian via ctypes callback.

    pd is column-major (F-order): element [row, col] is at offset
    row + col*nrowpd in complex128 units, i.e. 2*(row + col*nrowpd) in double units.
    """
    _c128_set(pd, 0 + 0 * nrowpd, LAM1)   # J[0, 0]
    _c128_set(pd, 0 + 1 * nrowpd, C)      # J[0, 1]
    _c128_set(pd, 1 + 1 * nrowpd, LAM2)   # J[1, 1]


@JAC_CTYPE
def _jac_banded_ctypes(neq, t, y, ml, mu, pd, nrowpd, ctx):
    """Banded Jacobian via ctypes callback.

    Banded storage: element J[i, j] goes to pd[mu + i - j, j].
    With LBAND=0, UBAND=1 (ml=0, mu=1):
      J[0,0]=LAM1 -> pd[1, 0] -> offset 2*(1 + 0*nrowpd)
      J[0,1]=C    -> pd[0, 1] -> offset 2*(0 + 1*nrowpd)
      J[1,1]=LAM2 -> pd[1, 1] -> offset 2*(1 + 1*nrowpd)
    """
    _c128_set(pd, mu + 0 - 0 + 0 * nrowpd, LAM1)  # row=mu+i-j=1, col=0
    _c128_set(pd, mu + 0 - 1 + 1 * nrowpd, C)     # row=mu+i-j=0, col=1
    _c128_set(pd, mu + 1 - 1 + 1 * nrowpd, LAM2)  # row=mu+i-j=1, col=1


# ---------------------------------------------------------------------------
# Helper to drive a tspan through all three output modes
# ---------------------------------------------------------------------------


def _run_all_modes(fun, jac=None, extra_kw=None):
    """Return (steps_sol, endpoint_sol, knots_sol)."""
    kw = dict(in_place=True, rtol=RTOL, atol=ATOL)
    if extra_kw:
        kw.update(extra_kw)
    if jac is not None:
        kw["jac"] = jac

    steps_sol = solve_complex_ivp(fun, [T0, TF], Y0, save_steps=True, **kw)
    endpoint_sol = solve_complex_ivp(fun, [T0, TF], Y0, save_steps=False, **kw)
    knots_sol = solve_complex_ivp(fun, np.linspace(T0, TF, 11), Y0, **kw)
    return steps_sol, endpoint_sol, knots_sol


# ===========================================================================
# 1. ctypes callbacks — RHS only (no Jacobian, miter=2)
# ===========================================================================


def test_ctypes_fun_steps():
    """ctypes RHS, save_steps=True: solution matches analytic at all steps."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, rtol=RTOL, atol=ATOL
    )
    assert isinstance(sol.t, np.ndarray)
    assert sol.t[0] == T0 and sol.t[-1] == pytest.approx(TF)
    assert sol.y.shape == (2, len(sol.t))
    _check(sol.t, sol.y)


def test_ctypes_fun_endpoint():
    """ctypes RHS, save_steps=False (endpoint-only): scalar t and 1-D y."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, save_steps=False,
        rtol=RTOL, atol=ATOL,
    )
    assert np.isscalar(sol.t)
    assert sol.t == pytest.approx(TF)
    assert sol.y.ndim == 1 and sol.y.shape == (2,)
    ref = exact(TF)
    assert np.allclose(sol.y, ref, rtol=1e-5)


def test_ctypes_fun_knots():
    """ctypes RHS, knot mode: solution at exactly the requested times."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0, in_place=True, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_array_equal(sol.t, tspan)
    assert sol.y.shape == (2, 11)
    _check(sol.t, sol.y)


# ===========================================================================
# 2. ctypes callbacks — RHS + dense Jacobian (miter=1)
# ===========================================================================


def test_ctypes_dense_jac_steps():
    """ctypes RHS + dense Jacobian, save_steps=True."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0,
        in_place=True, jac=_jac_dense_ctypes, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_ctypes_dense_jac_endpoint():
    """ctypes RHS + dense Jacobian, endpoint-only."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0,
        in_place=True, jac=_jac_dense_ctypes, save_steps=False, rtol=RTOL, atol=ATOL,
    )
    ref = exact(TF)
    assert np.allclose(sol.y, ref, rtol=1e-5)


def test_ctypes_dense_jac_knots():
    """ctypes RHS + dense Jacobian, knot mode."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0,
        in_place=True, jac=_jac_dense_ctypes, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ===========================================================================
# 3. ctypes callbacks — RHS + banded Jacobian (miter=4)
# ===========================================================================


def test_ctypes_banded_jac_steps():
    """ctypes RHS + banded Jacobian, save_steps=True."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0,
        in_place=True, jac=_jac_banded_ctypes,
        lband=LBAND, uband=UBAND, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_ctypes_banded_jac_endpoint():
    """ctypes RHS + banded Jacobian, endpoint-only."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0,
        in_place=True, jac=_jac_banded_ctypes,
        lband=LBAND, uband=UBAND, save_steps=False, rtol=RTOL, atol=ATOL,
    )
    ref = exact(TF)
    assert np.allclose(sol.y, ref, rtol=1e-5)


def test_ctypes_banded_jac_knots():
    """ctypes RHS + banded Jacobian, knot mode."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0,
        in_place=True, jac=_jac_banded_ctypes,
        lband=LBAND, uband=UBAND, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ===========================================================================
# 4. Output modes — result type and shape
# ===========================================================================


def test_ctypes_steps_result_type():
    """save_steps=True: t is 1-D ndarray, y is 2-D ndarray (2, m)."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, rtol=RTOL, atol=ATOL,
    )
    assert isinstance(sol.t, np.ndarray) and sol.t.ndim == 1
    assert isinstance(sol.y, np.ndarray) and sol.y.ndim == 2
    assert sol.y.shape[0] == 2 and sol.y.shape[1] == len(sol.t)
    assert sol.y.dtype == np.complex128


def test_ctypes_endpoint_result_type():
    """save_steps=False: t is a scalar float, y is 1-D ndarray."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, save_steps=False,
        rtol=RTOL, atol=ATOL,
    )
    assert np.isscalar(sol.t)
    assert sol.y.ndim == 1 and sol.y.shape == (2,)


def test_ctypes_knots_result_shape():
    """Knot mode: output times equal the requested knots."""
    tspan = np.linspace(T0, TF, 9)
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0, in_place=True, rtol=RTOL, atol=ATOL,
    )
    np.testing.assert_array_equal(sol.t, tspan)
    assert sol.y.shape == (2, 9)


# ===========================================================================
# 5. Statistics counters present and positive
# ===========================================================================


def test_ctypes_stats():
    """Solver statistics are present and sensible for ctypes callbacks."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, rtol=RTOL, atol=ATOL,
    )
    assert sol.nsteps > 0
    assert sol.nfev > 0
    assert sol.njev >= 0
    assert sol.nlu >= 0


# ===========================================================================
# 6. Backward integration
# ===========================================================================


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_ctypes_backward(mode):
    """Backward integration (TF → T0) with ctypes callback."""
    y_tf = exact(TF).astype(np.complex128)  # shape (2,) at t=TF
    if mode == "knots":
        tspan = np.linspace(TF, T0, 11)
    else:
        tspan = [TF, T0]
    save = mode == "steps"

    sol = solve_complex_ivp(
        _fun_ctypes, tspan, y_tf,
        in_place=True, save_steps=save, rtol=RTOL, atol=ATOL,
    )

    if mode == "endpoint":
        ref = exact(T0)  # shape (2,) for scalar t
        assert sol.t == pytest.approx(T0)
        assert np.allclose(sol.y, ref, rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 7. refine > 1 (ZVINDY interpolation through compiled-callback path)
# ===========================================================================


def test_ctypes_refine():
    """refine=4 inserts 3 interpolated points per step via ctypes callback."""
    REFINE = 4
    sol_base = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, rtol=RTOL, atol=ATOL, refine=1,
    )
    sol_ref = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, in_place=True, rtol=RTOL, atol=ATOL, refine=REFINE,
    )
    n_steps = len(sol_base.t) - 1
    assert len(sol_ref.t) == len(sol_base.t) + n_steps * (REFINE - 1)
    _check(sol_ref.t, sol_ref.y)


# ===========================================================================
# 8. Scalar rotation ODE (neq=1, different problem, simple analytic solution)
#
# dy/dt = i*y,  y(0) = 1,  exact solution y(t) = exp(i*t)
# ===========================================================================

FUN_ROTATION_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)


@FUN_ROTATION_CTYPE
def _fun_rotation_ctypes(neq, t, y, dy, ctx):
    """RHS for dy/dt = i*y."""
    yr, yi = y[0], y[1]
    dy[0] = -yi   # Re(i*(yr + i*yi)) = -yi
    dy[1] = yr    # Im(i*(yr + i*yi)) =  yr


def test_ctypes_rotation_ode():
    """Scalar rotation: y(t)=exp(i*t), verify after half revolution."""
    y0_rot = np.array([1.0 + 0j], dtype=np.complex128)
    sol = solve_complex_ivp(
        _fun_rotation_ctypes, [0.0, np.pi], y0_rot,
        in_place=True, rtol=1e-8, atol=1e-10,
    )
    expected = np.exp(1j * np.pi)  # should be -1
    np.testing.assert_allclose(sol.y[0, -1], expected, atol=1e-6)


def test_ctypes_rotation_ode_knots():
    """Scalar rotation, knot mode."""
    y0_rot = np.array([1.0 + 0j], dtype=np.complex128)
    tspan = np.linspace(0, 2 * np.pi, 7)
    sol = solve_complex_ivp(
        _fun_rotation_ctypes, tspan, y0_rot,
        in_place=True, rtol=1e-8, atol=1e-10,
    )
    ref = np.exp(1j * tspan).reshape(1, -1)
    np.testing.assert_allclose(sol.y, ref, atol=1e-6)


# ===========================================================================
# 9. Numba @cfunc tests  (skipped when numba is not installed)
#
# pytest.importorskip lives inside each fixture so that only the numba-
# dependent tests are skipped, not the whole module.
# ===========================================================================


@pytest.fixture(scope="module")
def numba_fun():
    """Return a numba @cfunc RHS for the coupled ODE."""
    numba = pytest.importorskip("numba", reason="numba not installed")
    from numba import cfunc, types  # noqa: F811

    @cfunc(
        types.void(
            types.int32,
            types.float64,
            types.CPointer(types.complex128),
            types.CPointer(types.complex128),
            types.voidptr,
        )
    )
    def _fun_numba(neq, t, y, dy, ctx):
        lam1 = -1.0 + 2.0j
        lam2 = -2.0 + 1.0j
        c = 0.0 + 0.5j
        dy[0] = lam1 * y[0] + c * y[1]
        dy[1] = lam2 * y[1]

    return _fun_numba


@pytest.fixture(scope="module")
def numba_jac_dense():
    """Return a numba @cfunc dense Jacobian for the coupled ODE."""
    pytest.importorskip("numba", reason="numba not installed")
    from numba import cfunc, types  # noqa: F811

    @cfunc(
        types.void(
            types.int32,
            types.float64,
            types.CPointer(types.complex128),
            types.int32,
            types.int32,
            types.CPointer(types.complex128),
            types.int32,
            types.voidptr,
        )
    )
    def _jac_numba(neq, t, y, ml, mu, pd, nrowpd, ctx):
        # pd is column-major (F-order): element [i, j] at pd[i + j*nrowpd]
        lam1 = -1.0 + 2.0j
        lam2 = -2.0 + 1.0j
        c = 0.0 + 0.5j
        pd[0]              = lam1  # J[0,0]
        pd[nrowpd]         = c     # J[0,1]
        pd[1 + nrowpd]     = lam2  # J[1,1]

    return _jac_numba


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_numba_fun_all_modes(numba_fun, mode):
    """numba @cfunc RHS tested across all three output modes."""
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"

    sol = solve_complex_ivp(
        numba_fun, tspan, Y0,
        in_place=True, save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        ref = exact(TF)
        assert np.allclose(sol.y, ref, rtol=1e-5)
    else:
        _check(sol.t, sol.y)


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_numba_dense_jac_all_modes(numba_fun, numba_jac_dense, mode):
    """numba @cfunc RHS + dense Jacobian across all output modes."""
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"

    sol = solve_complex_ivp(
        numba_fun, tspan, Y0,
        in_place=True, jac=numba_jac_dense, save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        ref = exact(TF)
        assert np.allclose(sol.y, ref, rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 10. ZVODE_FUN_CTYPE / ZVODE_JAC_CTYPE — canonical prototype objects
# ===========================================================================


def test_zvode_fun_ctype_produces_working_callback():
    """Using ZVODE_FUN_CTYPE directly produces a callback solve_complex_ivp accepts."""

    @ZVODE_FUN_CTYPE
    def fun(neq, t, y_ptr, dy_ptr, ctx):
        # rotation: dy/dt = i*y
        buf_y  = (ctypes.c_double * (2 * neq)).from_address(y_ptr)
        buf_dy = (ctypes.c_double * (2 * neq)).from_address(dy_ptr)
        for i in range(neq):
            yr, yi = buf_y[2 * i], buf_y[2 * i + 1]
            buf_dy[2 * i]     = -yi
            buf_dy[2 * i + 1] =  yr

    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    sol = solve_complex_ivp(fun, [0.0, np.pi], y0, in_place=True,
                            rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol.y[0, -1], np.exp(1j * np.pi), atol=1e-6)


def test_zvode_jac_ctype_attribute():
    """ZVODE_JAC_CTYPE is a ctypes function type with 8 arguments."""

    @ZVODE_JAC_CTYPE
    def dummy_jac(neq, t, y, ml, mu, pd, nrowpd, ctx):
        pass

    assert len(dummy_jac._argtypes_) == 8
    assert dummy_jac._restype_ is None


# ===========================================================================
# 11. check_cfunc_signature — validation helper
# ===========================================================================


def test_check_cfunc_signature_valid_fun():
    """check_cfunc_signature does not raise for a correctly typed RHS."""
    check_cfunc_signature(_fun_ctypes, kind="fun")  # must not raise


def test_check_cfunc_signature_valid_jac():
    """check_cfunc_signature does not raise for a correctly typed Jacobian."""
    check_cfunc_signature(_jac_dense_ctypes, kind="jac")  # must not raise


def test_check_cfunc_signature_wrong_return_type():
    """Non-void return type raises ValueError."""
    bad_proto = ctypes.CFUNCTYPE(
        ctypes.c_int,    # returns int, should be void
        ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )

    @bad_proto
    def bad_fun(neq, t, y, dy, ctx):
        return 0

    with pytest.raises(ValueError, match="void"):
        check_cfunc_signature(bad_fun)


def test_check_cfunc_signature_missing_ctx():
    """Wrong argument count (missing ctx) raises ValueError naming the count."""
    bad_proto = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p,  # 4 args, should be 5
    )

    @bad_proto
    def bad_fun(neq, t, y, dy):
        pass

    with pytest.raises(ValueError, match="5"):
        check_cfunc_signature(bad_fun)


def test_check_cfunc_signature_neq_wrong_type():
    """c_int64 for neq raises ValueError naming 'neq'."""
    bad_proto = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int64,  # neq should be c_int (int32)
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )

    @bad_proto
    def bad_fun(neq, t, y, dy, ctx):
        pass

    with pytest.raises(ValueError, match="neq"):
        check_cfunc_signature(bad_fun)


def test_check_cfunc_signature_t_wrong_type():
    """c_float for t raises ValueError naming 't'."""
    bad_proto = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int,
        ctypes.c_float,  # t should be c_double
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )

    @bad_proto
    def bad_fun(neq, t, y, dy, ctx):
        pass

    with pytest.raises(ValueError, match="'t'"):
        check_cfunc_signature(bad_fun)


def test_check_cfunc_signature_jac_wrong_ml():
    """c_int64 for ml in a jac callback raises ValueError naming 'ml'."""
    bad_jac_proto = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int, ctypes.c_double, ctypes.c_void_p,
        ctypes.c_int64,  # ml should be c_int
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    )

    @bad_jac_proto
    def bad_jac(neq, t, y, ml, mu, pd, nrowpd, ctx):
        pass

    with pytest.raises(ValueError, match="ml"):
        check_cfunc_signature(bad_jac, kind="jac")


def test_check_cfunc_signature_wrong_kind():
    """Invalid kind argument raises ValueError."""
    with pytest.raises(ValueError, match="kind"):
        check_cfunc_signature(_fun_ctypes, kind="rhs")


def test_check_cfunc_signature_non_cfunc_raises():
    """Passing a plain Python callable raises TypeError."""
    with pytest.raises(TypeError, match="ctypes"):
        check_cfunc_signature(lambda t, y, dy: None)


def test_check_cfunc_signature_numba_passes_through():
    """numba @cfunc passes check_cfunc_signature without inspection."""
    pytest.importorskip("numba", reason="numba not installed")
    from numba import cfunc, types

    @cfunc(
        types.void(
            types.int32, types.float64,
            types.CPointer(types.complex128),
            types.CPointer(types.complex128),
            types.voidptr,
        )
    )
    def nb_fun(neq, t, y, dy, ctx):
        dy[0] = y[0]

    check_cfunc_signature(nb_fun)  # must not raise
