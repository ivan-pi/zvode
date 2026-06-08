"""Tests for compiled (ctypes / numba) callbacks in solve_complex_ivp.

Problem: 2-component coupled linear ODE
    dy[0]/dt = LAM1*y[0] + C*y[1]
    dy[1]/dt = LAM2*y[1]

Analytic solution:
    y[1](t) = Y0[1] * exp(LAM2*t)
    y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t)
    B = C*Y0[1]/(LAM2-LAM1),  A = Y0[0] - B

All module-level ctypes callbacks are defined at module scope so that ctypes
does not garbage-collect their underlying C thunks during testing.
"""

import ctypes

import numpy as np
import pytest

from zvode import solve_complex_ivp, ZVODE_FUN_CTYPE, ZVODE_JAC_CTYPE

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

LAM1 = -1 + 2j
LAM2 = -2 + 1j
C = 0.5j
Y0 = np.array([1.0 + 0j, 0.0 + 1j])
T0 = 0.0
TF = 2.0
LBAND = 0
UBAND = 1
RTOL = 1e-8
ATOL = 1e-10

_B = C * Y0[1] / (LAM2 - LAM1)
_A = Y0[0] - _B


def exact(t):
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
# ctypes callbacks using ZVODE_FUN_CTYPE / ZVODE_JAC_CTYPE
#
# ZVODE_FUN_CTYPE uses c_void_p for the y and dy pointer arguments,
# so we access elements via (c_double * 2n).from_address(ptr).
# ---------------------------------------------------------------------------

@ZVODE_FUN_CTYPE
def _fun_ctypes(neq, t, y_ptr, dy_ptr, ctx):
    buf_y  = (ctypes.c_double * (2 * neq)).from_address(y_ptr)
    buf_dy = (ctypes.c_double * (2 * neq)).from_address(dy_ptr)
    y0c = complex(buf_y[0], buf_y[1])
    y1c = complex(buf_y[2], buf_y[3])
    res0 = LAM1 * y0c + C * y1c
    res1 = LAM2 * y1c
    buf_dy[0] = res0.real; buf_dy[1] = res0.imag
    buf_dy[2] = res1.real; buf_dy[3] = res1.imag


@ZVODE_JAC_CTYPE
def _jac_dense_ctypes(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # F-order: element J[i,j] at offset i + j*nrowpd (in complex128 units)
    buf = (ctypes.c_double * (2 * nrowpd * neq)).from_address(pd_ptr)
    def setc(row, col, val):
        k = row + col * nrowpd
        buf[2*k] = val.real; buf[2*k+1] = val.imag
    setc(0, 0, LAM1)  # df0/dy0
    setc(0, 1, C)     # df0/dy1
    setc(1, 1, LAM2)  # df1/dy1


@ZVODE_JAC_CTYPE
def _jac_banded_ctypes(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # Banded storage: J[i,j] -> pd[mu+i-j, j]
    # ml=0, mu=1: diagonal and one superdiagonal
    buf = (ctypes.c_double * (2 * nrowpd * neq)).from_address(pd_ptr)
    def setc(row, col, val):
        k = row + col * nrowpd
        buf[2*k] = val.real; buf[2*k+1] = val.imag
    setc(mu + 0 - 0, 0, LAM1)  # J[0,0]: row=mu=1, col=0
    setc(mu + 0 - 1, 1, C)     # J[0,1]: row=mu-1=0, col=1
    setc(mu + 1 - 1, 1, LAM2)  # J[1,1]: row=mu=1, col=1


# ---------------------------------------------------------------------------
# Parametric RHS using ctx to pass eigenvalues
# (neq=1 scalar: dy/dt = lam*y, lam passed as a complex128 via ctx)
# ---------------------------------------------------------------------------

@ZVODE_FUN_CTYPE
def _fun_ctx(neq, t, y_ptr, dy_ptr, ctx):
    # ctx points to a complex128 (two doubles: real then imag)
    lam_buf = (ctypes.c_double * 2).from_address(ctx)
    lam = complex(lam_buf[0], lam_buf[1])
    buf_y  = (ctypes.c_double * 2).from_address(y_ptr)
    buf_dy = (ctypes.c_double * 2).from_address(dy_ptr)
    res = lam * complex(buf_y[0], buf_y[1])
    buf_dy[0] = res.real
    buf_dy[1] = res.imag


# ===========================================================================
# 1. ctypes RHS — all three output modes
# ===========================================================================


def test_ctypes_fun_steps():
    sol = solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    assert sol.t[0] == T0 and sol.t[-1] == pytest.approx(TF)
    assert sol.y.shape == (2, len(sol.t))
    _check(sol.t, sol.y)


def test_ctypes_fun_endpoint():
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, save_steps=False, rtol=RTOL, atol=ATOL,
    )
    assert np.isscalar(sol.t) and sol.t == pytest.approx(TF)
    assert sol.y.ndim == 1 and sol.y.shape == (2,)
    assert np.allclose(sol.y, exact(TF), rtol=1e-5)


def test_ctypes_fun_knots():
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(_fun_ctypes, tspan, Y0, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(sol.t, tspan)
    assert sol.y.shape == (2, 11)
    _check(sol.t, sol.y)


# ===========================================================================
# 2. ctypes RHS + dense Jacobian — all three output modes
# ===========================================================================


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_ctypes_dense_jac(mode):
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0,
        jac=_jac_dense_ctypes, save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        assert np.allclose(sol.y, exact(TF), rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 3. ctypes RHS + banded Jacobian — all three output modes
# ===========================================================================


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_ctypes_banded_jac(mode):
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, Y0,
        jac=_jac_banded_ctypes, lband=LBAND, uband=UBAND,
        save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        assert np.allclose(sol.y, exact(TF), rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 4. ctx parameter — user data passed through void pointer
# ===========================================================================


def test_ctx_passed_to_compiled_fun():
    """Eigenvalue parameter passed via ctx produces the correct trajectory."""
    lam = -0.5 + 1.0j
    params = np.array([lam.real, lam.imag])  # two doubles
    ctx_ptr = ctypes.c_void_p(params.ctypes.data)
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol = solve_complex_ivp(
        _fun_ctx, [0.0, 1.0], y0, ctx=ctx_ptr, rtol=1e-8, atol=1e-10,
    )
    expected = y0[0] * np.exp(lam * sol.t)
    np.testing.assert_allclose(sol.y[0], expected, rtol=1e-5)


def test_ctx_none_gives_null():
    """ctx=None (default) does not raise and integration succeeds."""
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, ctx=None, rtol=RTOL, atol=ATOL,
    )
    assert sol.t[-1] == pytest.approx(TF)
    _check(sol.t, sol.y)


def test_ctx_with_python_callable_warns():
    """ctx is ignored (with UserWarning) when both fun and jac are Python callables."""
    def py_fun(t, y):
        return LAM1 * y[0] * np.ones(1, dtype=complex)

    ctx_ptr = ctypes.c_void_p(0)
    y0 = np.array([1.0 + 0j])
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(py_fun, [0.0, 0.1], y0, ctx=ctx_ptr)


def test_ctx_wrong_type_raises():
    """Passing an unsupported ctx type raises TypeError."""
    with pytest.raises(TypeError, match="c_void_p"):
        solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, ctx=12345)


# ===========================================================================
# 5. Mixed Python / compiled callbacks
# ===========================================================================


def test_python_fun_ctypes_jac():
    """Python RHS + ctypes dense Jacobian completes without error."""
    def py_fun(t, y):
        return np.array([LAM1 * y[0] + C * y[1], LAM2 * y[1]])

    sol = solve_complex_ivp(
        py_fun, [T0, TF], Y0,
        jac=_jac_dense_ctypes, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_ctypes_fun_python_jac():
    """ctypes RHS + Python dense Jacobian completes without error."""
    def py_jac(t, y):
        return np.array([[LAM1, C], [0.0, LAM2]])

    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0,
        jac=py_jac, rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ===========================================================================
# 6. Output type / shape assertions
# ===========================================================================


def test_steps_result_shapes():
    sol = solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    assert isinstance(sol.t, np.ndarray) and sol.t.ndim == 1
    assert sol.y.ndim == 2 and sol.y.shape == (2, len(sol.t))
    assert sol.y.dtype == np.complex128


def test_endpoint_result_shapes():
    sol = solve_complex_ivp(
        _fun_ctypes, [T0, TF], Y0, save_steps=False, rtol=RTOL, atol=ATOL,
    )
    assert np.isscalar(sol.t)
    assert sol.y.ndim == 1 and sol.y.shape == (2,)


def test_knots_result_shapes():
    tspan = np.linspace(T0, TF, 7)
    sol = solve_complex_ivp(_fun_ctypes, tspan, Y0, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(sol.t, tspan)
    assert sol.y.shape == (2, 7)


# ===========================================================================
# 7. Statistics counters
# ===========================================================================


def test_stats_present_and_positive():
    sol = solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    assert sol.nsteps > 0
    assert sol.nfev > 0
    assert sol.njev >= 0
    assert sol.nlu >= 0


# ===========================================================================
# 8. Backward integration
# ===========================================================================


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_ctypes_backward(mode):
    y_tf = exact(TF).astype(np.complex128)
    tspan = np.linspace(TF, T0, 9) if mode == "knots" else [TF, T0]
    save = mode == "steps"
    sol = solve_complex_ivp(
        _fun_ctypes, tspan, y_tf,
        save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        assert sol.t == pytest.approx(T0)
        assert np.allclose(sol.y, exact(T0), rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 9. refine > 1
# ===========================================================================


def test_ctypes_refine():
    REFINE = 4
    sol_base = solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL, refine=1)
    sol_ref  = solve_complex_ivp(_fun_ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL, refine=REFINE)
    n_steps = len(sol_base.t) - 1
    assert len(sol_ref.t) == len(sol_base.t) + n_steps * (REFINE - 1)
    _check(sol_ref.t, sol_ref.y)


# ===========================================================================
# 10. Numba @cfunc tests  (skipped when numba is not installed)
#
# numba CFunc objects are not ctypes._CFuncPtr instances; pass .ctypes
# to get a ctypes function pointer that solve_complex_ivp detects.
# ===========================================================================


@pytest.fixture(scope="module")
def numba_fun():
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
    def _fun_nb(neq, t, y, dy, ctx):
        lam1 = -1.0 + 2.0j
        lam2 = -2.0 + 1.0j
        c    =  0.0 + 0.5j
        dy[0] = lam1 * y[0] + c * y[1]
        dy[1] = lam2 * y[1]

    return _fun_nb


@pytest.fixture(scope="module")
def numba_jac_dense():
    pytest.importorskip("numba", reason="numba not installed")
    from numba import cfunc, types

    @cfunc(
        types.void(
            types.int32, types.float64,
            types.CPointer(types.complex128),
            types.int32, types.int32,
            types.CPointer(types.complex128),
            types.int32, types.voidptr,
        )
    )
    def _jac_nb(neq, t, y, ml, mu, pd, nrowpd, ctx):
        lam1 = -1.0 + 2.0j
        lam2 = -2.0 + 1.0j
        c    =  0.0 + 0.5j
        pd[0]          = lam1  # J[0,0]
        pd[nrowpd]     = c     # J[0,1]
        pd[1 + nrowpd] = lam2  # J[1,1]

    return _jac_nb


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_numba_fun_all_modes(numba_fun, mode):
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"
    sol = solve_complex_ivp(
        numba_fun.ctypes, tspan, Y0,
        save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        assert np.allclose(sol.y, exact(TF), rtol=1e-5)
    else:
        _check(sol.t, sol.y)


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_numba_dense_jac_all_modes(numba_fun, numba_jac_dense, mode):
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = mode == "steps"
    sol = solve_complex_ivp(
        numba_fun.ctypes, tspan, Y0,
        jac=numba_jac_dense.ctypes, save_steps=save, rtol=RTOL, atol=ATOL,
    )
    if mode == "endpoint":
        assert np.allclose(sol.y, exact(TF), rtol=1e-5)
    else:
        _check(sol.t, sol.y)


# ===========================================================================
# 11. ZVODE_FUN_CTYPE / ZVODE_JAC_CTYPE canonical prototypes
# ===========================================================================


def test_zvode_fun_ctype_working_callback():
    """ZVODE_FUN_CTYPE-decorated callback is accepted by solve_complex_ivp."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    @ZVODE_FUN_CTYPE
    def rotation(neq, t, y_ptr, dy_ptr, ctx):
        buf_y  = (ctypes.c_double * 2).from_address(y_ptr)
        buf_dy = (ctypes.c_double * 2).from_address(dy_ptr)
        yr, yi = buf_y[0], buf_y[1]
        buf_dy[0] = -yi
        buf_dy[1] =  yr

    sol = solve_complex_ivp(rotation, [0.0, np.pi], y0, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol.y[0, -1], np.exp(1j * np.pi), atol=1e-6)


def test_zvode_jac_ctype_argtypes():
    """ZVODE_JAC_CTYPE has 8 arguments and void return."""
    @ZVODE_JAC_CTYPE
    def dummy(neq, t, y, ml, mu, pd, nrowpd, ctx):
        pass

    assert len(dummy._argtypes_) == 8
    assert dummy._restype_ is None
