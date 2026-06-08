"""Tests for the ctypes compiled callback path.

These tests exercise the API specified in docs/compiled-callbacks-design.md.
All integration and export tests verify the fully implemented behaviour:

- No ``in_place`` parameter: callback kind is detected from type alone.
- New ``ctx`` keyword: an optional ``ctypes.c_void_p`` passed to both
  compiled callbacks on every invocation (NULL when ``ctx=None``).
- ``ZVODE_FUN_CTYPE`` and ``ZVODE_JAC_CTYPE`` exported from ``zvode``.
"""

import ctypes

import numpy as np
import pytest

from zvode import solve_complex_ivp

# ---------------------------------------------------------------------------
# Problem: coupled 2-component complex ODE
#   dy[0]/dt = LAM1*y[0] + C*y[1]
#   dy[1]/dt = LAM2*y[1]
# Exact solution:
#   y[1](t) = Y0[1]*exp(LAM2*t)
#   y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t),  B = C*Y0[1]/(LAM2-LAM1)
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

# Parameters array kept alive for the entire module; _CTX is a c_void_p
# pointing at its first element (a complex128[3] = [LAM1, LAM2, C]).
_PARAMS = np.array([LAM1, LAM2, C], dtype=np.complex128)
_CTX = ctypes.cast(_PARAMS.ctypes.data, ctypes.c_void_p)


def _exact(t):
    t = np.asarray(t, dtype=float)
    return np.array([
        _A * np.exp(LAM1 * t) + _B * np.exp(LAM2 * t),
        Y0[1] * np.exp(LAM2 * t),
    ])


def _check(t_arr, y_arr, sol_rtol=1e-5):
    ref = _exact(t_arr)
    assert np.allclose(y_arr, ref, rtol=sol_rtol), (
        f"max err={np.max(np.abs(y_arr - ref)):.2e}"
    )


# ---------------------------------------------------------------------------
# Expected ctypes CFUNCTYPE definitions
# (matches design spec; these are now the canonical zvode exports)
# ---------------------------------------------------------------------------

_ZVODE_FUN_CTYPE = ctypes.CFUNCTYPE(
    None,            # void return
    ctypes.c_int,    # neq
    ctypes.c_double, # t
    ctypes.c_void_p, # const double complex *y
    ctypes.c_void_p, # double complex *dy
    ctypes.c_void_p, # void *ctx
)

_ZVODE_JAC_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,    # neq
    ctypes.c_double, # t
    ctypes.c_void_p, # const double complex *y
    ctypes.c_int,    # ml
    ctypes.c_int,    # mu
    ctypes.c_void_p, # double complex *pd  (column-major)
    ctypes.c_int,    # nrowpd
    ctypes.c_void_p, # void *ctx
)


# ---------------------------------------------------------------------------
# Low-level helpers: create numpy views over raw C memory addresses
# ---------------------------------------------------------------------------

def _ro128(addr, count):
    """Read-only complex128 view of *count* elements at *addr*."""
    buf = (ctypes.c_double * (2 * count)).from_address(addr)
    return np.frombuffer(buf, dtype=np.complex128)


def _rw128(addr, count):
    """Writable complex128 view of *count* elements at *addr*."""
    buf = (ctypes.c_double * (2 * count)).from_address(addr)
    return np.ctypeslib.as_array(buf).view(np.complex128)


# ---------------------------------------------------------------------------
# Callbacks — hardcoded parameters (ctx ignored)
# ---------------------------------------------------------------------------

@_ZVODE_FUN_CTYPE
def _fun(neq, t, y_ptr, dy_ptr, ctx):
    y = _ro128(y_ptr, neq)
    dy = _rw128(dy_ptr, neq)
    dy[0] = LAM1 * y[0] + C * y[1]
    dy[1] = LAM2 * y[1]


@_ZVODE_JAC_CTYPE
def _jac_dense(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # pd is column-major: pd[i, j] = df_i/dy_j  (dense, nrowpd == neq)
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[0, 0] = LAM1
    pd[0, 1] = C
    pd[1, 1] = LAM2


@_ZVODE_JAC_CTYPE
def _jac_banded(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # Band storage: pd[mu + i - j, j] = df_i/dy_j
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[mu,     0] = LAM1   # df[0]/dy[0]
    pd[mu - 1, 1] = C      # df[0]/dy[1]
    pd[mu,     1] = LAM2   # df[1]/dy[1]


# ---------------------------------------------------------------------------
# Callbacks — parameterized via ctx (void * → complex128[3]: [LAM1, LAM2, C])
# ---------------------------------------------------------------------------

@_ZVODE_FUN_CTYPE
def _fun_ctx(neq, t, y_ptr, dy_ptr, ctx):
    p = _ro128(ctx, 3)              # p = [lam1, lam2, c]
    y = _ro128(y_ptr, neq)
    dy = _rw128(dy_ptr, neq)
    dy[0] = p[0] * y[0] + p[2] * y[1]
    dy[1] = p[1] * y[1]


@_ZVODE_JAC_CTYPE
def _jac_dense_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = _ro128(ctx, 3)
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[0, 0] = p[0]   # LAM1
    pd[0, 1] = p[2]   # C
    pd[1, 1] = p[1]   # LAM2


@_ZVODE_JAC_CTYPE
def _jac_banded_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = _ro128(ctx, 3)
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[mu,     0] = p[0]   # df[0]/dy[0]
    pd[mu - 1, 1] = p[2]   # df[0]/dy[1]
    pd[mu,     1] = p[1]   # df[1]/dy[1]


# ---------------------------------------------------------------------------
# Python callbacks for mixed-mode and ctx-warning tests
# ---------------------------------------------------------------------------

def _python_rhs(t, y):
    dy = np.empty(len(y), dtype=np.complex128)
    dy[0] = LAM1 * y[0] + C * y[1]
    dy[1] = LAM2 * y[1]
    return dy


def _python_jac(t, y):
    pd = np.zeros((len(y), len(y)), dtype=np.complex128)
    pd[0, 0] = LAM1
    pd[0, 1] = C
    pd[1, 1] = LAM2
    return pd


# ---------------------------------------------------------------------------
# 1. Export checks
# ---------------------------------------------------------------------------


def test_export_fun_ctype():
    """zvode.ZVODE_FUN_CTYPE must be importable and be a ctypes CFUNCTYPE."""
    from zvode import ZVODE_FUN_CTYPE  # noqa: F401

    assert issubclass(ZVODE_FUN_CTYPE, ctypes._CFuncPtr)


def test_export_jac_ctype():
    """zvode.ZVODE_JAC_CTYPE must be importable and be a ctypes CFUNCTYPE."""
    from zvode import ZVODE_JAC_CTYPE  # noqa: F401

    assert issubclass(ZVODE_JAC_CTYPE, ctypes._CFuncPtr)


# ---------------------------------------------------------------------------
# 2. RHS only, no ctx
# ---------------------------------------------------------------------------


def test_fun_only_steps():
    """Compiled RHS without Jacobian; collect all accepted steps."""
    sol = solve_complex_ivp(_fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


def test_fun_only_endpoint():
    """Compiled RHS without Jacobian; endpoint-only mode."""
    sol = solve_complex_ivp(
        _fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL, save_steps=False
    )
    np.testing.assert_allclose(sol.y, _exact(TF), rtol=1e-5)


def test_fun_only_knots():
    """Compiled RHS without Jacobian; output at requested knots."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(_fun, tspan, Y0, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(sol.t, tspan)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 3. Dense Jacobian, no ctx
# ---------------------------------------------------------------------------


def test_dense_jac():
    """Compiled RHS + compiled dense Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun, [T0, TF], Y0, jac=_jac_dense, rtol=RTOL, atol=ATOL
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 4. Banded Jacobian, no ctx
# ---------------------------------------------------------------------------


def test_banded_jac():
    """Compiled RHS + compiled banded Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun, [T0, TF], Y0,
        jac=_jac_banded, lband=LBAND, uband=UBAND,
        rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 5. Parameterized via ctx, no Jacobian
# ---------------------------------------------------------------------------


def test_fun_ctx():
    """Compiled RHS parameterized through ctx; no Jacobian."""
    sol = solve_complex_ivp(
        _fun_ctx, [T0, TF], Y0, ctx=_CTX, rtol=RTOL, atol=ATOL
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 6. Dense Jacobian with ctx
# ---------------------------------------------------------------------------


def test_dense_jac_ctx():
    """Compiled RHS + compiled dense Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx, [T0, TF], Y0,
        jac=_jac_dense_ctx, ctx=_CTX,
        rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 7. Banded Jacobian with ctx
# ---------------------------------------------------------------------------


def test_banded_jac_ctx():
    """Compiled RHS + compiled banded Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx, [T0, TF], Y0,
        jac=_jac_banded_ctx, lband=LBAND, uband=UBAND, ctx=_CTX,
        rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 8. Mixed mode: Python RHS + compiled Jacobian
#
# The design spec allows gradual porting: one callback can be Python while
# the other is compiled.
# ---------------------------------------------------------------------------


def test_mixed_python_rhs_compiled_dense_jac():
    """Python return-value RHS with a compiled dense Jacobian."""
    sol = solve_complex_ivp(
        _python_rhs, [T0, TF], Y0,
        jac=_jac_dense,
        rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_mixed_python_rhs_compiled_banded_jac():
    """Python return-value RHS with a compiled banded Jacobian."""
    sol = solve_complex_ivp(
        _python_rhs, [T0, TF], Y0,
        jac=_jac_banded, lband=LBAND, uband=UBAND,
        rtol=RTOL, atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 9. ctx keyword validation
# ---------------------------------------------------------------------------


def test_ctx_none_is_null():
    """ctx=None passes NULL to compiled callbacks and must not raise."""
    sol = solve_complex_ivp(_fun, [T0, TF], Y0, ctx=None, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


def test_ctx_with_python_rhs_warns():
    """ctx != None alongside a Python RHS must issue a UserWarning."""
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(_python_rhs, [T0, TF], Y0, ctx=_CTX)


def test_ctx_with_both_python_warns():
    """ctx != None when both fun and jac are Python callables must warn.

    ctx is meaningless for Python callbacks; passing it is likely a mistake,
    so a UserWarning must be issued regardless of whether jac is also present.
    """
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(
            _python_rhs, [T0, TF], Y0, jac=_python_jac, ctx=_CTX
        )


def test_ctx_invalid_type_raises():
    """A non-c_void_p, non-None ctx must raise TypeError immediately.

    The error message must mention c_void_p to confirm it's a ctx-type error
    rather than an "unexpected keyword argument" from the missing implementation.
    """
    with pytest.raises(TypeError, match="c_void_p"):
        solve_complex_ivp(_fun, [T0, TF], Y0, ctx=42, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 10. Python fallback backend with compiled callbacks
# ---------------------------------------------------------------------------


def test_python_backend_rejects_compiled_callbacks(monkeypatch):
    """Compiled callbacks with ZVODE_BACKEND=python raise RuntimeError.

    The Python fallback loop (_zvode_adaptive/_zvode_knots) can only accept
    Python callables.  Compiled callbacks are only supported through the C
    integration loop (drive_knots / drive_adaptive).
    """
    import zvode.solve as _solve

    monkeypatch.setattr(_solve, "_USE_C_KNOTS", False)
    with pytest.raises(RuntimeError, match="ZVODE_BACKEND"):
        solve_complex_ivp(_fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
