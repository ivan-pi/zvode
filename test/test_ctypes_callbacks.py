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

from _shared import (
    LAM1,
    LAM2,
    C,
    Y0,
    T0,
    TF,
    LBAND,
    UBAND,
    RTOL,
    ATOL,
    coupled_exact as _exact,
    assert_coupled as _check,
    coupled_fun as _python_rhs,
    coupled_jac_dense as _python_jac,
    ro128 as _ro128,
    rw128 as _rw128,
)

# Problem: coupled 2-component complex ODE (defined in _shared.py, along with its
# analytic solution and the Python RHS/Jacobian used by the mixed-mode tests).

# Parameters array kept alive for the entire module; _CTX is a c_void_p
# pointing at its first element (a complex128[3] = [LAM1, LAM2, C]).
_PARAMS = np.array([LAM1, LAM2, C], dtype=np.complex128)
_CTX = ctypes.cast(_PARAMS.ctypes.data, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# Expected ctypes CFUNCTYPE definitions
# (matches design spec; these are now the canonical zvode exports)
# ---------------------------------------------------------------------------

_ZVODE_FUN_CTYPE = ctypes.CFUNCTYPE(
    None,  # void return
    ctypes.c_int,  # neq
    ctypes.c_double,  # t
    ctypes.c_void_p,  # const double complex *y
    ctypes.c_void_p,  # double complex *dy
    ctypes.c_void_p,  # void *ctx
)

_ZVODE_JAC_CTYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,  # neq
    ctypes.c_double,  # t
    ctypes.c_void_p,  # const double complex *y
    ctypes.c_int,  # ml
    ctypes.c_int,  # mu
    ctypes.c_void_p,  # double complex *pd  (column-major)
    ctypes.c_int,  # nrowpd
    ctypes.c_void_p,  # void *ctx
)


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
    pd[mu, 0] = LAM1  # df[0]/dy[0]
    pd[mu - 1, 1] = C  # df[0]/dy[1]
    pd[mu, 1] = LAM2  # df[1]/dy[1]


# ---------------------------------------------------------------------------
# Callbacks — parameterized via ctx (void * → complex128[3]: [LAM1, LAM2, C])
# ---------------------------------------------------------------------------


@_ZVODE_FUN_CTYPE
def _fun_ctx(neq, t, y_ptr, dy_ptr, ctx):
    p = _ro128(ctx, 3)  # p = [lam1, lam2, c]
    y = _ro128(y_ptr, neq)
    dy = _rw128(dy_ptr, neq)
    dy[0] = p[0] * y[0] + p[2] * y[1]
    dy[1] = p[1] * y[1]


@_ZVODE_JAC_CTYPE
def _jac_dense_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = _ro128(ctx, 3)
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[0, 0] = p[0]  # LAM1
    pd[0, 1] = p[2]  # C
    pd[1, 1] = p[1]  # LAM2


@_ZVODE_JAC_CTYPE
def _jac_banded_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = _ro128(ctx, 3)
    pd = _rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[mu, 0] = p[0]  # df[0]/dy[0]
    pd[mu - 1, 1] = p[2]  # df[0]/dy[1]
    pd[mu, 1] = p[1]  # df[1]/dy[1]


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
    sol = solve_complex_ivp(_fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL, save_steps=False)
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
    sol = solve_complex_ivp(_fun, [T0, TF], Y0, jac=_jac_dense, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 4. Banded Jacobian, no ctx
# ---------------------------------------------------------------------------


def test_banded_jac():
    """Compiled RHS + compiled banded Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun,
        [T0, TF],
        Y0,
        jac=_jac_banded,
        lband=LBAND,
        uband=UBAND,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 5. Parameterized via ctx, no Jacobian
# ---------------------------------------------------------------------------


def test_fun_ctx():
    """Compiled RHS parameterized through ctx; no Jacobian."""
    sol = solve_complex_ivp(_fun_ctx, [T0, TF], Y0, ctx=_CTX, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 6. Dense Jacobian with ctx
# ---------------------------------------------------------------------------


def test_dense_jac_ctx():
    """Compiled RHS + compiled dense Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx,
        [T0, TF],
        Y0,
        jac=_jac_dense_ctx,
        ctx=_CTX,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 7. Banded Jacobian with ctx
# ---------------------------------------------------------------------------


def test_banded_jac_ctx():
    """Compiled RHS + compiled banded Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx,
        [T0, TF],
        Y0,
        jac=_jac_banded_ctx,
        lband=LBAND,
        uband=UBAND,
        ctx=_CTX,
        rtol=RTOL,
        atol=ATOL,
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
        _python_rhs,
        [T0, TF],
        Y0,
        jac=_jac_dense,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_mixed_python_rhs_compiled_banded_jac():
    """Python return-value RHS with a compiled banded Jacobian."""
    sol = solve_complex_ivp(
        _python_rhs,
        [T0, TF],
        Y0,
        jac=_jac_banded,
        lband=LBAND,
        uband=UBAND,
        rtol=RTOL,
        atol=ATOL,
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
        solve_complex_ivp(_python_rhs, [T0, TF], Y0, jac=_python_jac, ctx=_CTX)


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
