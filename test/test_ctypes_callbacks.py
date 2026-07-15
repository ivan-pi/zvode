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

from shared import COUPLED, ro128, rw128

# Problem: coupled 2-component complex ODE (defined in shared.py, along with its
# analytic solution and the Python RHS/Jacobian used by the mixed-mode tests).

# Parameters array kept alive for the entire module; _CTX is a c_void_p
# pointing at its first element (a complex128[3] = [LAM1, LAM2, C]).
_PARAMS = np.array([COUPLED.lam1, COUPLED.lam2, COUPLED.c], dtype=np.complex128)
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
    y = ro128(y_ptr, neq)
    dy = rw128(dy_ptr, neq)
    dy[0] = COUPLED.lam1 * y[0] + COUPLED.c * y[1]
    dy[1] = COUPLED.lam2 * y[1]


@_ZVODE_JAC_CTYPE
def _jac_dense(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # pd is column-major: pd[i, j] = df_i/dy_j  (dense, nrowpd == neq)
    pd = rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[0, 0] = COUPLED.lam1
    pd[0, 1] = COUPLED.c
    pd[1, 1] = COUPLED.lam2


@_ZVODE_JAC_CTYPE
def _jac_banded(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    # Band storage: pd[mu + i - j, j] = df_i/dy_j
    pd = rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[mu, 0] = COUPLED.lam1  # df[0]/dy[0]
    pd[mu - 1, 1] = COUPLED.c  # df[0]/dy[1]
    pd[mu, 1] = COUPLED.lam2  # df[1]/dy[1]


# ---------------------------------------------------------------------------
# Callbacks — parameterized via ctx (void * → complex128[3]: [LAM1, LAM2, C])
# ---------------------------------------------------------------------------


@_ZVODE_FUN_CTYPE
def _fun_ctx(neq, t, y_ptr, dy_ptr, ctx):
    p = ro128(ctx, 3)  # p = [lam1, lam2, c]
    y = ro128(y_ptr, neq)
    dy = rw128(dy_ptr, neq)
    dy[0] = p[0] * y[0] + p[2] * y[1]
    dy[1] = p[1] * y[1]


@_ZVODE_JAC_CTYPE
def _jac_dense_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = ro128(ctx, 3)
    pd = rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
    pd[0, 0] = p[0]  # LAM1
    pd[0, 1] = p[2]  # C
    pd[1, 1] = p[1]  # LAM2


@_ZVODE_JAC_CTYPE
def _jac_banded_ctx(neq, t, y_ptr, ml, mu, pd_ptr, nrowpd, ctx):
    p = ro128(ctx, 3)
    pd = rw128(pd_ptr, nrowpd * neq).reshape((nrowpd, neq), order="F")
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
    sol = solve_complex_ivp(_fun, COUPLED.tspan, COUPLED.y0, **COUPLED.tols)
    COUPLED.assert_close(sol.t, sol.y)


def test_fun_only_endpoint():
    """Compiled RHS without Jacobian; endpoint-only mode."""
    sol = solve_complex_ivp(
        _fun, COUPLED.tspan, COUPLED.y0, **COUPLED.tols, save_steps=False
    )
    np.testing.assert_allclose(sol.y, COUPLED.exact(COUPLED.tf), rtol=1e-5)


def test_fun_only_knots():
    """Compiled RHS without Jacobian; output at requested knots."""
    tspan = np.linspace(COUPLED.t0, COUPLED.tf, 11)
    sol = solve_complex_ivp(_fun, tspan, COUPLED.y0, **COUPLED.tols)
    np.testing.assert_array_equal(sol.t, tspan)
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 3. Dense Jacobian, no ctx
# ---------------------------------------------------------------------------


def test_dense_jac():
    """Compiled RHS + compiled dense Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun, COUPLED.tspan, COUPLED.y0, jac=_jac_dense, **COUPLED.tols
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 4. Banded Jacobian, no ctx
# ---------------------------------------------------------------------------


def test_banded_jac():
    """Compiled RHS + compiled banded Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun,
        COUPLED.tspan,
        COUPLED.y0,
        jac=_jac_banded,
        lband=COUPLED.lband,
        uband=COUPLED.uband,
        rtol=COUPLED.rtol,
        atol=COUPLED.atol,
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 5. Parameterized via ctx, no Jacobian
# ---------------------------------------------------------------------------


def test_fun_ctx():
    """Compiled RHS parameterized through ctx; no Jacobian."""
    sol = solve_complex_ivp(
        _fun_ctx, COUPLED.tspan, COUPLED.y0, ctx=_CTX, **COUPLED.tols
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 6. Dense Jacobian with ctx
# ---------------------------------------------------------------------------


def test_dense_jac_ctx():
    """Compiled RHS + compiled dense Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx,
        COUPLED.tspan,
        COUPLED.y0,
        jac=_jac_dense_ctx,
        ctx=_CTX,
        rtol=COUPLED.rtol,
        atol=COUPLED.atol,
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 7. Banded Jacobian with ctx
# ---------------------------------------------------------------------------


def test_banded_jac_ctx():
    """Compiled RHS + compiled banded Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx,
        COUPLED.tspan,
        COUPLED.y0,
        jac=_jac_banded_ctx,
        lband=COUPLED.lband,
        uband=COUPLED.uband,
        ctx=_CTX,
        rtol=COUPLED.rtol,
        atol=COUPLED.atol,
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 8. Mixed mode: Python RHS + compiled Jacobian
#
# The design spec allows gradual porting: one callback can be Python while
# the other is compiled.
# ---------------------------------------------------------------------------


def test_mixed_python_rhs_compiled_dense_jac():
    """Python return-value RHS with a compiled dense Jacobian."""
    sol = solve_complex_ivp(
        COUPLED.fun,
        COUPLED.tspan,
        COUPLED.y0,
        jac=_jac_dense,
        rtol=COUPLED.rtol,
        atol=COUPLED.atol,
    )
    COUPLED.assert_close(sol.t, sol.y)


def test_mixed_python_rhs_compiled_banded_jac():
    """Python return-value RHS with a compiled banded Jacobian."""
    sol = solve_complex_ivp(
        COUPLED.fun,
        COUPLED.tspan,
        COUPLED.y0,
        jac=_jac_banded,
        lband=COUPLED.lband,
        uband=COUPLED.uband,
        rtol=COUPLED.rtol,
        atol=COUPLED.atol,
    )
    COUPLED.assert_close(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 9. ctx keyword validation
# ---------------------------------------------------------------------------


def test_ctx_none_is_null():
    """ctx=None passes NULL to compiled callbacks and must not raise."""
    sol = solve_complex_ivp(_fun, COUPLED.tspan, COUPLED.y0, ctx=None, **COUPLED.tols)
    COUPLED.assert_close(sol.t, sol.y)


def test_ctx_with_python_rhs_warns():
    """ctx != None alongside a Python RHS must issue a UserWarning."""
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(COUPLED.fun, COUPLED.tspan, COUPLED.y0, ctx=_CTX)


def test_ctx_with_both_python_warns():
    """ctx != None when both fun and jac are Python callables must warn.

    ctx is meaningless for Python callbacks; passing it is likely a mistake,
    so a UserWarning must be issued regardless of whether jac is also present.
    """
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(
            COUPLED.fun, COUPLED.tspan, COUPLED.y0, jac=COUPLED.jac_dense, ctx=_CTX
        )


def test_ctx_invalid_type_raises():
    """A non-c_void_p, non-None ctx must raise TypeError immediately.

    The error message must mention c_void_p to confirm it's a ctx-type error
    rather than an "unexpected keyword argument" from the missing implementation.
    """
    with pytest.raises(TypeError, match="c_void_p"):
        solve_complex_ivp(_fun, COUPLED.tspan, COUPLED.y0, ctx=42, **COUPLED.tols)


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
        solve_complex_ivp(_fun, COUPLED.tspan, COUPLED.y0, **COUPLED.tols)
