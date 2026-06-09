"""Tests for the numba @cfunc compiled callback path.

Verifies that numba-compiled callbacks integrate correctly via
solve_complex_ivp.  Numba users pass ``my_rhs.ctypes`` (a
``ctypes._CFuncPtr``) rather than the @cfunc object itself.

Two parameterization patterns from the design spec are exercised:

Pattern 3 — compiled closure:
    Parameters (LAM1, LAM2, C) are captured from the Python scope at
    compilation time.  ``ctx`` is present in the signature but ignored.

Explicit ctx:
    Parameters are stored in a numpy float64 array (real + imaginary parts
    interleaved) whose address is passed as ``ctx``.  Inside the @cfunc the
    void pointer is reinterpreted as a float64 array via ``numba.carray``.

The entire module is skipped when numba is not installed.
"""

import ctypes

import numpy as np
import pytest

numba = pytest.importorskip("numba")

import numba as nb  # noqa: E402 (after importorskip)
from numba import cfunc, types  # noqa: E402

from zvode import solve_complex_ivp  # noqa: E402

# ---------------------------------------------------------------------------
# Problem: coupled 2-component complex ODE (same as test_ctypes_callbacks.py)
#   dy[0]/dt = LAM1*y[0] + C*y[1]
#   dy[1]/dt = LAM2*y[1]
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

# Parameters stored as float64 [re(LAM1), im(LAM1), re(LAM2), im(LAM2), re(C), im(C)]
# for ctx-parameterized tests; complex128 is avoided to sidestep any
# voidptr ↔ CPointer(complex128) casting issues inside nopython mode.
_PARAMS_F64 = np.array(
    [LAM1.real, LAM1.imag, LAM2.real, LAM2.imag, C.real, C.imag],
    dtype=np.float64,
)
_CTX = ctypes.cast(_PARAMS_F64.ctypes.data, ctypes.c_void_p)


def _exact(t):
    t = np.asarray(t, dtype=float)
    return np.array(
        [
            _A * np.exp(LAM1 * t) + _B * np.exp(LAM2 * t),
            Y0[1] * np.exp(LAM2 * t),
        ]
    )


def _check(t_arr, y_arr, sol_rtol=1e-5):
    ref = _exact(t_arr)
    assert np.allclose(y_arr, ref, rtol=sol_rtol), (
        f"max err={np.max(np.abs(y_arr - ref)):.2e}"
    )


# ---------------------------------------------------------------------------
# Expected numba cfunc signatures (as per design spec; exported from zvode)
# ---------------------------------------------------------------------------

_zvode_fun_sig = types.void(
    types.int32,  # neq
    types.float64,  # t
    types.CPointer(types.complex128),  # const double complex *y
    types.CPointer(types.complex128),  # double complex *dy
    types.voidptr,  # void *ctx
)

_zvode_jac_sig = types.void(
    types.int32,  # neq
    types.float64,  # t
    types.CPointer(types.complex128),  # const double complex *y
    types.int32,  # ml
    types.int32,  # mu
    types.CPointer(types.complex128),  # double complex *pd  (column-major)
    types.int32,  # nrowpd
    types.voidptr,  # void *ctx
)


# ---------------------------------------------------------------------------
# Compiled closures (Pattern 3 in design spec): parameters are compile-time
# constants captured from the enclosing Python scope; ctx is ignored.
# ---------------------------------------------------------------------------


@cfunc(_zvode_fun_sig)
def _fun(neq, t, y, dy, ctx):
    dy[0] = LAM1 * y[0] + C * y[1]
    dy[1] = LAM2 * y[1]


@cfunc(_zvode_jac_sig)
def _jac_dense(neq, t, y, ml, mu, pd, nrowpd, ctx):
    # nb.farray creates a 2-D Fortran-order view: J[i,j] = df_i/dy_j
    J = nb.farray(pd, (nrowpd, neq))
    J[0, 0] = LAM1
    J[0, 1] = C
    J[1, 1] = LAM2


@cfunc(_zvode_jac_sig)
def _jac_banded(neq, t, y, ml, mu, pd, nrowpd, ctx):
    # Band storage: J[mu + i - j, j] = df_i/dy_j
    J = nb.farray(pd, (nrowpd, neq))
    J[mu, 0] = LAM1  # df[0]/dy[0]
    J[mu - 1, 1] = C  # df[0]/dy[1]
    J[mu, 1] = LAM2  # df[1]/dy[1]


# ---------------------------------------------------------------------------
# Parameterized via ctx: float64[6] = [re1, im1, re2, im2, re_c, im_c]
# ---------------------------------------------------------------------------


@cfunc(_zvode_fun_sig)
def _fun_ctx(neq, t, y, dy, ctx):
    p = nb.carray(ctx, (6,), dtype=np.float64)
    lam1 = p[0] + 1j * p[1]
    lam2 = p[2] + 1j * p[3]
    c = p[4] + 1j * p[5]
    dy[0] = lam1 * y[0] + c * y[1]
    dy[1] = lam2 * y[1]


@cfunc(_zvode_jac_sig)
def _jac_dense_ctx(neq, t, y, ml, mu, pd, nrowpd, ctx):
    p = nb.carray(ctx, (6,), dtype=np.float64)
    lam1 = p[0] + 1j * p[1]
    lam2 = p[2] + 1j * p[3]
    c = p[4] + 1j * p[5]
    J = nb.farray(pd, (nrowpd, neq))
    J[0, 0] = lam1
    J[0, 1] = c
    J[1, 1] = lam2


@cfunc(_zvode_jac_sig)
def _jac_banded_ctx(neq, t, y, ml, mu, pd, nrowpd, ctx):
    p = nb.carray(ctx, (6,), dtype=np.float64)
    lam1 = p[0] + 1j * p[1]
    lam2 = p[2] + 1j * p[3]
    c = p[4] + 1j * p[5]
    J = nb.farray(pd, (nrowpd, neq))
    J[mu, 0] = lam1
    J[mu - 1, 1] = c
    J[mu, 1] = lam2


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


def test_export_fun_sig():
    """zvode.zvode_fun_sig must be importable and usable as a @cfunc signature."""
    import zvode

    sig = zvode.zvode_fun_sig

    # Verify it produces a working @cfunc (compilation is the real check)
    @cfunc(sig)
    def _probe(neq, t, y, dy, ctx):
        dy[0] = y[0]

    assert _probe.ctypes is not None


def test_export_jac_sig():
    """zvode.zvode_jac_sig must be importable and usable as a @cfunc signature."""
    import zvode

    sig = zvode.zvode_jac_sig

    @cfunc(sig)
    def _probe(neq, t, y, ml, mu, pd, nrowpd, ctx):
        pd[0] = 0.0 + 0j

    assert _probe.ctypes is not None


# ---------------------------------------------------------------------------
# 2. Compiled closure — no ctx (Pattern 3 from the design spec)
# ---------------------------------------------------------------------------


def test_fun_only_steps():
    """Compiled closure RHS, no Jacobian; collect all steps."""
    sol = solve_complex_ivp(_fun.ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


def test_fun_only_endpoint():
    """Compiled closure RHS, no Jacobian; endpoint-only mode."""
    sol = solve_complex_ivp(
        _fun.ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL, save_steps=False
    )
    np.testing.assert_allclose(sol.y, _exact(TF), rtol=1e-5)


def test_fun_only_knots():
    """Compiled closure RHS, no Jacobian; output at requested knots."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(_fun.ctypes, tspan, Y0, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(sol.t, tspan)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 3. Dense Jacobian, compiled closure
# ---------------------------------------------------------------------------


def test_dense_jac():
    """Compiled closure RHS + compiled dense Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun.ctypes,
        [T0, TF],
        Y0,
        jac=_jac_dense.ctypes,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 4. Banded Jacobian, compiled closure
# ---------------------------------------------------------------------------


def test_banded_jac():
    """Compiled closure RHS + compiled banded Jacobian; no ctx."""
    sol = solve_complex_ivp(
        _fun.ctypes,
        [T0, TF],
        Y0,
        jac=_jac_banded.ctypes,
        lband=LBAND,
        uband=UBAND,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 5. make_rhs factory (design spec Pattern 3, per-call compilation)
# ---------------------------------------------------------------------------


def test_make_rhs_factory():
    """make_rhs creates a fresh @cfunc capturing parameters as constants.

    Each call to make_rhs triggers a fresh numba compilation; the resulting
    ctypes pointer is passed directly to solve_complex_ivp.
    """

    def make_rhs(lam1, lam2, coupling):
        @cfunc(_zvode_fun_sig)
        def rhs(neq, t, y, dy, ctx):
            dy[0] = lam1 * y[0] + coupling * y[1]
            dy[1] = lam2 * y[1]

        return rhs

    my_rhs = make_rhs(LAM1, LAM2, C)
    sol = solve_complex_ivp(my_rhs.ctypes, [T0, TF], Y0, rtol=RTOL, atol=ATOL)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 6. Parameterized via ctx, no Jacobian
# ---------------------------------------------------------------------------


def test_fun_ctx():
    """Compiled RHS parameterized via ctx; no Jacobian."""
    sol = solve_complex_ivp(
        _fun_ctx.ctypes, [T0, TF], Y0, ctx=_CTX, rtol=RTOL, atol=ATOL
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 7. Dense Jacobian with ctx
# ---------------------------------------------------------------------------


def test_dense_jac_ctx():
    """Compiled RHS + compiled dense Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx.ctypes,
        [T0, TF],
        Y0,
        jac=_jac_dense_ctx.ctypes,
        ctx=_CTX,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 8. Banded Jacobian with ctx
# ---------------------------------------------------------------------------


def test_banded_jac_ctx():
    """Compiled RHS + compiled banded Jacobian; both parameterized via ctx."""
    sol = solve_complex_ivp(
        _fun_ctx.ctypes,
        [T0, TF],
        Y0,
        jac=_jac_banded_ctx.ctypes,
        lband=LBAND,
        uband=UBAND,
        ctx=_CTX,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 9. Mixed mode: Python RHS + compiled Jacobian
# ---------------------------------------------------------------------------


def test_mixed_python_rhs_compiled_dense_jac():
    """Python return-value RHS with a compiled dense Jacobian."""
    sol = solve_complex_ivp(
        _python_rhs,
        [T0, TF],
        Y0,
        jac=_jac_dense.ctypes,
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
        jac=_jac_banded.ctypes,
        lband=LBAND,
        uband=UBAND,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 10. ctx keyword with all-Python callbacks warns
# ---------------------------------------------------------------------------


def test_ctx_with_both_python_warns():
    """ctx != None when both fun and jac are Python callables must warn.

    ctx is meaningless for Python callbacks; passing it is likely a mistake,
    so a UserWarning must be issued regardless of whether jac is also present.
    """
    with pytest.warns(UserWarning, match="ctx"):
        solve_complex_ivp(_python_rhs, [T0, TF], Y0, jac=_python_jac, ctx=_CTX)
