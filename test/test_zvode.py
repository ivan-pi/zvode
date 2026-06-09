"""Tests for the _zvode extension module.

These smoke-tests exercise _zvode.zvode(...) directly, bypassing any
higher-level wrapper.  The goal is to verify that the C extension and the
underlying Fortran solver are wired together correctly.

Conventions (from the ZVODE documentation)
-------------------------------------------
istate  1  – first call (input)
        2  – successful return (output)
        negative – error (output)

itask   1  – integrate to TOUT, interpolating at TOUT

itol    1  – scalar rtol *and* scalar atol (both length-1 arrays)

mf = 10   Adams / non-stiff, functional iteration (no Jacobian needed)
mf = 11   Adams / non-stiff, user-supplied dense Jacobian
mf = 21   BDF   / stiff,     user-supplied dense Jacobian
mf = 22   BDF   / stiff,     internally-generated dense Jacobian

Work-array sizes (from the ZVODE docs, using default MAXORD)
-------------------------------------------------------------
MF = 10:  LZW = 15*NEQ,             LRW = 20+NEQ,  LIW = 30
MF = 11:  LZW = 15*NEQ + 2*NEQ**2,  LRW = 20+NEQ,  LIW = 30+NEQ
MF = 21:  LZW = 8*NEQ + 2*NEQ**2,   LRW = 20+NEQ,  LIW = 30+NEQ
MF = 22:  LZW = 8*NEQ + 2*NEQ**2,   LRW = 20+NEQ,  LIW = 30+NEQ

Optional output stored by ZVODE on a successful call (0-based Python indices)
  rwork[10]  = HU   – step size last used
  rwork[12]  = TCUR – current internal time reached by the solver
  iwork[10]  = NST  – number of steps taken
  iwork[11]  = NFE  – number of f evaluations
  iwork[12]  = NJE  – number of Jacobian evaluations
"""

import numpy as np
from numpy.testing import assert_allclose

import pytest

from zvode import _zvode
from zvode.zvode_impl import ZVODEDenseOutput


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


def _call_zvode(
    fun,
    y,
    t,
    tout,
    zwork,
    rwork,
    iwork,
    *,
    mf=10,
    itol=1,
    rtol=1e-7,
    atol=1e-9,
    itask=1,
    istate=1,
    iopt=0,
    jac=None,
):
    """Thin wrapper that converts scalar tolerances to 1-element arrays."""
    rtol_arr = np.array([rtol], dtype=np.float64)
    atol_arr = np.array([atol], dtype=np.float64)
    return _zvode.zvode(
        fun,
        y,
        t,
        tout,
        itol,
        rtol_arr,
        atol_arr,
        itask,
        istate,
        iopt,
        zwork,
        rwork,
        iwork,
        jac,
        mf,
    )


def _array_range(arr):
    base = arr.ctypes.data
    return f"[{hex(base)}, {hex(base + arr.nbytes)})"


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
        dy[0] = -y[0]

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 1.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)
    iopt = 0

    print(f"lzw = {zwork.shape}")
    print(f"lrw = {rwork.shape}")
    print(f"liw = {iwork.shape}")

    print(f"y     : {_array_range(y)}")
    print(f"zwork : {_array_range(zwork)}")
    print(f"rwork : {_array_range(rwork)}")
    print(f"iwork : {_array_range(iwork)}")

    itol = 1
    rtol_arr = np.array([1e-6], dtype=np.float64)
    atol_arr = np.array([1e-8], dtype=np.float64)

    itask = 1
    istate = 1

    t_new, istate_new = _zvode.zvode(
        fun,
        y,
        t,
        tout,
        itol,
        rtol_arr,
        atol_arr,
        itask,
        istate,
        iopt,
        zwork,
        rwork,
        iwork,
        None,
        mf,
    )

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"
    assert t_new == tout, "ZVODE did not reach TOUT"

    assert_allclose(y[0].real, np.exp(-tout), rtol=1e-6, atol=1e-8)
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

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = np.pi / 2.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork, mf=mf)

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    expected = np.exp(1j * tout)  # = 0 + 1j (exactly at pi/2)
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

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    rtol = np.array([1e-7], dtype=np.float64)
    atol = np.array([1e-9], dtype=np.float64)
    istate = 1
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    checkpoints = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]

    for tout in checkpoints:
        t, istate = _zvode.zvode(
            fun, y, t, tout, 1, rtol, atol, 1, istate, 0, zwork, rwork, iwork, None, mf
        )

        assert istate == 2, f"ZVODE failed at tout={tout:.4f} with istate={istate}"

        expected = np.exp(1j * tout)
        assert_allclose(
            y[0],
            expected,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Solution mismatch at tout = {tout:.4f}",
        )


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

    y = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 5.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(
        fun, y, t, tout, zwork, rwork, iwork, mf=mf, rtol=1e-6, atol=1e-8
    )

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    expected = np.array([np.exp(-0.1 * tout), np.exp(-2.0 * tout)]) + 0j
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

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 5.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(
        fun, y, t, tout, zwork, rwork, iwork, mf=mf, rtol=1e-6, atol=1e-8
    )

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

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 2.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(fun, y, t, tout, zwork, rwork, iwork, mf=mf)

    assert istate_new == 2

    hu = rwork[10]  # last step size used
    tcur = rwork[12]  # current internal time
    nst = iwork[10]  # number of steps
    nfe = iwork[11]  # number of f evaluations

    assert hu > 0, f"HU should be positive, got {hu}"
    assert tcur >= tout, f"TCUR ({tcur}) should be >= TOUT ({tout})"
    assert nst > 0, f"NST should be positive, got {nst}"
    assert nfe > 0, f"NFE should be positive, got {nfe}"


def test_zvode_wrong_array_type():
    """
    Passing a plain Python list where a numpy ndarray is required should
    raise a TypeError, not crash the interpreter.
    """
    neq = 1
    mf = 10

    def fun(t, y, dy):
        dy[0] = -y[0]

    rtol = np.array([1e-6], dtype=np.float64)
    atol = np.array([1e-8], dtype=np.float64)
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    # y is a list instead of a numpy array
    with pytest.raises(TypeError):
        _zvode.zvode(
            fun,
            [1.0 + 0j],
            0.0,
            1.0,
            1,
            rtol,
            atol,
            1,
            1,
            0,
            zwork,
            rwork,
            iwork,
            None,
            mf,
        )

    # zwork is a list instead of a numpy array
    y = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(TypeError):
        _zvode.zvode(
            fun,
            y,
            0.0,
            1.0,
            1,
            rtol,
            atol,
            1,
            1,
            0,
            list(zwork),
            rwork,
            iwork,
            None,
            mf,
        )


def test_zvode_bdf_user_jacobian():
    """
    Test the BDF stiff solver with a user-supplied dense complex Jacobian (MF=21).

    System:
        dy[0]/dt = -y[0] + 1j * y[1]
        dy[1]/dt = -1j * y[0] - 2.0 * y[1]

    Jacobian:
        J[0, 0] = -1.0;  J[0, 1] = 1j
        J[1, 0] = -1j;   J[1, 1] = -2.0
    """
    neq = 2
    mf = 21

    def fun(t, y, dy):
        dy[0] = -y[0] + 1j * y[1]
        dy[1] = -1j * y[0] - 2.0 * y[1]

    def jac(t, y, J):
        J[0, 0] = -1.0 + 0j
        J[0, 1] = 1j
        J[1, 0] = -1j
        J[1, 1] = -2.0 + 0j

    y = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 1.0
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(
        fun, y, t, tout, zwork, rwork, iwork, mf=mf, jac=jac, rtol=1e-8, atol=1e-10
    )

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    # Check that the Jacobian was evaluated (NJE is stored in iwork[12] in ZVODE)
    nje = iwork[12]
    assert nje > 0, f"User Jacobian was not evaluated (NJE = {nje})"


def test_zvode_adams_user_jacobian():
    """
    Test the Adams non-stiff solver with a user-supplied dense complex Jacobian (MF=11).

    System (Non-linear):
        dy/dt = 1j * y**2

    Jacobian:
        J[0, 0] = 2j * y[0]

    Analytic solution: y(t) = y(0) / (1 - 1j * y(0) * t)
    """
    neq = 1
    mf = 11

    def fun(t, y, dy):
        dy[0] = 1j * y[0] ** 2

    def jac(t, y, J):
        J[0, 0] = 2j * y[0]

    y = np.array([1.0 + 0j], dtype=np.complex128)
    t = 0.0
    tout = 0.5  # Kept small to avoid approaching the singularity
    zwork, rwork, iwork = _make_workspaces(neq, mf)

    t_new, istate_new = _call_zvode(
        fun, y, t, tout, zwork, rwork, iwork, mf=mf, jac=jac, rtol=1e-8, atol=1e-10
    )

    assert istate_new == 2, f"ZVODE failed with istate = {istate_new}"

    expected = 1.0 / (1.0 - 1j * 1.0 * tout)
    assert_allclose(y[0], expected, rtol=1e-6)

    nje = iwork[12]
    assert nje > 0, f"User Jacobian was not evaluated (NJE = {nje})"


# ---------------------------------------------------------------------------
# zvindy tests
# ---------------------------------------------------------------------------


def _nordsieck_from_poly(polys, tn, h, nq):
    """
    Build a Nordsieck history array for a list of polynomials evaluated at tn.

    Each column j of the returned array contains h^j/j! * p^(j)(tn), which is
    exactly what ZVINDY expects.  The array is F-contiguous as required.

    Parameters
    ----------
    polys : list of np.poly1d  -- one per ODE component
    tn    : float              -- current solver time (right end of interval)
    h     : float              -- step size (also used as hu)
    nq    : int                -- order (number of Nordsieck columns is nq+1)

    Returns
    -------
    yh : complex128 ndarray, shape (n, nq+1), F-contiguous
    """
    n = len(polys)
    yh = np.zeros((n, nq + 1), dtype=np.complex128, order="F")
    factorial = 1
    for j in range(nq + 1):
        if j > 0:
            factorial *= j
        for i, p in enumerate(polys):
            deriv = p.deriv(j)
            yh[i, j] = (h**j / factorial) * deriv(tn)
    return yh


def test_zvindy_cubic_interpolation():
    """
    ZVINDY must reproduce a cubic polynomial exactly when the Nordsieck
    array is built from that polynomial's Taylor coefficients (nq=3).

    We test K=0 (value), K=1 (first derivative), K=2 (second derivative),
    and K=3 (third derivative) at an interior point.
    """
    # Two independent cubic polynomials (real coefficients, complex arrays)
    p0 = np.poly1d([1.0, -2.0, 3.0, -4.0])  # t^3 - 2t^2 + 3t - 4
    p1 = np.poly1d([-3.0, 0.0, 1.0, 2.0])  # -3t^3 + t + 2

    polys = [p0, p1]
    n = len(polys)
    nq = 3

    tn = 4.0  # right end of the interpolation interval
    h = 2.0  # step size; hu = h so valid range is [tn-h, tn] = [2, 4]
    hu = h
    t = 3.0  # interior interpolation point

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    dky = np.zeros(n, dtype=np.complex128)

    # K=0: interpolated value should match the polynomial
    _zvode.zvindy(t, 0, yh, h, tn, hu, dky)
    expected = np.array([p(t) for p in polys], dtype=np.complex128)
    assert_allclose(dky, expected, rtol=1e-13, err_msg="K=0 value mismatch for cubic")

    # K=1: first derivative
    _zvode.zvindy(t, 1, yh, h, tn, hu, dky)
    expected = np.array([p.deriv(1)(t) for p in polys], dtype=np.complex128)
    assert_allclose(
        dky, expected, rtol=1e-12, err_msg="K=1 derivative mismatch for cubic"
    )

    # K=2: second derivative
    _zvode.zvindy(t, 2, yh, h, tn, hu, dky)
    expected = np.array([p.deriv(2)(t) for p in polys], dtype=np.complex128)
    assert_allclose(
        dky, expected, rtol=1e-12, err_msg="K=2 derivative mismatch for cubic"
    )

    # K=3: third derivative (constant for a cubic)
    _zvode.zvindy(t, 3, yh, h, tn, hu, dky)
    expected = np.array([p.deriv(3)(t) for p in polys], dtype=np.complex128)
    assert_allclose(
        dky, expected, rtol=1e-11, err_msg="K=3 derivative mismatch for cubic"
    )


def test_zvindy_quintic_interpolation():
    """
    ZVINDY must reproduce a quintic polynomial exactly when nq=5.

    Uses complex polynomial coefficients to exercise the complex arithmetic
    path.  Tests K=0 through K=5.
    """
    # Complex quintic polynomials: coefficients [a5, a4, ..., a0]
    p0 = np.poly1d([(1 + 2j), -3j, (2 - 1j), 0.5, -1.0, (3 + 0j)])
    p1 = np.poly1d([(-2 + 1j), 1.0, 0j, (1 - 3j), 2j, (-1 + 2j)])

    polys = [p0, p1]
    n = len(polys)
    nq = 5

    tn = 1.0
    h = 0.5  # valid interpolation range: [0.5, 1.0]
    hu = h
    t = 0.75

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    dky = np.zeros(n, dtype=np.complex128)

    for k in range(nq + 1):
        _zvode.zvindy(t, k, yh, h, tn, hu, dky)
        expected = np.array([p.deriv(k)(t) for p in polys], dtype=np.complex128)
        assert_allclose(
            dky, expected, rtol=1e-10, err_msg=f"K={k} mismatch for complex quintic"
        )


def test_zvindy_at_endpoints():
    """
    At t=tn and t=tn-hu the interpolation should still be exact.
    """
    p0 = np.poly1d([2.0, -1.0, 0.5, 1.0])
    polys = [p0]
    n = 1
    nq = 3

    tn = 3.0
    h = 1.5
    hu = h

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    dky = np.zeros(n, dtype=np.complex128)

    for t_eval in [tn, tn - hu]:
        _zvode.zvindy(t_eval, 0, yh, h, tn, hu, dky)
        expected = np.array([p(t_eval) for p in polys], dtype=np.complex128)
        assert_allclose(
            dky, expected, rtol=1e-13, err_msg=f"K=0 mismatch at t={t_eval}"
        )


def test_zvindy_out_of_range_raises():
    """
    Requests outside [tn-hu, tn] or with invalid k should raise ValueError.
    """
    p0 = np.poly1d([1.0, 0.0, 0.0, 0.0])
    nq = 3
    tn = 2.0
    h = 1.0
    hu = h

    yh = _nordsieck_from_poly([p0], tn, h, nq)
    dky = np.zeros(1, dtype=np.complex128)

    with pytest.raises(ValueError):
        _zvode.zvindy(tn + 0.1, 0, yh, h, tn, hu, dky)  # t > tn

    with pytest.raises(ValueError):
        _zvode.zvindy(tn - hu - 0.1, 0, yh, h, tn, hu, dky)  # t < tn-hu

    with pytest.raises(ValueError):
        _zvode.zvindy(tn - 0.5, nq + 1, yh, h, tn, hu, dky)  # k > nq


# ---------------------------------------------------------------------------
# ZVODEDenseOutput tests
# ---------------------------------------------------------------------------


def test_dense_output_cubic_scalar():
    """
    ZVODEDenseOutput must reproduce a cubic polynomial exactly at a scalar t.
    """
    p0 = np.poly1d([1.0, -2.0, 3.0, -4.0])
    p1 = np.poly1d([-3.0, 0.0, 1.0, 2.0])
    polys = [p0, p1]

    tn, h = 4.0, 2.0
    t_old = tn - h
    nq = 3

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    interp = ZVODEDenseOutput(t_old, tn, yh, h)

    t_eval = 3.0
    result = interp(t_eval)
    expected = np.array([p(t_eval) for p in polys], dtype=np.complex128)
    assert result.shape == (len(polys),)
    assert_allclose(
        result, expected, rtol=1e-13, err_msg="Cubic scalar evaluation mismatch"
    )


def test_dense_output_cubic_array():
    """
    ZVODEDenseOutput must return shape (n, m) and be exact for each point
    when called with an array of m interpolation times.
    """
    p0 = np.poly1d([1.0, -2.0, 3.0, -4.0])
    p1 = np.poly1d([-3.0, 0.0, 1.0, 2.0])
    polys = [p0, p1]

    tn, h = 4.0, 2.0
    t_old = tn - h
    nq = 3

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    interp = ZVODEDenseOutput(t_old, tn, yh, h)

    t_eval = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    result = interp(t_eval)
    assert result.shape == (len(polys), len(t_eval))

    for k, t in enumerate(t_eval):
        expected = np.array([p(t) for p in polys], dtype=np.complex128)
        assert_allclose(
            result[:, k],
            expected,
            rtol=1e-13,
            err_msg=f"Cubic array evaluation mismatch at t={t}",
        )


def test_dense_output_quintic_complex():
    """
    ZVODEDenseOutput is exact for a complex quintic polynomial (nq=5).
    """
    p0 = np.poly1d([(1 + 2j), -3j, (2 - 1j), 0.5, -1.0, (3 + 0j)])
    p1 = np.poly1d([(-2 + 1j), 1.0, 0j, (1 - 3j), 2j, (-1 + 2j)])
    polys = [p0, p1]

    tn, h = 1.0, 0.5
    t_old = tn - h
    nq = 5

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    interp = ZVODEDenseOutput(t_old, tn, yh, h)

    t_eval = np.linspace(t_old, tn, 9)
    result = interp(t_eval)

    for k, t in enumerate(t_eval):
        expected = np.array([p(t) for p in polys], dtype=np.complex128)
        assert_allclose(
            result[:, k],
            expected,
            rtol=1e-10,
            err_msg=f"Quintic complex evaluation mismatch at t={t}",
        )


def test_dense_output_endpoints():
    """
    At t=t_old and t=tn the dense output must recover the exact polynomial value.
    """
    p0 = np.poly1d([2.0, -1.0, 0.5, 1.0])
    polys = [p0]

    tn, h = 3.0, 1.5
    t_old = tn - h
    nq = 3

    yh = _nordsieck_from_poly(polys, tn, h, nq)
    interp = ZVODEDenseOutput(t_old, tn, yh, h)

    for t_eval in [t_old, tn]:
        result = interp(t_eval)
        expected = np.array([p(t_eval) for p in polys], dtype=np.complex128)
        assert_allclose(
            result, expected, rtol=1e-13, err_msg=f"Endpoint mismatch at t={t_eval}"
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_zvode_scalar_real_decay()
    test_zvode_complex_rotation()
    test_zvode_multistep_continuation()
    test_zvode_two_component_system()
    test_zvode_bdf_method()
    test_zvode_optional_output_populated()
    test_zvode_wrong_array_type()
    test_zvode_bdf_user_jacobian()
    test_zvode_adams_user_jacobian()
    test_zvindy_cubic_interpolation()
    test_zvindy_quintic_interpolation()
    test_zvindy_at_endpoints()
    test_zvindy_out_of_range_raises()
    test_dense_output_cubic_scalar()
    test_dense_output_cubic_array()
    test_dense_output_quintic_complex()
    test_dense_output_endpoints()
    print("All tests passed.")
