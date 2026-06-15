"""Tests for the procedural solve_complex_ivp interface.

Covers the interface contract (output modes, Jacobian types, backward
integration, options, argument validation, and shape semantics) and, in the
"Regression and accuracy tests" section at the end, numerical-accuracy and
solver-option regressions formerly housed in test_extra.py.

The primary problem verified against an analytic solution is a coupled
complex ODE with two components.

System under test: 2-component coupled complex ODE

    dy[0]/dt = LAM1*y[0] + C*y[1]
    dy[1]/dt = LAM2*y[1]

with analytic solution

    y[1](t) = y0[1] * exp(LAM2*t)
    y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t)

where B = C*y0[1]/(LAM2 - LAM1), A = y0[0] - B.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zvode import solve_complex_ivp, ZVODEError

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
    coupled_exact as exact,
    assert_coupled as _check,
    coupled_fun as fun,
    coupled_jac_dense as jac_dense,
    coupled_jac_banded as jac_banded,
    ro128,
    rw128,
)

# The coupled 2-component complex ODE under test, its analytic solution, the
# Python RHS/Jacobian callbacks, and the ctypes view helpers all live in
# _shared.py (see the module docstring above for the problem statement).


# ---------------------------------------------------------------------------
# 1. Output modes × methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_save_steps_true(method):
    """Default mode: collect all accepted steps."""
    sol = solve_complex_ivp(fun, [T0, TF], Y0, method=method, rtol=RTOL, atol=ATOL)
    assert isinstance(sol.t, np.ndarray)
    assert sol.t.ndim == 1 and sol.t[0] == T0 and sol.t[-1] == TF
    assert sol.y.ndim == 2 and sol.y.shape == (2, len(sol.t))
    _check(sol.t, sol.y)


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_save_steps_false(method):
    """Endpoint-only mode: scalar t and 1-D y."""
    sol = solve_complex_ivp(
        fun, [T0, TF], Y0, method=method, rtol=RTOL, atol=ATOL, save_steps=False
    )
    assert np.isscalar(sol.t)
    assert sol.t == pytest.approx(TF)
    assert sol.y.ndim == 1 and sol.y.shape == (2,)
    ref = exact(TF)  # shape (2,)
    assert np.allclose(sol.y, ref, rtol=1e-5)


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_knots_mode(method):
    """Knot mode (len(tspan) > 2): output at exactly the requested times."""
    tspan = np.linspace(T0, TF, 11)
    sol = solve_complex_ivp(fun, tspan, Y0, method=method, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(sol.t, tspan)
    assert sol.y.shape == (2, 11)
    assert sol.y.dtype == np.complex128
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 2. Jacobian types × methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["Adams", "BDF"])
@pytest.mark.parametrize(
    "jac_fn,jac_kwargs",
    [
        pytest.param(None, {}, id="no_jac"),
        pytest.param(jac_dense, {}, id="dense_jac"),
        pytest.param(jac_banded, {"lband": LBAND, "uband": UBAND}, id="banded_jac"),
    ],
)
def test_jacobian_types(method, jac_fn, jac_kwargs):
    """Dense and banded user Jacobians against the no-Jacobian baseline."""
    sol = solve_complex_ivp(
        fun, [T0, TF], Y0, method=method, rtol=RTOL, atol=ATOL, jac=jac_fn, **jac_kwargs
    )
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# 3. Backward integration × output modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_backward_integration(mode):
    """tspan strictly decreasing: integrate from TF back to T0."""
    y_tf = exact(TF)  # shape (2,); initial condition at t=TF
    if mode == "knots":
        tspan = np.linspace(TF, T0, 11)
    else:
        tspan = [TF, T0]
    save = mode == "steps"

    sol = solve_complex_ivp(fun, tspan, y_tf, save_steps=save, rtol=RTOL, atol=ATOL)

    if mode == "endpoint":
        ref = exact(T0)
        assert sol.t == pytest.approx(T0)
        assert np.allclose(sol.y, ref, rtol=1e-5)
    else:
        _check(sol.t, sol.y)


def test_backward_integration_with_first_step():
    """Backward integration with an explicit first_step must not fail.

    first_step is a positive magnitude; solve_complex_ivp must negate it
    (i.e. multiply by sign(t_bound - t0)) before passing it to ZVODE as H0.
    Without this correction ZVODE sees (TOUT - T)*H0 < 0 and returns
    ISTATE = -3 ("Illegal input detected").
    """
    y_tf = exact(TF)
    sol = solve_complex_ivp(
        fun,
        [TF, T0],
        y_tf,
        first_step=0.1,
        rtol=RTOL,
        atol=ATOL,
    )
    _check(sol.t, sol.y)


def test_negative_first_step_raises():
    """A negative first_step must raise ValueError regardless of direction.

    first_step is documented as a positive magnitude; a negative value is
    nonsensical and should be rejected before ZVODE is ever called.
    """
    with pytest.raises(ValueError, match="first_step"):
        solve_complex_ivp(fun, [T0, TF], Y0, first_step=-0.1)


# ---------------------------------------------------------------------------
# 4. allow_overshoot
# ---------------------------------------------------------------------------


def test_allow_overshoot_false():
    """allow_overshoot=False (default): last output point must equal TF exactly."""
    sol = solve_complex_ivp(
        fun, [T0, TF], Y0, allow_overshoot=False, rtol=RTOL, atol=ATOL
    )
    assert sol.t[-1] == pytest.approx(TF)


def test_allow_overshoot_true():
    """allow_overshoot=True: last output point may go slightly past TF."""
    sol = solve_complex_ivp(
        fun, [T0, TF], Y0, allow_overshoot=True, rtol=RTOL, atol=ATOL
    )
    assert sol.t[-1] >= TF - 1e-12
    # Solution at TF should still be accurate regardless of overshoot
    # Find the closest output point to TF and verify the analytic match
    idx = np.argmin(np.abs(sol.t - TF))
    _check(sol.t[idx : idx + 1], sol.y[:, idx : idx + 1])


# ---------------------------------------------------------------------------
# 5. max_num_steps exceeded → RuntimeError
# ---------------------------------------------------------------------------


def test_max_num_steps_exceeded():
    """Solver raises ZVODEError when max_num_steps is too small.

    save_steps=False asks the solver to reach TF in one shot, requiring far
    more than 2 internal steps, so the step limit is hit and the error is
    raised.  With save_steps=True the solver advances one step per call, so
    the per-output-point limit would never be reached with max_num_steps=2.

    ZVODEError subclasses RuntimeError (the historical contract) and carries
    the failure verdict on its `result`; here the endpoint-only path reports
    ISTATE=-1 (excess work), complementing the knots-mode failure exercised
    in test_result_object.py.
    """
    with pytest.raises(ZVODEError, match="ISTATE") as excinfo:
        solve_complex_ivp(fun, [T0, TF], Y0, save_steps=False, max_num_steps=2)

    res = excinfo.value.result
    assert res.success is False
    assert res.status == -1
    assert str(excinfo.value) == res.message


# ---------------------------------------------------------------------------
# 6. Result object and statistics
# ---------------------------------------------------------------------------


def test_result_object():
    """The result exposes the documented fields and the success verdict, keeps
    the solver counters always present (no opt-in), and supports both attribute
    and dict access (one solve covers all of these)."""
    sol = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL)

    # Documented fields present, reachable as attributes and as dict keys.
    for attr in ("t", "y", "success", "status", "message", "nfev", "njev", "nlu"):
        assert hasattr(sol, attr)
    np.testing.assert_array_equal(sol["t"], sol.t)
    np.testing.assert_array_equal(sol["y"], sol.y)
    assert sol["nfev"] == sol.nfev and sol["nlu"] == sol.nlu

    # Success verdict.
    assert sol.success is True
    assert sol.status == 0
    assert isinstance(sol.message, str) and sol.message

    # Solver counters always present.  nni/ncfn/netf names are provisional and
    # may be revised before stabilisation.
    assert sol.nsteps > 0 and sol.nfev > 0
    assert sol.njev >= 0 and sol.nlu >= 0
    assert sol.nni >= 0 and sol.ncfn >= 0 and sol.netf >= 0


# ---------------------------------------------------------------------------
# 7. refine > 1 (denser output via ZVINDY interpolation)
# ---------------------------------------------------------------------------


def test_refine():
    """refine=4 inserts 3 interpolated points per step; solution should match."""
    REFINE = 4
    sol_base = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL, refine=1)
    sol_ref = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL, refine=REFINE)
    # Each of the (n-1) inter-step intervals gains (refine-1) extra points.
    n_steps = len(sol_base.t) - 1
    assert len(sol_ref.t) == len(sol_base.t) + n_steps * (REFINE - 1)
    _check(sol_ref.t, sol_ref.y)


# ---------------------------------------------------------------------------
# 8. Argument validation
# ---------------------------------------------------------------------------


def test_non_monotonic_tspan_raises():
    """Non-monotonic tspan must raise ValueError."""
    with pytest.raises(ValueError, match="monotonic"):
        solve_complex_ivp(fun, [0.0, 1.0, 0.5], Y0)


def test_non_monotonic_mixed_tspan_raises():
    """tspan with mixed sign differences must raise ValueError."""
    with pytest.raises(ValueError, match="monotonic"):
        solve_complex_ivp(fun, [0.0, 2.0, 1.0, 3.0], Y0)


def test_real_y0_warns():
    """Real y0 triggers a UserWarning (not a hard error)."""
    with pytest.warns(UserWarning, match="complex"):
        solve_complex_ivp(fun, [T0, TF], np.array([1.0, 0.0]))


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        solve_complex_ivp(fun, [T0, TF], Y0, method="RK4")


def test_invalid_refine_raises():
    with pytest.raises(ValueError, match="refine"):
        solve_complex_ivp(fun, [T0, TF], Y0, refine=0)


def test_miter1_without_jac_raises():
    """miter=1 without a jac callable raises ValueError."""
    with pytest.raises(ValueError, match="jac"):
        solve_complex_ivp(fun, [T0, TF], Y0, miter=1)


def test_miter4_without_jac_raises():
    """miter=4 without a jac callable raises ValueError."""
    with pytest.raises(ValueError, match="jac"):
        solve_complex_ivp(fun, [T0, TF], Y0, miter=4, lband=LBAND, uband=UBAND)


def test_miter4_without_band_params_raises():
    """miter=4 with jac but without band parameters raises ValueError."""
    with pytest.raises(ValueError, match="lband"):
        solve_complex_ivp(fun, [T0, TF], Y0, jac=jac_dense, miter=4)


def test_miter5_without_band_params_raises():
    """miter=5 without band parameters raises ValueError."""
    with pytest.raises(ValueError, match="lband"):
        solve_complex_ivp(fun, [T0, TF], Y0, miter=5)


def test_miter4_dense_jac_shape_raises():
    """miter=4 with a dense (n×n) jac raises ValueError before Fortran is called.

    jac_dense returns (2, 2) but miter=4 with lband=0, uband=1 expects (2, 2)
    here — but using lband=0, uband=0 the expected shape is (1, 2), making the
    mismatch detectable.
    """
    with pytest.raises(ValueError, match="shape"):
        solve_complex_ivp(fun, [T0, TF], Y0, jac=jac_dense, miter=4, lband=0, uband=0)


def test_miter1_banded_jac_shape_raises():
    """miter=1 with a banded-format jac raises ValueError before Fortran is called.

    jac_banded returns (lband+uband+1, n) = (2, 2) but miter=1 expects (n, n) = (2, 2)
    — for this problem the shapes coincidentally match, so use a clearly wrong shape.
    """

    def jac_wrong(t, y):
        return np.zeros((1, len(y)), dtype=np.complex128)  # (1, 2) for miter=1

    with pytest.raises(ValueError, match="shape"):
        solve_complex_ivp(fun, [T0, TF], Y0, jac=jac_wrong, miter=1)


def test_compiled_callback_works():
    """Compiled ctypes callbacks are now fully supported (no in_place required).

    The new API detects compiled callbacks by type and routes them through the
    C function-pointer path automatically.  A valid ZVODE_FUN_CTYPE callback
    must produce the correct solution.
    """
    from zvode import ZVODE_FUN_CTYPE

    @ZVODE_FUN_CTYPE
    def cfun(neq, t, y_ptr, dy_ptr, ctx):
        y = ro128(y_ptr, neq)
        dy = rw128(dy_ptr, neq)
        dy[0] = LAM1 * y[0] + C * y[1]
        dy[1] = LAM2 * y[1]

    sol = solve_complex_ivp(cfun, [T0, TF], Y0, rtol=1e-8, atol=1e-10)
    _check(sol.t, sol.y)


# ---------------------------------------------------------------------------
# Jacobian shape semantics: scalar-like forms for a single-equation system
#
# Problem: dy/dt = -1j*y,  y(0) = 1,  exact solution y(t) = exp(-1j*t).
# For neq=1, miter=1 requires shape (1, 1).  The table below lists every
# natural way a user might write "the scalar -1j" and what np.asarray()
# makes of it:
#
#   Form               np.asarray(...)   shape    result
#   -----------------  ----------------  -------  --------
#   -1j                complex scalar    ()       ValueError
#   [-1j]              1-D list          (1,)     ValueError
#   [[-1j]]            nested list       (1, 1)   accepted ← correct form
#   np.array(-1j)      0-D ndarray       ()       ValueError
#   np.array([-1j])    1-D ndarray       (1,)     ValueError
# ---------------------------------------------------------------------------


def _S_FUN(t, y):
    return -1j * y


_S_Y0 = np.array([1.0 + 0j], dtype=np.complex128)
_S_TSPAN = [0.0, 1.0]
_S_EXACT_FINAL = np.exp(-1j * 1.0)


@pytest.mark.parametrize(
    "jac,label",
    [
        (lambda t, y: -1j, "scalar complex"),
        (lambda t, y: [-1j], "1-D list"),
        (lambda t, y: np.array(-1j), "0-D ndarray"),
        (lambda t, y: np.array([-1j]), "1-D ndarray"),
    ],
)
def test_scalar_jac_shape_raises(jac, label):
    """Jacobians that don't return a (1,1) array must raise ValueError naming 'shape'."""
    with pytest.raises(ValueError, match="shape"):
        solve_complex_ivp(_S_FUN, _S_TSPAN, _S_Y0, jac=jac)


def test_nested_list_jac_accepted():
    """[[item]] produces shape (1,1) after np.asarray() and is the correct scalar form."""
    sol = solve_complex_ivp(
        _S_FUN,
        _S_TSPAN,
        _S_Y0,
        jac=lambda t, y: [[-1j]],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(sol.y[0, -1], _S_EXACT_FINAL, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# RHS shape semantics: scalar-like forms for a single-equation system
#
# Problem: dy/dt = -1j*y,  y(0) = 1,  exact solution y(t) = exp(-1j*t).
# For neq=1, fun must return shape (1,).  The table below contrasts with the
# Jacobian convention (which requires (1, 1)):
#
#   Form                    np.asarray(...)   shape    result
#   ----------------------  ----------------  -------  --------
#   -1j*y[0]  scalar        complex scalar    ()       ValueError
#   np.array(-1j*y[0]) 0-D  0-D ndarray       ()       ValueError
#   [[-1j*y[0]]] 2-D list   nested list       (1, 1)   ValueError
#   [-1j*y[0]]  1-D list    1-D list          (1,)     accepted  ← correct form
# ---------------------------------------------------------------------------

_F_Y0 = np.array([1.0 + 0j], dtype=np.complex128)
_F_TSPAN = [0.0, 1.0]
_F_EXACT_FINAL = np.exp(-1j * 1.0)


@pytest.mark.parametrize(
    "fun,label",
    [
        (lambda t, y: -1j * y[0], "scalar"),
        (lambda t, y: np.array(-1j * y[0]), "0-D ndarray"),
        (lambda t, y: [[-1j * y[0]]], "2-D list"),
    ],
)
def test_wrong_fun_shape_raises(fun, label):
    """RHS functions that don't return a (1,) array must raise ValueError naming 'fun'."""
    with pytest.raises(ValueError, match="fun"):
        solve_complex_ivp(fun, _F_TSPAN, _F_Y0)


def test_1d_list_fun_accepted():
    """[-1j*y[0]] produces shape (1,) after np.asarray() and is the correct scalar form."""
    sol = solve_complex_ivp(
        lambda t, y: [-1j * y[0]],
        _F_TSPAN,
        _F_Y0,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(sol.y[0, -1], _F_EXACT_FINAL, rtol=1e-5, atol=1e-8)


# ===========================================================================
# Regression and accuracy tests (merged from the former test_extra.py)
#
#   A. Damped harmonic oscillator accuracy (Adams, BDF)
#   B. Nonlinear complex oscillator accuracy
#   C. Error paths: negative atol, short tspan
#   D. Single-element (n=1) system: decay, damped oscillation, pure rotation
#   E. refine > 1 interpolation accuracy (refine=2, refine=5)
#   F. max_order constrains the Adams solver order (n=2, complex eigenvalues)
#   G. Adaptive step-buffer (StepBuf) growth and output structure
#
# Test A uses a real-valued ODE (real coefficients, real IC) run through the
# complex solver.  This is NOT the intended use of solve_complex_ivp; it is
# included solely as a numerical accuracy regression against a known exact
# solution.  (The SciPy-BDF cross-check that lived here is dropped: it is a
# strict subset of test_scipy_cross_validation.py.)
# ===========================================================================


# ---------------------------------------------------------------------------
# A. Numerical accuracy: underdamped harmonic oscillator (real ODE in complex)
# ---------------------------------------------------------------------------

OMEGA = 2.0
GAMMA = 0.5
OMEGA_D = np.sqrt(OMEGA**2 - GAMMA**2)  # ≈ 1.936


def osc_fun(t, y):
    return np.array([y[1], -(OMEGA**2) * y[0] - 2 * GAMMA * y[1]], dtype=complex)


def osc_exact(t):
    """Exact solution starting from y0=[1, 0]: [x(t), v(t)]."""
    et = np.exp(-GAMMA * t)
    x = et * (np.cos(OMEGA_D * t) + (GAMMA / OMEGA_D) * np.sin(OMEGA_D * t))
    v = -et * (OMEGA**2 / OMEGA_D) * np.sin(OMEGA_D * t)
    return np.array([x + 0j, v + 0j])


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_damped_oscillator_accuracy(method):
    """Both Adams and BDF track the underdamped oscillator to within 1e-5 relative error."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    sol = solve_complex_ivp(
        osc_fun, [0.0, 10.0], y0, method=method, rtol=1e-10, atol=1e-12
    )

    ref = osc_exact(sol.t)
    # rtol=1e-5 accommodates global error accumulation over t=[0,10];
    # atol=1e-9 handles near-zero values at the end of the damped range.
    assert_allclose(sol.y, ref, rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# B. Numerical accuracy: nonlinear complex oscillator (docs/example.py)
#
#    dw/dt = -i w² z          z(0) = 1        z(t) = exp(it)
#    dz/dt =  i z             w(0) = 1/2.1    w(t) = 1/(exp(it) + 1.1)
# ---------------------------------------------------------------------------


def nl_osc_fun(t, y):
    w, z = y[0], y[1]
    return np.array([-1j * w**2 * z, 1j * z], dtype=np.complex128)


def nl_osc_exact(t):
    z = np.exp(1j * t)
    w = 1.0 / (z + 1.1)
    return np.array([w, z])


def test_nonlinear_oscillator_accuracy():
    """solve_complex_ivp tracks the nonlinear complex oscillator to within 1e-6."""
    y0 = np.array([1.0 / 2.1 + 0j, 1.0 + 0j])
    sol = solve_complex_ivp(nl_osc_fun, [0.0, 4 * np.pi], y0, rtol=1e-10, atol=1e-12)

    ref = nl_osc_exact(sol.t)
    assert_allclose(sol.y, ref, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# C. Error paths
# ---------------------------------------------------------------------------

FUN_1D = lambda t, y: -y  # noqa: E731
Y0_1D = np.array([1.0 + 0j])


@pytest.mark.parametrize(
    "tspan,kwargs,match",
    [
        ([0.0, 1.0], {"atol": -1e-10}, "positive"),  # negative atol
        ([0.0], {}, "two elements"),  # tspan too short
    ],
)
def test_error_paths(tspan, kwargs, match):
    """Invalid arguments raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        solve_complex_ivp(FUN_1D, tspan, Y0_1D, **kwargs)


# ---------------------------------------------------------------------------
# D. Edge case: single-element (n=1) system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lam",
    [
        pytest.param(-1.0 + 0j, id="decay"),
        pytest.param(-1.0 + 2j, id="damped_osc"),
        pytest.param(1j, id="rotation"),
    ],
)
def test_single_element_system(lam):
    """n=1 scalar complex ODE y'=lam*y integrates correctly for three qualitatively different lam."""
    y0 = np.array([1.0 + 0j])
    sol = solve_complex_ivp(
        lambda t, y: np.array([lam * y[0]]),
        [0.0, 2.0],
        y0,
        rtol=1e-10,
        atol=1e-12,
    )
    assert_allclose(sol.y[0], y0[0] * np.exp(lam * sol.t), rtol=1e-7)


# ---------------------------------------------------------------------------
# E. Edge case: refine > 1 interpolation accuracy
# ---------------------------------------------------------------------------

OMEGA_R = np.pi  # one full Rabi oscillation over t ∈ [0, 2]


def rabi_fun(t, y):
    h = OMEGA_R / 2
    return np.array([-1j * h * y[1], -1j * h * y[0]], dtype=complex)


def rabi_exact(t):
    return np.array([np.cos(OMEGA_R * t / 2) + 0j, -1j * np.sin(OMEGA_R * t / 2)])


@pytest.mark.parametrize("refine", [2, 5])
def test_refine_interpolation_accuracy(refine):
    """ZVINDY-interpolated points match the Rabi exact solution for refine=2 and refine=5."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    sol = solve_complex_ivp(
        rabi_fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12, refine=refine
    )

    ref = rabi_exact(sol.t)
    # atol=1e-9 guards the zero-crossing where the exact value is ~1e-16.
    assert_allclose(sol.y, ref, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# F. Edge case: max_order constraint
# ---------------------------------------------------------------------------

# Two-component decoupled system with complex eigenvalues: y' = diag(lam) * y
# Complex lam → solution oscillates and decays; genuinely complex-valued.
# Exact endpoint: y[i](T) = y0[i] * exp(lam[i] * T)
MAX_ORDER_LAM = np.array([-1.0 + 2j, -2.0 + 1j])
MAX_ORDER_T = 5.0


def max_order_fun(t, y):
    return MAX_ORDER_LAM * y


def test_max_order_constraint():
    """max_order=1 forces first-order Adams steps: more steps, same correct endpoint."""
    y0 = np.array([1.0 + 0j, 1.0 + 0j])
    kw = dict(method="Adams", rtol=1e-8, atol=1e-10, save_steps=False)

    sol_default = solve_complex_ivp(max_order_fun, [0.0, MAX_ORDER_T], y0, **kw)
    sol_order1 = solve_complex_ivp(
        max_order_fun, [0.0, MAX_ORDER_T], y0, max_order=1, **kw
    )

    exact_end = y0 * np.exp(MAX_ORDER_LAM * MAX_ORDER_T)
    assert_allclose(sol_default.y, exact_end, rtol=1e-6)
    # Adams order-1 global error is O(sqrt(rtol)) ≈ 4e-4 for rtol=1e-8.
    assert_allclose(sol_order1.y, exact_end, rtol=1e-3)
    assert sol_order1.nsteps > sol_default.nsteps, (
        f"max_order=1 should need more steps than default "
        f"(got {sol_order1.nsteps} vs {sol_default.nsteps})"
    )


# ---------------------------------------------------------------------------
# G. Adaptive step-buffer (StepBuf) growth and output structure
#
# These tests target the malloc/realloc-backed StepBuf used by
# drive_adaptive.  The adaptive path is taken when tspan has exactly two
# elements and save_steps=True (the default).  A long, accurate integration
# records well over a thousand accepted steps, forcing the buffer to grow
# (realloc-double) many times past its initial capacity of 10 — so these
# tests exercise stepbuf_init/append/grow/finalize end to end.
# ---------------------------------------------------------------------------

# Underdamped oscillator written as a genuinely-complex first order system;
# tight tolerances guarantee many (~1000) accepted steps.
BUF_LAM = -0.5 + 4j


def buf_fun(t, y):
    return np.array([BUF_LAM * y[0]], dtype=complex)


def buf_exact(t):
    return np.array([np.exp(BUF_LAM * t)])


def test_adaptive_buffer_no_refine():
    """Adaptive path (refine=1) records the IC plus every accepted step.

    Asserts the buffer grew well past its initial capacity (so realloc ran
    repeatedly), that the time column is strictly increasing and spans the
    full interval, and that the value column is F-contiguous complex128 with
    the interpolation-free endpoints matching the exact solution.
    """
    y0 = np.array([1.0 + 0j])
    sol = solve_complex_ivp(
        buf_fun, [0.0, 12.0], y0, rtol=1e-11, atol=1e-13, refine=1
    )

    assert sol.success, sol.message

    # Output structure produced by stepbuf_copy_out.
    assert sol.t.dtype == np.float64
    assert sol.y.dtype == np.complex128
    assert sol.y.flags["F_CONTIGUOUS"]
    assert sol.y.shape == (1, sol.t.size)

    # Far more points than STEPBUF_INIT_CAP (=10): the buffer reallocated
    # several times.  This is the whole point of the growth path.
    assert sol.t.size > 100

    # IC at index 0, strictly increasing time, exact endpoints.
    assert sol.t[0] == 0.0
    assert sol.t[-1] == pytest.approx(12.0)
    assert np.all(np.diff(sol.t) > 0.0)
    assert sol.y[0, 0] == y0[0]
    assert_allclose(sol.y[0], buf_exact(sol.t)[0], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("refine", [1, 3, 4])
def test_adaptive_buffer_point_count_scales_with_refine(refine):
    """With N accepted steps, the buffer holds exactly 1 + N*refine points.

    Each accepted step appends (refine - 1) interpolated points followed by
    the step endpoint; the leading IC is appended once.  Holding the problem
    and tolerances fixed keeps N constant across refine, so the total point
    count scales linearly — a direct check that stepbuf_append is called the
    expected number of times on both the refine and non-refine branches.
    """
    y0 = np.array([1.0 + 0j])
    kw = dict(rtol=1e-11, atol=1e-13)

    sol1 = solve_complex_ivp(buf_fun, [0.0, 12.0], y0, refine=1, **kw)
    sol = solve_complex_ivp(buf_fun, [0.0, 12.0], y0, refine=refine, **kw)

    n_steps = sol1.t.size - 1  # points excluding the initial condition
    assert sol.t.size == 1 + n_steps * refine

    # Structure and accuracy hold on the refined output too.
    assert sol.y.flags["F_CONTIGUOUS"]
    assert sol.t[0] == 0.0
    assert sol.t[-1] == pytest.approx(12.0)
    assert np.all(np.diff(sol.t) > 0.0)
    assert_allclose(sol.y[0], buf_exact(sol.t)[0], rtol=1e-6, atol=1e-9)


def test_adaptive_buffer_refine_inserts_interior_points():
    """Refinement inserts the requested interior points between step endpoints.

    Compares the refine=1 step grid against refine=3: every refine=1 time must
    still appear, and exactly (refine - 1) extra points must fall strictly
    inside each step — confirming the interpolated samples are appended to the
    buffer in order, not appended at the boundaries.
    """
    y0 = np.array([1.0 + 0j])
    kw = dict(rtol=1e-10, atol=1e-12)

    endpoints = solve_complex_ivp(buf_fun, [0.0, 12.0], y0, refine=1, **kw).t
    refined = solve_complex_ivp(buf_fun, [0.0, 12.0], y0, refine=3, **kw).t

    # The endpoint grid is a subsequence of the refined grid (steps unchanged).
    assert np.all(np.isin(endpoints, refined))

    # Between consecutive step endpoints there are exactly two interior points.
    for a, b in zip(endpoints[:-1], endpoints[1:]):
        interior = refined[(refined > a) & (refined < b)]
        assert interior.size == 2
        assert np.all(np.diff(interior) > 0.0)
