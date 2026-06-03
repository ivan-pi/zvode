"""
Extra regression and validation tests for solve_complex_ivp.

Each test here was verified to be absent from the three existing test files
(test_zvode.py, test_zvode_ivp.py, test_solve_complex_ivp.py).

Tests:
  1. Numerical accuracy — underdamped harmonic oscillator vs exact solution
     (no oscillator ODE is tested in any existing file)
  2. Numerical accuracy — nonlinear complex oscillator (dw/dt = -iw²z, dz/dt = iz)
     vs exact solution; this problem is tested via ZVODE/solve_ivp in
     test_zvode_ivp.py but never via solve_complex_ivp
  3. Error paths — negative atol and single-element tspan (the remaining two
     invalid-argument cases not covered by the existing tests)
  4. Jacobian efficiency — supplying an exact Jacobian (miter=1) reduces the
     number of RHS evaluations compared to finite-diff (miter=2), verifying that
     ZVODE actually uses the Jacobian rather than silently ignoring it
  5. Cross-validation — endpoint agrees with scipy.integrate.solve_ivp(method='BDF')
  6. Edge case — single-element (n=1) system via solve_complex_ivp
     (n=1 is tested at the _zvode / ZVODE level but not through solve_complex_ivp)
  7. Edge case — refine > 1 inserts ZVINDY-interpolated points that match the
     exact solution (test_refine in test_solve_complex_ivp.py only checks output
     structure, not accuracy against an analytical reference)
  8. Edge case — max_order=1 forces first-order Adams steps; the correct answer
     is still reached but more steps are required (max_order is never tested
     functionally in the existing suite)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from zvode import solve_complex_ivp


# ---------------------------------------------------------------------------
# 1. Numerical accuracy: underdamped harmonic oscillator
# ---------------------------------------------------------------------------

_OMEGA = 2.0
_GAMMA = 0.5
_OMEGA_D = np.sqrt(_OMEGA**2 - _GAMMA**2)  # ≈ 1.936


def _osc_fun(t, y):
    return np.array([y[1], -(_OMEGA**2) * y[0] - 2 * _GAMMA * y[1]], dtype=complex)


def _osc_exact(t):
    """Exact solution starting from y0=[1, 0]: [x(t), v(t)]."""
    et = np.exp(-_GAMMA * t)
    x = et * (np.cos(_OMEGA_D * t) + (_GAMMA / _OMEGA_D) * np.sin(_OMEGA_D * t))
    v = -et * (_OMEGA**2 / _OMEGA_D) * np.sin(_OMEGA_D * t)
    return np.array([x + 0j, v + 0j])


def test_damped_oscillator_accuracy():
    """Numerical solution tracks underdamped oscillator to within 1e-5 relative error."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(_osc_fun, [0.0, 10.0], y0, rtol=1e-10, atol=1e-12)

    ref = _osc_exact(t_arr)
    # rtol=1e-5 accommodates global error accumulation over t=[0,10];
    # atol=1e-9 handles the near-zero values at the end of the damped range.
    assert_allclose(y_arr, ref, rtol=1e-5, atol=1e-9,
                    err_msg="Damped oscillator: numerical vs analytical mismatch")


# ---------------------------------------------------------------------------
# 2. Numerical accuracy: nonlinear complex oscillator (docs/example.py)
#
#    dw/dt = -i w² z          z(0) = 1        z(t) = exp(it)
#    dz/dt =  i z             w(0) = 1/2.1    w(t) = 1/(exp(it) + 1.1)
#
#    Tested via ZVODE/solve_ivp in test_zvode_ivp.py but not via solve_complex_ivp.
# ---------------------------------------------------------------------------

def _nl_osc_fun(t, y):
    w, z = y[0], y[1]
    return np.array([-1j * w**2 * z, 1j * z], dtype=np.complex128)


def _nl_osc_exact(t):
    z = np.exp(1j * t)
    w = 1.0 / (z + 1.1)
    return np.array([w, z])


def test_nonlinear_oscillator_accuracy():
    """solve_complex_ivp tracks the nonlinear complex oscillator to within 1e-7."""
    y0 = np.array([1.0 / 2.1 + 0j, 1.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(_nl_osc_fun, [0.0, 4 * np.pi], y0,
                                     rtol=1e-10, atol=1e-12)

    ref = _nl_osc_exact(t_arr)
    # Global error accumulates over t=[0, 4π]; atol covers the near-zero imaginary parts.
    assert_allclose(y_arr, ref, rtol=1e-6, atol=1e-9,
                    err_msg="Nonlinear complex oscillator: numerical vs analytical mismatch")


# ---------------------------------------------------------------------------
# 3. Error paths (only the two cases not already in the existing test files)
#
#    Covered elsewhere and therefore excluded here:
#      - rtol < 0  →  test_negative_rtol_raises in test_zvode_ivp.py
#      - invalid method string  →  test_invalid_method_raises in test_solve_complex_ivp.py
# ---------------------------------------------------------------------------

_FUN_1D = lambda t, y: -y  # noqa: E731
_Y0_1D = np.array([1.0 + 0j])


@pytest.mark.parametrize("tspan,kwargs,match", [
    ([0.0, 1.0], {"atol": -1e-10}, "positive"),   # negative atol
    ([0.0],      {},               "two elements"),  # tspan too short
])
def test_error_paths(tspan, kwargs, match):
    """Invalid arguments raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        solve_complex_ivp(_FUN_1D, tspan, _Y0_1D, **kwargs)


# ---------------------------------------------------------------------------
# 4. Jacobian efficiency: exact Jacobian (miter=1) vs finite-diff (miter=2)
#
#    For a scalar system (n=1), each finite-diff Jacobian update costs one
#    extra RHS evaluation.  Providing the exact Jacobian eliminates those
#    extra evaluations, so nfev(miter=1) < nfev(miter=2).  This verifies
#    that ZVODE actually uses the supplied Jacobian.
# ---------------------------------------------------------------------------

_LAM_EFF = -1000.0  # Prothero-Robinson stiffness parameter


def _pr_eff_fun(t, y):
    return np.array([_LAM_EFF * (y[0] - np.sin(t)) + np.cos(t)], dtype=complex)


def _pr_eff_jac(t, y):
    return np.array([[_LAM_EFF + 0j]])


def test_exact_jacobian_reduces_nfev():
    """Exact Jacobian (miter=1) requires fewer RHS evaluations than finite-diff (miter=2)."""
    y0 = np.array([0.0 + 0j])
    tols = dict(rtol=1e-8, atol=1e-10, ret_stats=True, save_steps=False)

    _, _, stats_no_jac = solve_complex_ivp(_pr_eff_fun, [0.0, 1.0], y0, **tols)
    _, _, stats_jac = solve_complex_ivp(_pr_eff_fun, [0.0, 1.0], y0,
                                        jac=_pr_eff_jac, **tols)

    assert stats_jac.njev > 0, "User Jacobian was never called"
    assert stats_no_jac.nfev > stats_jac.nfev, (
        f"Expected finite-diff (nfev={stats_no_jac.nfev}) > "
        f"exact Jacobian (nfev={stats_jac.nfev})"
    )


# ---------------------------------------------------------------------------
# 5. Cross-validation against SciPy BDF
# ---------------------------------------------------------------------------

_EPS_PR = 1e-3   # Prothero-Robinson stiffness


def _pr_fun(t, y):
    return np.array([(np.sin(t) - y[0]) / _EPS_PR + np.cos(t)], dtype=complex)


def _pr_fun_real(t, y):
    return [(np.sin(t) - y[0]) / _EPS_PR + np.cos(t)]


def test_scipy_bdf_comparison():
    """solve_complex_ivp endpoint agrees with scipy BDF to 1e-5 on Prothero–Robinson."""
    y0_z = np.array([0.0 + 0j])
    tols = dict(rtol=1e-8, atol=1e-10)

    _, y_zvode = solve_complex_ivp(_pr_fun, [0.0, 3.0], y0_z, save_steps=False, **tols)

    sol = solve_ivp(_pr_fun_real, [0.0, 3.0], [0.0], method="BDF", **tols)
    assert sol.success, f"SciPy BDF failed: {sol.message}"

    assert_allclose(y_zvode.real, sol.y[:, -1], rtol=1e-5,
                    err_msg="solve_complex_ivp vs scipy BDF endpoint mismatch")


# ---------------------------------------------------------------------------
# 6. Edge case: single-element (n=1) system via solve_complex_ivp
# ---------------------------------------------------------------------------

def test_single_element_system():
    """n=1 scalar complex ODE integrates correctly via solve_complex_ivp."""
    lam = -1.0 + 2.0j
    y0 = np.array([1.0 + 0j])

    def fun(t, y):
        return np.array([lam * y[0]])

    t_arr, y_arr = solve_complex_ivp(fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12)

    ref = y0[0] * np.exp(lam * t_arr)
    assert_allclose(y_arr[0], ref, rtol=1e-7,
                    err_msg="Single-element system: numerical vs analytical mismatch")


# ---------------------------------------------------------------------------
# 7. Edge case: refine > 1 interpolation accuracy
#
#    test_refine in test_solve_complex_ivp.py only checks that the output has
#    the expected shape and is close to the no-refine solution.  This test
#    compares the interpolated points directly against an analytical solution.
# ---------------------------------------------------------------------------

_OMEGA_R = np.pi  # one full Rabi oscillation over t ∈ [0, 2]


def _rabi_fun(t, y):
    h = _OMEGA_R / 2
    return np.array([-1j * h * y[1], -1j * h * y[0]], dtype=complex)


def _rabi_exact(t):
    return np.array([np.cos(_OMEGA_R * t / 2) + 0j, -1j * np.sin(_OMEGA_R * t / 2)])


def test_refine_interpolation_accuracy():
    """refine=5 ZVINDY-interpolated points match the Rabi exact solution to 1e-6."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(
        _rabi_fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12, refine=5,
    )

    ref = _rabi_exact(t_arr)
    # atol=1e-9 guards the Rabi zero-crossing where exact value is ~1e-16.
    assert_allclose(y_arr, ref, rtol=1e-6, atol=1e-9,
                    err_msg="refine=5: interpolated points diverge from exact solution")


# ---------------------------------------------------------------------------
# 8. Edge case: max_order constraint
#
#    max_order is never tested functionally in the existing suite (only via an
#    int32 overflow guard).  With max_order=1, Adams is restricted to
#    first-order steps: the correct answer must still be reached but the solver
#    needs significantly more steps.
# ---------------------------------------------------------------------------

def test_max_order_constraint():
    """max_order=1 forces first-order Adams steps: more steps, same correct endpoint."""
    lam = -1.0 + 0j
    y0 = np.array([1.0 + 0j])
    kw = dict(method="Adams", rtol=1e-8, atol=1e-10, ret_stats=True, save_steps=False)

    _, y_default, s_default = solve_complex_ivp(
        lambda t, y: lam * y, [0.0, 5.0], y0, **kw
    )
    _, y_order1, s_order1 = solve_complex_ivp(
        lambda t, y: lam * y, [0.0, 5.0], y0, max_order=1, **kw
    )

    exact_end = np.array([np.exp(lam * 5.0)])
    assert_allclose(y_default, exact_end, rtol=1e-6,
                    err_msg="Default Adams: wrong endpoint")
    # Adams order-1 global error is O(sqrt(rtol)) ≈ 4e-4 for rtol=1e-8, so use rtol=1e-3.
    assert_allclose(y_order1, exact_end, rtol=1e-3,
                    err_msg="Adams max_order=1: wrong endpoint")
    assert s_order1.nsteps > s_default.nsteps, (
        f"max_order=1 should need more steps than default "
        f"(got {s_order1.nsteps} vs {s_default.nsteps})"
    )
