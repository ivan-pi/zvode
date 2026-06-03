"""
Regression and validation tests for solve_complex_ivp:

  1. Damped harmonic oscillator accuracy (Adams, BDF)
  2. Nonlinear complex oscillator accuracy
  3. Error paths: negative atol, short tspan
  4. Exact Jacobian reduces RHS evaluations vs finite-diff
  5. Cross-validation against scipy.integrate.solve_ivp(method='BDF') on Van der Pol (n=2)
  6. Single-element (n=1) system: decay, damped oscillation, pure rotation
  7. refine > 1 interpolation accuracy (refine=2, refine=5)
  8. max_order constrains Adams solver order (n=2 decoupled decay)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from zvode import solve_complex_ivp


# ---------------------------------------------------------------------------
# 1. Numerical accuracy: underdamped harmonic oscillator
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
    sol = solve_complex_ivp(osc_fun, [0.0, 10.0], y0, method=method, rtol=1e-10, atol=1e-12)

    ref = osc_exact(sol.t)
    # rtol=1e-5 accommodates global error accumulation over t=[0,10];
    # atol=1e-9 handles near-zero values at the end of the damped range.
    assert_allclose(sol.y, ref, rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Numerical accuracy: nonlinear complex oscillator (docs/example.py)
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
# 3. Error paths
# ---------------------------------------------------------------------------

FUN_1D = lambda t, y: -y  # noqa: E731
Y0_1D = np.array([1.0 + 0j])


@pytest.mark.parametrize("tspan,kwargs,match", [
    ([0.0, 1.0], {"atol": -1e-10}, "positive"),    # negative atol
    ([0.0],      {},               "two elements"), # tspan too short
])
def test_error_paths(tspan, kwargs, match):
    """Invalid arguments raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        solve_complex_ivp(FUN_1D, tspan, Y0_1D, **kwargs)


# ---------------------------------------------------------------------------
# 4. Jacobian efficiency: exact Jacobian (miter=1) vs finite-diff (miter=2)
#
#    For n=1 each finite-diff Jacobian update costs one extra RHS evaluation,
#    so nfev(miter=2) > nfev(miter=1) is a precise and checkable inequality.
# ---------------------------------------------------------------------------

LAM_EFF = -1000.0  # Prothero-Robinson stiffness parameter


def pr_eff_fun(t, y):
    return np.array([LAM_EFF * (y[0] - np.sin(t)) + np.cos(t)], dtype=complex)


def pr_eff_jac(t, y):
    return np.array([[LAM_EFF + 0j]])


def test_exact_jacobian_reduces_nfev():
    """Exact Jacobian (miter=1) requires fewer RHS evaluations than finite-diff (miter=2)."""
    y0 = np.array([0.0 + 0j])
    tols = dict(rtol=1e-8, atol=1e-10, save_steps=False)

    sol_no_jac = solve_complex_ivp(pr_eff_fun, [0.0, 1.0], y0, **tols)
    sol_jac = solve_complex_ivp(pr_eff_fun, [0.0, 1.0], y0, jac=pr_eff_jac, **tols)

    assert sol_jac.njev > 0, "User Jacobian was never called"
    assert sol_no_jac.nfev > sol_jac.nfev, (
        f"Expected finite-diff (nfev={sol_no_jac.nfev}) > "
        f"exact Jacobian (nfev={sol_jac.nfev})"
    )


# ---------------------------------------------------------------------------
# 5. Cross-validation against SciPy BDF
# ---------------------------------------------------------------------------

VDP_MU = 10.0  # Van der Pol stiffness (mildly stiff, nonlinear, n=2)


def vdp_fun(t, y):
    return np.array([y[1], VDP_MU * (1 - y[0]**2) * y[1] - y[0]], dtype=complex)


def vdp_fun_real(t, y):
    return [y[1], VDP_MU * (1 - y[0]**2) * y[1] - y[0]]


def test_scipy_bdf_comparison():
    """solve_complex_ivp endpoint agrees with scipy BDF to 1e-5 on the Van der Pol oscillator."""
    y0_z = np.array([2.0 + 0j, 0.0 + 0j])
    tols = dict(rtol=1e-8, atol=1e-10)

    sol_zvode = solve_complex_ivp(vdp_fun, [0.0, 0.5], y0_z, save_steps=False, **tols)
    sol_scipy = solve_ivp(vdp_fun_real, [0.0, 0.5], [2.0, 0.0], method="BDF", **tols)

    assert sol_scipy.success, f"SciPy BDF failed: {sol_scipy.message}"
    assert_allclose(sol_zvode.y.real, sol_scipy.y[:, -1], rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Edge case: single-element (n=1) system
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [
    pytest.param(-1.0 + 0j, id="decay"),
    pytest.param(-1.0 + 2j, id="damped_osc"),
    pytest.param(1j,         id="rotation"),
])
def test_single_element_system(lam):
    """n=1 scalar complex ODE y'=lam*y integrates correctly for three qualitatively different lam."""
    y0 = np.array([1.0 + 0j])
    sol = solve_complex_ivp(
        lambda t, y: np.array([lam * y[0]]),
        [0.0, 2.0], y0, rtol=1e-10, atol=1e-12,
    )
    assert_allclose(sol.y[0], y0[0] * np.exp(lam * sol.t), rtol=1e-7)


# ---------------------------------------------------------------------------
# 7. Edge case: refine > 1 interpolation accuracy
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
    sol = solve_complex_ivp(rabi_fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12, refine=refine)

    ref = rabi_exact(sol.t)
    # atol=1e-9 guards the zero-crossing where the exact value is ~1e-16.
    assert_allclose(sol.y, ref, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# 8. Edge case: max_order constraint
# ---------------------------------------------------------------------------

# Two-component decoupled linear system: y' = diag(lam1, lam2) * y
# Exact endpoint: y[i](T) = y0[i] * exp(lam[i] * T)
MAX_ORDER_LAM = np.array([-1.0 + 0j, -2.0 + 0j])
MAX_ORDER_T = 5.0


def max_order_fun(t, y):
    return MAX_ORDER_LAM * y


def test_max_order_constraint():
    """max_order=1 forces first-order Adams steps: more steps, same correct endpoint."""
    y0 = np.array([1.0 + 0j, 1.0 + 0j])
    kw = dict(method="Adams", rtol=1e-8, atol=1e-10, save_steps=False)

    sol_default = solve_complex_ivp(max_order_fun, [0.0, MAX_ORDER_T], y0, **kw)
    sol_order1 = solve_complex_ivp(max_order_fun, [0.0, MAX_ORDER_T], y0, max_order=1, **kw)

    exact_end = y0 * np.exp(MAX_ORDER_LAM * MAX_ORDER_T)
    assert_allclose(sol_default.y, exact_end, rtol=1e-6)
    # Adams order-1 global error is O(sqrt(rtol)) ≈ 4e-4 for rtol=1e-8.
    assert_allclose(sol_order1.y, exact_end, rtol=1e-3)
    assert sol_order1.nsteps > sol_default.nsteps, (
        f"max_order=1 should need more steps than default "
        f"(got {sol_order1.nsteps} vs {sol_default.nsteps})"
    )
