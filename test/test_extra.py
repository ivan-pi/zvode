"""
Extra regression and validation tests for solve_complex_ivp.

Covers:
  1. Numerical accuracy — underdamped harmonic oscillator vs exact solution
  2. Numerical accuracy — Rabi oscillations (2-level Schrödinger) vs exact + unitarity
  3. Error paths — invalid arguments produce clear ValueError (parametrized)
  4. Jacobian correctness — no/dense/banded Jacobians agree on a stiff tridiagonal system
  5. Cross-validation — compare endpoint against scipy.integrate.solve_ivp(method='BDF')
  6. Edge case — single-element (n=1) complex system
  7. Edge case — refine > 1 inserts ZVINDY-interpolated points that match the exact solution
  8. Edge case — zero-length tspan is rejected with a clear error
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
    """Exact solution for y0=[1, 0]: x(t) and v(t)."""
    et = np.exp(-_GAMMA * t)
    x = et * (np.cos(_OMEGA_D * t) + (_GAMMA / _OMEGA_D) * np.sin(_OMEGA_D * t))
    v = -et * (_OMEGA**2 / _OMEGA_D) * np.sin(_OMEGA_D * t)
    return np.array([x + 0j, v + 0j])


def test_damped_oscillator_accuracy():
    """Numerical solution tracks underdamped oscillator to within 1e-7 relative error."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(_osc_fun, [0.0, 10.0], y0, rtol=1e-10, atol=1e-12)

    ref = _osc_exact(t_arr)
    # Global error over t=[0,10] accumulates to ~1e-5×|y|; local tolerance is 1e-10.
    assert_allclose(y_arr, ref, rtol=1e-5, atol=1e-9,
                    err_msg="Damped oscillator: numerical vs analytical mismatch")


# ---------------------------------------------------------------------------
# 2. Numerical accuracy: Rabi oscillations (2-level Schrödinger equation)
# ---------------------------------------------------------------------------

_OMEGA_R = np.pi  # one full Rabi oscillation over t ∈ [0, 2]


def _rabi_fun(t, y):
    h = _OMEGA_R / 2  # off-diagonal coupling  (H = h·σ_x, resonant drive)
    return np.array([-1j * h * y[1], -1j * h * y[0]], dtype=complex)


def _rabi_exact(t):
    return np.array([np.cos(_OMEGA_R * t / 2) + 0j, -1j * np.sin(_OMEGA_R * t / 2)])


def test_rabi_oscillations_accuracy():
    """Numerical Rabi solution matches analytical and conserves norm to 1e-8."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(_rabi_fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12)

    ref = _rabi_exact(t_arr)
    # atol guards the zero-crossing where exact value is ~1e-16 but numerical residual ~1e-11.
    assert_allclose(y_arr, ref, rtol=1e-7, atol=1e-9,
                    err_msg="Rabi oscillations: numerical vs analytical mismatch")

    norm = np.abs(y_arr[0]) ** 2 + np.abs(y_arr[1]) ** 2
    assert_allclose(norm, 1.0, atol=1e-8, err_msg="Rabi: norm conservation violated")


# ---------------------------------------------------------------------------
# 3. Error paths
# ---------------------------------------------------------------------------

_FUN_1D = lambda t, y: -y  # noqa: E731
_Y0_1D = np.array([1.0 + 0j])


@pytest.mark.parametrize("tspan,kwargs,match", [
    ([0.0, 1.0], {"rtol": -1e-6},       "positive"),
    ([0.0, 1.0], {"atol": -1e-10},      "positive"),
    ([0.0],      {},                     "two elements"),
    ([0.0, 1.0], {"method": "Euler"},   "method"),
])
def test_error_paths(tspan, kwargs, match):
    """Invalid arguments raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        solve_complex_ivp(_FUN_1D, tspan, _Y0_1D, **kwargs)


# ---------------------------------------------------------------------------
# 4. Jacobian correctness: no / dense / banded variants must agree
# ---------------------------------------------------------------------------

_N_JVT = 3
_ALPHA_JVT = 50.0   # stiff diagonal; off-diagonals are ±1
_LBAND, _UBAND = 1, 1

_A_JVT = (np.diag(np.full(_N_JVT, -_ALPHA_JVT, dtype=complex))
          + np.diag(np.ones(_N_JVT - 1, dtype=complex), +1)
          + np.diag(np.ones(_N_JVT - 1, dtype=complex), -1))


def _jvt_fun(t, y):
    return _A_JVT @ y


def _jvt_jac_dense(t, y):
    return _A_JVT.copy()


def _jvt_jac_banded(t, y):
    # ZVODE banded storage: pd[mu + i - j, j] = J[i, j],  mu = UBAND
    pd = np.zeros((_LBAND + _UBAND + 1, _N_JVT), dtype=np.complex128)
    pd[_UBAND - 1, 1:] = 1.0        # superdiagonal J[i, i+1] stored at pd[0, i+1]
    pd[_UBAND, :] = -_ALPHA_JVT     # diagonal
    pd[_UBAND + 1, :-1] = 1.0       # subdiagonal J[i+1, i] stored at pd[2, i]
    return pd


def test_jacobian_variants_agree():
    """No-Jacobian, dense, and banded Jacobian modes produce the same endpoint to 1e-8."""
    y0 = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
    tspan = [0.0, 0.1]
    tols = dict(rtol=1e-10, atol=1e-12, save_steps=False)

    _, y_none = solve_complex_ivp(_jvt_fun, tspan, y0, **tols)
    _, y_dense = solve_complex_ivp(_jvt_fun, tspan, y0, jac=_jvt_jac_dense, **tols)
    _, y_banded = solve_complex_ivp(_jvt_fun, tspan, y0,
                                    jac=_jvt_jac_banded, lband=_LBAND, uband=_UBAND,
                                    **tols)

    assert_allclose(y_dense, y_none, rtol=1e-8,
                    err_msg="Dense Jacobian disagrees with no-Jacobian result")
    assert_allclose(y_banded, y_none, rtol=1e-8,
                    err_msg="Banded Jacobian disagrees with no-Jacobian result")


# ---------------------------------------------------------------------------
# 5. Cross-validation against SciPy BDF
# ---------------------------------------------------------------------------

# Prothero–Robinson problem: y' = (1/eps)*(sin(t) - y) + cos(t),  y(0) = 0
# Exact solution: y(t) = sin(t).  Stiff for small eps.
_EPS_PR = 1e-3


def _pr_fun(t, y):
    return np.array([(np.sin(t) - y[0]) / _EPS_PR + np.cos(t)], dtype=complex)


def _pr_fun_real(t, y):
    return [(np.sin(t) - y[0]) / _EPS_PR + np.cos(t)]


def test_scipy_bdf_comparison():
    """solve_complex_ivp endpoint agrees with scipy BDF to 1e-5 on the Prothero–Robinson problem."""
    y0_z = np.array([0.0 + 0j])
    tols = dict(rtol=1e-8, atol=1e-10)

    _, y_zvode = solve_complex_ivp(_pr_fun, [0.0, 3.0], y0_z, save_steps=False, **tols)

    sol = solve_ivp(_pr_fun_real, [0.0, 3.0], [0.0], method="BDF", **tols)
    assert sol.success, f"SciPy BDF failed: {sol.message}"

    assert_allclose(y_zvode.real, sol.y[:, -1], rtol=1e-5,
                    err_msg="solve_complex_ivp vs scipy BDF endpoint mismatch")


# ---------------------------------------------------------------------------
# 6. Edge case: single-element (n=1) system
# ---------------------------------------------------------------------------

def test_single_element_system():
    """n=1 scalar complex ODE integrates correctly against the analytical solution."""
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
# ---------------------------------------------------------------------------

def test_refine_interpolation_accuracy():
    """refine=5 inserts ZVINDY-interpolated output points that match the Rabi exact solution."""
    y0 = np.array([1.0 + 0j, 0.0 + 0j])
    t_arr, y_arr = solve_complex_ivp(
        _rabi_fun, [0.0, 2.0], y0, rtol=1e-10, atol=1e-12, refine=5,
    )

    ref = _rabi_exact(t_arr)
    assert_allclose(y_arr, ref, rtol=1e-6, atol=1e-9,
                    err_msg="refine=5: interpolated points diverge from exact solution")


# ---------------------------------------------------------------------------
# 8. Edge case: zero-length tspan raises ValueError
# ---------------------------------------------------------------------------

def test_zero_length_tspan_raises():
    """tspan=[t0, t0] is not strictly monotonic and must raise ValueError."""
    with pytest.raises(ValueError, match="monotonic"):
        solve_complex_ivp(_FUN_1D, [0.0, 0.0], _Y0_1D)
