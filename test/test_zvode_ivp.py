"""Tests for the ZVODE OdeSolver class via scipy.integrate.solve_ivp.

Two analytic examples from the docs folder are used (the QME example is
excluded):

1. Complex exponential decay (docs/demo.py):
       dy/dt = -y,  y(0) = 0.5+1j,  y(t) = (0.5+1j)*exp(-t)

2. Complex oscillator (docs/example.py):
       dw/dt = -i*w^2*z,  dz/dt = i*z
       w(0) = 1/2.1,  z(0) = 1
       Solution:  z(t) = exp(it),  w(t) = 1/(exp(it) + 1.1)

All four user-facing miter options are tested:
    miter=1 – BDF + user-supplied dense Jacobian
    miter=2 – BDF + internally-generated dense Jacobian
    miter=3 – BDF + diagonal Jacobian approximation
    miter=4 – BDF + user-supplied banded Jacobian

Correctness is checked at the automatically selected output points.

# TODO: add tests for dense output (solve_ivp dense_output=True) once
#       that path is fully wired up in the ZVODE solver.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from zvode import ZVODE


# ---------------------------------------------------------------------------
# Example 1: complex exponential decay  (docs/demo.py)
# ---------------------------------------------------------------------------

def fun_decay(t, y):
    return -y


def jac_decay_dense(t, y):
    return -np.eye(len(y), dtype=np.complex128)


def jac_decay_banded(t, y):
    # ml=0, mu=0 -> band array shape (1, n); diagonal entry is J[j,j] = -1
    n = len(y)
    pd = np.zeros((1, n), dtype=np.complex128)
    pd[0, :] = -1.0
    return pd


def sol_decay(t, y0):
    """Analytic solution: y(t) = y0 * exp(-t)."""
    return y0 * np.exp(-np.asarray(t))


# ---------------------------------------------------------------------------
# Example 2: complex oscillator  (docs/example.py)
# ---------------------------------------------------------------------------

def fun_oscillator(t, y):
    w, z = y[0], y[1]
    return np.array([-1j * w**2 * z, 1j * z], dtype=np.complex128)


def jac_oscillator_dense(t, y):
    w, z = y[0], y[1]
    J = np.zeros((2, 2), dtype=np.complex128)
    J[0, 0] = -2j * w * z
    J[0, 1] = -1j * w**2
    J[1, 1] = 1j
    return J


def jac_oscillator_banded(t, y):
    # ml=0, mu=1 -> band array shape (2, 2)
    # Storage convention: pd[i - j + mu, j] = J[i, j]
    #   J[0,0] -> pd[0-0+1, 0] = pd[1, 0]
    #   J[0,1] -> pd[0-1+1, 1] = pd[0, 1]
    #   J[1,1] -> pd[1-1+1, 1] = pd[1, 1]
    w, z = y[0], y[1]
    pd = np.zeros((2, 2), dtype=np.complex128)
    pd[1, 0] = -2j * w * z
    pd[0, 1] = -1j * w**2
    pd[1, 1] = 1j
    return pd


def sol_oscillator(t):
    """Analytic solution: z(t) = exp(it), w(t) = 1/(exp(it) + 1.1)."""
    z = np.exp(1j * t)
    w = 1.0 / (z + 1.1)
    return np.array([w, z], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Decay problem: miter 1, 2, 3 (dense / no explicit Jacobian)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("miter,jac", [
    (1, jac_decay_dense),
    (2, None),
    (3, None),
])
def test_decay_dense_miters(miter, jac):
    """Complex decay solved with dense miter options 1, 2, 3."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)

    sol = solve_ivp(fun_decay, t_span, y0,
                    method=ZVODE,
                    jac=jac,
                    miter=miter,
                    rtol=1e-8, atol=1e-10)

    assert sol.success, f"miter={miter}: {sol.message}"
    assert sol.status == 0
    assert sol.t[0] == t_span[0]
    assert sol.t[-1] == t_span[1]
    assert sol.nfev > 0
    assert sol.y.shape == (1, len(sol.t))
    assert sol.y.dtype == np.complex128

    expected = sol_decay(sol.t, y0[0])
    assert_allclose(sol.y[0], expected, rtol=1e-5, atol=1e-8,
                    err_msg=f"miter={miter}: solution mismatch")


# ---------------------------------------------------------------------------
# Decay problem: miter 4 (banded Jacobian, ml=0 mu=0)
# ---------------------------------------------------------------------------

def test_decay_miter4_banded():
    """Complex decay solved with user-supplied banded Jacobian (miter=4, ml=0, mu=0)."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)

    sol = solve_ivp(fun_decay, t_span, y0,
                    method=ZVODE,
                    jac=jac_decay_banded,
                    miter=4,
                    lband=0, uband=0,
                    rtol=1e-8, atol=1e-10)

    assert sol.success, f"miter=4: {sol.message}"
    assert sol.status == 0
    assert sol.t[-1] == t_span[1]
    assert sol.nfev > 0
    assert sol.y.dtype == np.complex128

    expected = sol_decay(sol.t, y0[0])
    assert_allclose(sol.y[0], expected, rtol=1e-5, atol=1e-8,
                    err_msg="miter=4: solution mismatch")


# ---------------------------------------------------------------------------
# Oscillator problem: miter 1, 2, 3, 4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("miter,jac,extra_kwargs", [
    (1, jac_oscillator_dense, {}),
    (2, None, {}),
    (3, None, {}),
    (4, jac_oscillator_banded, {'lband': 0, 'uband': 1}),
])
def test_oscillator_miter(miter, jac, extra_kwargs):
    """Complex oscillator trajectory checked at solver-selected output points."""
    t0 = 0.0
    t_end = 2 * np.pi
    y0 = np.array([1.0 / 2.1, 1.0], dtype=np.complex128)

    sol = solve_ivp(fun_oscillator, (t0, t_end), y0,
                    method=ZVODE,
                    jac=jac,
                    miter=miter,
                    rtol=1e-9, atol=1e-9,
                    **extra_kwargs)

    assert sol.success, f"miter={miter}: {sol.message}"
    assert sol.status == 0
    assert sol.t[0] == t0
    assert sol.t[-1] == t_end
    assert sol.nfev > 0
    assert sol.y.shape == (2, len(sol.t))
    assert sol.y.dtype == np.complex128

    for i, t in enumerate(sol.t):
        expected = sol_oscillator(t)
        assert_allclose(sol.y[:, i], expected, rtol=1e-5, atol=1e-7,
                        err_msg=f"miter={miter}: solution mismatch at t={t:.4f}")


# ---------------------------------------------------------------------------
# Solver counters
# ---------------------------------------------------------------------------

def test_solver_counters_with_jacobian():
    """With a user Jacobian (miter=1), njev should be positive."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol = solve_ivp(fun_decay, (0.0, 1.0), y0,
                    method=ZVODE,
                    jac=jac_decay_dense,
                    miter=1,
                    rtol=1e-8, atol=1e-10)

    assert sol.success
    assert sol.nfev > 0
    assert sol.njev > 0
    assert sol.nlu > 0


def test_solver_counters_without_jacobian():
    """Without a Jacobian (miter=2), nfev should be positive and njev may be 0."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol = solve_ivp(fun_decay, (0.0, 1.0), y0,
                    method=ZVODE,
                    miter=2,
                    rtol=1e-8, atol=1e-10)

    assert sol.success
    assert sol.nfev > 0


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
