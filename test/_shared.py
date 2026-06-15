"""Shared test fixtures: the canonical coupled 2-component complex ODE, its
analytic solution, and low-level ctypes view helpers.

This problem is exercised by the procedural-interface tests
(``test_solve_complex_ivp.py``) and by both compiled-callback suites
(``test_ctypes_callbacks.py``, ``test_numba_callbacks.py``), which previously
each carried their own byte-for-byte copy of the definitions below.

Problem:

    dy[0]/dt = LAM1*y[0] + C*y[1]
    dy[1]/dt = LAM2*y[1]

Analytic solution:

    y[1](t) = Y0[1]*exp(LAM2*t)
    y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t),  B = C*Y0[1]/(LAM2-LAM1), A = Y0[0]-B

The Jacobian is upper triangular (J[1,0]=0), so the banded storage uses
lband=0, uband=1.

This module is a plain importable helper (``from _shared import ...``), not a
pytest plugin; it deliberately defines no fixtures and is not collected.
"""

import ctypes

import numpy as np

# Problem parameters
LAM1 = -1 + 2j
LAM2 = -2 + 1j
C = 0.5j
Y0 = np.array([1.0 + 0j, 0.0 + 1j])
T0 = 0.0
TF = 2.0

# Banded Jacobian half-bandwidths (upper triangular: J[1,0] = 0)
LBAND = 0
UBAND = 1

# Integration tolerances tight enough for 1e-5 solution accuracy
RTOL = 1e-8
ATOL = 1e-10

_B = C * Y0[1] / (LAM2 - LAM1)
_A = Y0[0] - _B


def coupled_exact(t):
    """Analytic solution at time(s) ``t``, returned as a shape ``(2, ...)`` array."""
    t = np.asarray(t, dtype=float)
    y0 = _A * np.exp(LAM1 * t) + _B * np.exp(LAM2 * t)
    y1 = Y0[1] * np.exp(LAM2 * t)
    return np.array([y0, y1])


def assert_coupled(t_arr, y_arr, sol_rtol=1e-5):
    """Assert a trajectory matches the coupled-system analytic solution."""
    ref = coupled_exact(t_arr)
    assert np.allclose(y_arr, ref, rtol=sol_rtol), (
        f"max err={np.max(np.abs(y_arr - ref)):.2e}"
    )


def coupled_fun(t, y):
    """Python (return-value) RHS for the coupled system."""
    dy = np.empty(len(y), dtype=np.complex128)
    dy[0] = LAM1 * y[0] + C * y[1]
    dy[1] = LAM2 * y[1]
    return dy


def coupled_jac_dense(t, y):
    """Dense (n, n) Jacobian for the coupled system."""
    pd = np.zeros((len(y), len(y)), dtype=np.complex128)
    pd[0, 0] = LAM1
    pd[0, 1] = C
    pd[1, 1] = LAM2
    return pd


def coupled_jac_banded(t, y):
    """Banded Jacobian (ZVODE storage ``pd[uband + i - j, j] = J[i, j]``)."""
    pd = np.zeros((LBAND + UBAND + 1, len(y)), dtype=np.complex128)
    pd[UBAND, 0] = LAM1  # J[0, 0]
    pd[UBAND - 1, 1] = C  # J[0, 1]
    pd[UBAND, 1] = LAM2  # J[1, 1]
    return pd


def pack_banded(a, lband, uband):
    """Pack a dense matrix into ZVODE band storage ``packed[i-j+uband, j]``.

    Shape ``(lband + uband + 1, n)`` — the layout shared verbatim by
    ``solve_complex_ivp`` and ``scipy.integrate.ode('zvode')``.
    """
    n = a.shape[0]
    pd = np.zeros((lband + uband + 1, n), dtype=np.complex128)
    for j in range(n):
        for i in range(max(0, j - uband), min(n, j + lband + 1)):
            pd[i - j + uband, j] = a[i, j]
    return pd


def ro128(addr, count):
    """Read-only complex128 view of *count* elements at raw address *addr*."""
    buf = (ctypes.c_double * (2 * count)).from_address(addr)
    return np.frombuffer(buf, dtype=np.complex128)


def rw128(addr, count):
    """Writable complex128 view of *count* elements at raw address *addr*."""
    buf = (ctypes.c_double * (2 * count)).from_address(addr)
    return np.ctypeslib.as_array(buf).view(np.complex128)
