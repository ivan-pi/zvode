"""Shared test problem: the canonical coupled 2-component complex ODE.

A single ``Problem`` instance, :data:`COUPLED`, is the one definition of the
system exercised by the procedural-interface suite
(``test_solve_complex_ivp.py``) and both compiled-callback suites
(``test_ctypes_callbacks.py``, ``test_numba_callbacks.py``), which previously
each carried their own byte-for-byte copy.

Each entry point wants a *different* callable signature — a return-value Python
callback, a ctypes / numba ``@cfunc``, or a SciPy ``ode`` dataclass — so the
class is not itself "the callback".  Instead a test pulls the fields it needs
from the instance and builds the concrete callback at the call site; the
``fun`` / ``jac_*`` methods below cover the plain return-value Python path
directly, while the compiled suites unpack ``lam1/lam2/c`` and capture them as
constants in a hand-written ``@cfunc``.

This module is a plain importable helper (``from shared import ...``), not a
pytest plugin; it defines no fixtures and is not collected.
"""

import ctypes
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Problem:
    """A coupled 2-component complex IVP and everything tests need from it.

        dy[0]/dt = lam1*y[0] + c*y[1]
        dy[1]/dt = lam2*y[1]

    The Jacobian is upper triangular (J[1,0] = 0), so the banded storage uses
    ``lband=0, uband=1``.  Analytic solution:

        y[1](t) = y0[1]*exp(lam2*t)
        y[0](t) = A*exp(lam1*t) + B*exp(lam2*t),
                  B = c*y0[1]/(lam2 - lam1),  A = y0[0] - B
    """

    lam1: complex = -1 + 2j
    lam2: complex = -2 + 1j
    c: complex = 0.5j
    y0: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j, 0.0 + 1j]))
    t0: float = 0.0
    tf: float = 2.0
    lband: int = 0
    uband: int = 1
    # Integration tolerances tight enough for 1e-5 solution accuracy.
    rtol: float = 1e-8
    atol: float = 1e-10

    @property
    def tspan(self):
        return [self.t0, self.tf]

    @property
    def tols(self):
        return dict(rtol=self.rtol, atol=self.atol)

    def fun(self, t, y):
        """Python (return-value) RHS."""
        dy = np.empty(len(y), dtype=np.complex128)
        dy[0] = self.lam1 * y[0] + self.c * y[1]
        dy[1] = self.lam2 * y[1]
        return dy

    def jac_dense(self, t, y):
        """Dense (n, n) Jacobian."""
        pd = np.zeros((len(y), len(y)), dtype=np.complex128)
        pd[0, 0] = self.lam1
        pd[0, 1] = self.c
        pd[1, 1] = self.lam2
        return pd

    def jac_banded(self, t, y):
        """Banded Jacobian (ZVODE storage ``pd[uband + i - j, j] = J[i, j]``)."""
        pd = np.zeros((self.lband + self.uband + 1, len(y)), dtype=np.complex128)
        pd[self.uband, 0] = self.lam1  # J[0, 0]
        pd[self.uband - 1, 1] = self.c  # J[0, 1]
        pd[self.uband, 1] = self.lam2  # J[1, 1]
        return pd

    def exact(self, t):
        """Analytic solution at time(s) ``t``, as a shape ``(2, ...)`` array."""
        t = np.asarray(t, dtype=float)
        b = self.c * self.y0[1] / (self.lam2 - self.lam1)
        a = self.y0[0] - b
        sol0 = a * np.exp(self.lam1 * t) + b * np.exp(self.lam2 * t)
        sol1 = self.y0[1] * np.exp(self.lam2 * t)
        return np.array([sol0, sol1])

    def assert_close(self, t_arr, y_arr, sol_rtol=1e-5):
        """Assert a trajectory matches the analytic solution."""
        ref = self.exact(t_arr)
        assert np.allclose(y_arr, ref, rtol=sol_rtol), (
            f"max err={np.max(np.abs(y_arr - ref)):.2e}"
        )


# The single shared instance used across the suite.
COUPLED = Problem()


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
