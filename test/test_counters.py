"""Counter-accuracy tests for nfev / njev.

ZVODE maintains its NFE/NJE counters inside the Fortran COMMON block and
writes them to iwork[11] / iwork[12] after each accepted step.  The Python
layer calls the user's fun (and jac) once at construction time to probe the
return shape (_validate_fun_shape / _validate_jac_shape in _helpers.py).
These probe calls are tracked in Python-side counters and added to the
Fortran readout values, so the reported nfev / njev equal the true total
number of callback invocations.

The fix applies only to the in_place=False Python callback path; compiled
callbacks (ctypes / numba) skip the shape probe and are already exact.
"""

import numpy as np
import pytest

from zvode import ZVODE, solve_complex_ivp

# ---------------------------------------------------------------------------
# Shared problem: 2-component complex decay
#   dy/dt = diag(lam) * y,  lam = [-1+0j, -2+0j]
#   Exact: y_i(t) = y0_i * exp(lam_i * t)
# ---------------------------------------------------------------------------

LAM = np.array([-1.0 + 0j, -2.0 + 0j])
Y0 = np.array([1.0 + 0j, 0.5 + 0j])
TSPAN = [0.0, 2.0]
TOLS = dict(rtol=1e-8, atol=1e-10)


def _make_fun():
    """Return (fun, counter) where counter[0] is incremented on every call."""
    counter = [0]

    def fun(t, y):
        counter[0] += 1
        return LAM * y

    return fun, counter


def _make_jac():
    """Return (jac, counter) for a dense diagonal Jacobian."""
    counter = [0]

    def jac(t, y):
        counter[0] += 1
        return np.diag(LAM)

    return jac, counter


# ---------------------------------------------------------------------------
# solve_complex_ivp — nfev
# ---------------------------------------------------------------------------


def test_nfev_includes_probe_out_of_place():
    """result.nfev equals total fun calls including the shape-probe."""
    fun, counter = _make_fun()
    result = solve_complex_ivp(fun, TSPAN, Y0, **TOLS)
    assert result.nfev == counter[0]


# ---------------------------------------------------------------------------
# solve_complex_ivp — njev
# ---------------------------------------------------------------------------


def test_njev_includes_probe_out_of_place():
    """result.njev equals total jac calls including the shape-probe (in_place=False)."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    result = solve_complex_ivp(fun, TSPAN, Y0, jac=jac, **TOLS)
    assert result.njev == jac_counter[0]


# ---------------------------------------------------------------------------
# FD Jacobian nfev accounting (miter=2)
# ---------------------------------------------------------------------------
#
# When no jac is supplied, BDF defaults to miter=2: internally generated
# finite-difference Jacobian.  Each Jacobian assembly perturbs each of the n
# components in turn, so it costs exactly n extra function evaluations that
# are counted in nfev, not in a separate counter.  njev still counts the
# number of Jacobian assemblies, just as it does for an analytic Jacobian.
#
# Empirically (see probe above): nfev_FD - nfev_analytic == n * njev for runs
# with the same step sequence.  We verify this by solving the same problem
# twice at identical tolerances and comparing.


def test_fd_jacobian_nfev_overhead():
    """FD Jacobian (miter=2) costs exactly n extra nfev per assembly vs analytic."""
    n = len(Y0)
    fun, _ = _make_fun()
    jac, _ = _make_jac()

    result_fd = solve_complex_ivp(fun, TSPAN, Y0, method="BDF", **TOLS)
    result_analytic = solve_complex_ivp(fun, TSPAN, Y0, method="BDF", jac=jac, **TOLS)

    # The analytic path has one extra njev for the construction-time shape probe;
    # during integration both paths assemble the Jacobian the same number of times.
    assert result_analytic.njev == result_fd.njev + 1
    # The nfev probe offset is +1 for both paths, so it cancels in the difference.
    assert result_fd.nfev - result_analytic.nfev == n * result_fd.njev


# ---------------------------------------------------------------------------
# ZVODE class — nfev / njev
# ---------------------------------------------------------------------------


def _run_zvode(fun, jac=None):
    """Construct a ZVODE solver and step it to t_bound; return the solver."""
    solver = ZVODE(fun, TSPAN[0], Y0, TSPAN[1], jac=jac, **TOLS)
    while solver.status == "running":
        solver.step()
    assert solver.status == "finished", f"solver failed: {solver.status}"
    return solver


def test_zvode_nfev_includes_probe():
    """ZVODE.nfev equals total fun calls including the constructor probe."""
    fun, counter = _make_fun()
    solver = _run_zvode(fun)
    assert solver.nfev == counter[0]


def test_zvode_njev_includes_probe():
    """ZVODE.njev equals total jac calls including the constructor probe."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    solver = _run_zvode(fun, jac=jac)
    assert solver.njev == jac_counter[0]
