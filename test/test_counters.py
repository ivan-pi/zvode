"""Counter-accuracy tests for nfev / njev.

ZVODE maintains its NFE/NJE counters inside the Fortran COMMON block and
writes them to iwork[11] / iwork[12] after each accepted step.  The Python
layer calls the user's fun (and jac) once at construction time to probe the
return shape (_validate_fun_shape / _validate_jac_shape in _helpers.py), but
these probe calls happen before the Fortran state is initialised and are
therefore invisible to the Fortran counter.

Test layout
-----------
xfail tests
    Document the *correct* expected behaviour: after a complete integration
    the reported counter should equal the total number of times the callback
    was invoked, including the constructor probe.  They currently fail because
    the fix (maintaining a Python-side probe counter and adding it to the
    Fortran value on readout) has not yet been implemented.  When the fix
    lands each xfail will become an XPASS, signalling the task is done.

Passing tests
    * ``test_nfev_accurate_when_in_place`` — with in_place=True the shape-
      probe is skipped entirely, so the counter should be exact right now.
    * ``test_*_probe_offset_is_one`` — pin the *current* discrepancy to
      exactly 1 so that a refactor cannot silently introduce a larger offset.
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


def _make_inplace_fun():
    """Return (fun, counter) using the in-place calling convention."""
    counter = [0]

    def fun(t, y, dy):
        counter[0] += 1
        dy[:] = LAM * y

    return fun, counter


# ---------------------------------------------------------------------------
# solve_complex_ivp — nfev
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="probe in _validate_fun_shape not counted toward nfev; see _helpers.py FIXME",
    strict=True,
)
def test_nfev_includes_probe_out_of_place():
    """result.nfev equals total fun calls including the shape-probe (in_place=False)."""
    fun, counter = _make_fun()
    result = solve_complex_ivp(fun, TSPAN, Y0, **TOLS)
    assert result.nfev == counter[0]


def test_nfev_accurate_when_in_place():
    """result.nfev matches fun call count exactly when in_place=True (no probe)."""
    fun, counter = _make_inplace_fun()
    result = solve_complex_ivp(fun, TSPAN, Y0, in_place=True, **TOLS)
    assert result.nfev == counter[0]


def test_nfev_probe_offset_is_one():
    """Out-of-place shape probe adds exactly one uncounted call to the manual counter."""
    fun, counter = _make_fun()
    result = solve_complex_ivp(fun, TSPAN, Y0, **TOLS)
    assert counter[0] == result.nfev + 1


# ---------------------------------------------------------------------------
# solve_complex_ivp — njev
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="probe in _validate_jac_shape not counted toward njev; see _helpers.py FIXME",
    strict=True,
)
def test_njev_includes_probe_out_of_place():
    """result.njev equals total jac calls including the shape-probe (in_place=False)."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    result = solve_complex_ivp(fun, TSPAN, Y0, jac=jac, **TOLS)
    assert result.njev == jac_counter[0]


def test_njev_probe_offset_is_one():
    """Out-of-place Jacobian shape probe adds exactly one uncounted call."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    result = solve_complex_ivp(fun, TSPAN, Y0, jac=jac, **TOLS)
    assert jac_counter[0] == result.njev + 1


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

    assert result_fd.njev == result_analytic.njev, (
        "both paths should assemble the Jacobian the same number of times "
        f"(FD: {result_fd.njev}, analytic: {result_analytic.njev})"
    )
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


@pytest.mark.xfail(
    reason="probe in _validate_fun_shape not counted toward nfev; see _helpers.py FIXME",
    strict=True,
)
def test_zvode_nfev_includes_probe():
    """ZVODE.nfev equals total fun calls including the constructor probe."""
    fun, counter = _make_fun()
    solver = _run_zvode(fun)
    assert solver.nfev == counter[0]


@pytest.mark.xfail(
    reason="probe in _validate_jac_shape not counted toward njev; see _helpers.py FIXME",
    strict=True,
)
def test_zvode_njev_includes_probe():
    """ZVODE.njev equals total jac calls including the constructor probe."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    solver = _run_zvode(fun, jac=jac)
    assert solver.njev == jac_counter[0]


def test_zvode_nfev_probe_offset_is_one():
    """ZVODE constructor probe adds exactly one uncounted fun call."""
    fun, counter = _make_fun()
    solver = _run_zvode(fun)
    assert counter[0] == solver.nfev + 1


def test_zvode_njev_probe_offset_is_one():
    """ZVODE constructor probe adds exactly one uncounted jac call."""
    fun, _ = _make_fun()
    jac, jac_counter = _make_jac()
    solver = _run_zvode(fun, jac=jac)
    assert jac_counter[0] == solver.njev + 1
