"""Tests for solve_complex_ivp.

Regression and validation tests covering accuracy, error handling,
cross-validation against SciPy, and solver option behaviour:

  1. Damped harmonic oscillator accuracy (Adams, BDF)
  2. Nonlinear complex oscillator accuracy
  3. Error paths: negative atol, short tspan
  4. Cross-validation against scipy.integrate.solve_ivp(method='BDF') on a coupled complex system (n=2)
  5. Single-element (n=1) system: decay, damped oscillation, pure rotation
  6. refine > 1 interpolation accuracy (refine=2, refine=5)
  7. max_order constrains Adams solver order (n=2, complex eigenvalues)
  8. Adaptive step-buffer (StepBuf) growth and structure, with and without
     refinement (exercises the malloc/realloc backing store in drive_adaptive)

Note on real-in-complex problems
---------------------------------
Test 1 uses a real-valued ODE (real coefficients, real initial conditions) run
through a complex-typed solver.  The solution stays on the real axis throughout.
This is NOT the intended use of solve_complex_ivp, which targets genuinely
complex-valued dynamics (quantum systems, complex analytic flows, etc.).
It is included solely as a numerical accuracy regression against a known exact
solution, not as an example of how the solver should be used.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from zvode import solve_complex_ivp


# ---------------------------------------------------------------------------
# 1. Numerical accuracy: underdamped harmonic oscillator
#
#    Real-valued ODE (real coefficients, real IC) — see module note above.
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
# 4. Cross-validation against SciPy BDF
#
#    Coupled complex linear system (upper triangular, n=2):
#      dy0/dt = lam1*y0 + c*y1        lam1 = -1+2j, c = 0.5j
#      dy1/dt = lam2*y1               lam2 = -2+1j
#
#    Both ZVODE and scipy BDF accept complex y0 natively; comparing
#    their endpoints validates the solver against an independent implementation.
# ---------------------------------------------------------------------------

CROSS_LAM1 = -1.0 + 2j
CROSS_LAM2 = -2.0 + 1j
CROSS_C = 0.5j


def coupled_complex_fun(t, y):
    return np.array(
        [CROSS_LAM1 * y[0] + CROSS_C * y[1], CROSS_LAM2 * y[1]], dtype=complex
    )


def test_scipy_bdf_comparison():
    """solve_complex_ivp endpoint agrees with scipy BDF to 1e-5 on a coupled complex system."""
    y0 = np.array([1.0 + 0j, 0.0 + 1j])
    tols = dict(rtol=1e-8, atol=1e-10)

    sol_zvode = solve_complex_ivp(
        coupled_complex_fun, [0.0, 2.0], y0, save_steps=False, **tols
    )
    sol_scipy = solve_ivp(coupled_complex_fun, [0.0, 2.0], y0, method="BDF", **tols)

    # Both result objects expose the same success/message duck type.
    assert sol_zvode.success, f"ZVODE BDF failed: {sol_zvode.message}"
    assert sol_scipy.success, f"SciPy BDF failed: {sol_scipy.message}"
    assert_allclose(sol_zvode.y, sol_scipy.y[:, -1], rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Edge case: single-element (n=1) system
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
# 6. Edge case: refine > 1 interpolation accuracy
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
# 7. Edge case: max_order constraint
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
# 8. Adaptive step-buffer (StepBuf) growth and output structure
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
