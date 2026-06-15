"""Cross-validation of ``solve_complex_ivp`` against ``scipy.integrate.ode``.

Both this package and SciPy's stateful ``ode`` class expose the *same* ZVODE
solver (the algorithm is identical -- SciPy ships its own implementation, a
rewritten C core as of SciPy 1.17, but that is an implementation detail), so
running identical problems through both is a near-free regression guard for our
C-layer integration loops and our option/MITER mapping: if a refactor silently
changes how an option reaches ZVODE (tolerances, method, band widths, the
iteration-method flag, ...), the two trajectories diverge.

The forward comparison covers every

    {coupled-linear, tridiagonal, nonlinear} problem
        x {Adams, BDF} method
            x {no-jac, dense-jac, banded-jac} Jacobian mode

(banded is skipped for the dense 2x2 nonlinear system).  A small curated set of
*backward* (strictly decreasing knots) cases is added on top to confirm ZVODE's
H0 sign handling is mapped identically, without doubling the whole matrix.

All problems are **holomorphic** (each f[i] is analytic in every y[j], a hard
requirement of ZVODE's complex arithmetic): the linear systems are entire and
the nonlinear one is polynomial in ``y``.

Option mapping is aligned so both wrappers select the same ZVODE method flag
``MF = 10*METH + MITER``:

    mode      Adams (METH=1)      BDF (METH=2)
    --------  ------------------  ------------------
    no-jac    MITER=0 (MF=10)     MITER=2 (MF=22)
    dense     MITER=1 (MF=11)     MITER=1 (MF=21)
    banded    MITER=4 (MF=14)     MITER=4 (MF=24)

SciPy picks functional iteration (MITER=0) for the no-jac case unless
``with_jacobian=True``, whereas ``solve_complex_ivp`` defaults BDF to an
internally generated Jacobian (MITER=2); we pass ``with_jacobian=(method ==
'BDF')`` to SciPy so both land on the same MF.

The two are independent implementations with their own wrapper code and build
options, so we do not assert bit-for-bit equality -- only close agreement (in
practice they match to near machine precision).  The tolerance leaves headroom
for benign differences while still catching any real divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from zvode import solve_complex_ivp

from _shared import pack_banded

# scipy.integrate.ode is the reference; skip the whole module if absent.
ode = pytest.importorskip("scipy.integrate").ode

# Solver tolerances: tight, so the comparison reflects integrator behaviour
# rather than discretisation error.
RTOL = 1e-9
ATOL = 1e-12

# Step budget per output point.  solve_complex_ivp defaults to 1e6; SciPy's ode
# defaults to only 500, so we raise it to match.
NSTEPS = 1_000_000

# Agreement tolerance between the two wrappers (~100x looser than the solver
# tolerance): comfortably met, yet a meaningful option-mapping guard.
CMP_RTOL = 1e-7
CMP_ATOL = 1e-9

# Sanity tolerance against the analytic / matrix-exponential reference.
REF_RTOL = 1e-5
REF_ATOL = 1e-7

# Shared output knots: every problem starts at t=0 and reports at KNOTS points.
KNOTS = 9


@dataclass(frozen=True)
class Problem:
    """A holomorphic complex IVP and its callbacks.

    ``jac`` is the dense Jacobian; the banded variant is derived from it via
    ``pack_banded`` using ``band = (lband, uband)``, or skipped when ``band`` is
    ``None``.  ``reference`` returns the exact solution sampled at the knots, or
    ``None`` when no closed form is used.
    """

    name: str
    fun: Callable
    jac: Callable
    band: tuple[int, int] | None
    y0: np.ndarray
    tf: float
    reference: Callable | None

    @property
    def t_eval(self):
        return np.linspace(0.0, self.tf, KNOTS)

    @property
    def variants(self):
        v = ["no_jac", "dense"]
        if self.band is not None:
            v.append("banded")
        return v


# --- P1: coupled 2-component linear system (upper-triangular band) ----------
#
#   dy0/dt = L1*y0 + C*y1
#   dy1/dt =          L2*y1
#
# Analytic: y1 = y0_1*exp(L2 t); y0 = A*exp(L1 t) + B*exp(L2 t),
# with B = C*y0_1/(L2-L1), A = y0_0 - B.

L1, L2, C = -1 + 2j, -2 + 1j, 0.5j
P1_Y0 = np.array([1.0 + 0j, 0.0 + 1j], dtype=np.complex128)
P1_B = C * P1_Y0[1] / (L2 - L1)
P1_A = P1_Y0[0] - P1_B


def p1_fun(t, y):
    return np.array([L1 * y[0] + C * y[1], L2 * y[1]], dtype=np.complex128)


def p1_jac(t, y):
    return np.array([[L1, C], [0.0, L2]], dtype=np.complex128)


def p1_reference(t_eval):
    t = np.asarray(t_eval, dtype=float)
    y0 = P1_A * np.exp(L1 * t) + P1_B * np.exp(L2 * t)
    y1 = P1_Y0[1] * np.exp(L2 * t)
    return np.array([y0, y1])


# --- P2: tridiagonal complex "Schrodinger" system --------------------------
#
#   dy_k/dt = i*(y_{k-1} - 2 y_k + y_{k+1}),  Dirichlet (zero) boundaries.
#
# Linear, holomorphic; reference via the matrix exponential exp(i L t) y0.

N = 8
LMAT = (
    np.diag(np.full(N, -2.0)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
)
A2 = 1j * LMAT
P2_Y0 = (np.linspace(1.0, -1.0, N) + 1j * np.cos(np.arange(N))).astype(np.complex128)


def p2_fun(t, y):
    return A2 @ y


def p2_jac(t, y):
    return A2.copy()


def p2_reference(t_eval):
    from scipy.linalg import expm

    return np.column_stack([expm(A2 * t) @ P2_Y0 for t in t_eval])


# --- P3: nonlinear holomorphic 2-component system --------------------------
#
#   dy0/dt = i*y0 - 0.2*y0*y1
#   dy1/dt = -0.5i*y1 + 0.2*y0^2
#
# Polynomial in y => holomorphic, with a genuinely state-dependent Jacobian.
# No closed form: cross-validated solver-vs-solver only.

P3_Y0 = np.array([0.5 + 0j, 0.3j], dtype=np.complex128)


def p3_fun(t, y):
    return np.array(
        [1j * y[0] - 0.2 * y[0] * y[1], -0.5j * y[1] + 0.2 * y[0] ** 2],
        dtype=np.complex128,
    )


def p3_jac(t, y):
    return np.array(
        [[1j - 0.2 * y[1], -0.2 * y[0]], [0.4 * y[0], -0.5j]],
        dtype=np.complex128,
    )


PROBLEMS = [
    Problem("coupled2", p1_fun, p1_jac, (0, 1), P1_Y0, 2.0, p1_reference),
    Problem("tridiag8", p2_fun, p2_jac, (1, 1), P2_Y0, 1.0, p2_reference),
    Problem("nonlinear2", p3_fun, p3_jac, None, P3_Y0, 2.0, None),
]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def jac_for(prob, variant):
    """Resolve ``(jac_callable, lband, uband)`` for a Jacobian variant."""
    if variant == "dense":
        return prob.jac, None, None
    if variant == "banded":
        lband, uband = prob.band

        def banded(t, y):
            return pack_banded(np.asarray(prob.jac(t, y)), lband, uband)

        return banded, lband, uband
    return None, None, None


def backward(prob):
    """``(t_eval, y0)`` for integrating ``prob`` over strictly decreasing knots.

    Starts from the state at the largest time: the exact reference value when
    one exists, else the problem's own ``y0`` re-anchored at the end time (fine,
    since backward cases are cross-validated solver-vs-solver).
    """
    t_eval = prob.t_eval[::-1].copy()
    if prob.reference is not None:
        y0 = np.ascontiguousarray(prob.reference(prob.t_eval)[:, -1])
    else:
        y0 = prob.y0
    return t_eval, y0


def zvode_trajectory(prob, method, variant, t_eval, y0):
    """Trajectory at ``t_eval`` via ``solve_complex_ivp`` (knot mode)."""
    jac, lband, uband = jac_for(prob, variant)
    sol = solve_complex_ivp(
        prob.fun,
        t_eval,
        y0,
        method=method,
        rtol=RTOL,
        atol=ATOL,
        max_num_steps=NSTEPS,
        jac=jac,
        lband=lband,
        uband=uband,
    )
    assert sol.success
    np.testing.assert_array_equal(sol.t, t_eval)
    return sol.y


def scipy_trajectory(prob, method, variant, t_eval, y0):
    """Trajectory at ``t_eval`` via the stateful ``scipy.integrate.ode``.

    Configured to land on the same ZVODE MF as ``solve_complex_ivp`` for the
    variant, then advanced knot-by-knot to mirror knot-mode output.
    """
    jac, lband, uband = jac_for(prob, variant)
    kw = dict(method=method.lower(), rtol=RTOL, atol=ATOL, nsteps=NSTEPS)
    if variant == "banded":
        kw.update(lband=lband, uband=uband)
    elif variant == "no_jac":
        kw["with_jacobian"] = method == "BDF"  # MITER=2 for BDF, else MITER=0

    r = ode(prob.fun, jac)
    r.set_integrator("zvode", **kw)
    r.set_initial_value(y0, t_eval[0])

    out = np.empty((len(y0), len(t_eval)), dtype=np.complex128)
    out[:, 0] = y0
    for i, t in enumerate(t_eval[1:], start=1):
        r.integrate(t)
        assert r.successful(), f"scipy ode failed (code {r.get_return_code()})"
        out[:, i] = r.y
    return out


def assert_wrappers_agree(prob, method, variant, t_eval, y0):
    """Run both wrappers over ``(t_eval, y0)`` and assert they agree.

    Where a closed form exists, also confirm *both* track the true solution, so
    a bug shared by the two implementations (or a mistake in the problem setup)
    cannot make the test pass silently.
    """
    y_zvode = zvode_trajectory(prob, method, variant, t_eval, y0)
    y_scipy = scipy_trajectory(prob, method, variant, t_eval, y0)

    assert y_zvode.shape == y_scipy.shape == (len(y0), len(t_eval))
    max_diff = np.max(np.abs(y_zvode - y_scipy))
    assert np.allclose(y_zvode, y_scipy, rtol=CMP_RTOL, atol=CMP_ATOL), (
        f"{prob.name}/{method}/{variant}: trajectories diverge, max|d|={max_diff:.3e}"
    )

    if prob.reference is not None:
        ref = prob.reference(t_eval)
        assert np.allclose(y_zvode, ref, rtol=REF_RTOL, atol=REF_ATOL)
        assert np.allclose(y_scipy, ref, rtol=REF_RTOL, atol=REF_ATOL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FORWARD_CASES = [
    pytest.param(prob, method, variant, id=f"{prob.name}-{method}-{variant}")
    for prob in PROBLEMS
    for method in ("Adams", "BDF")
    for variant in prob.variants
]

# One backward case per problem, spanning banded/dense/no-jac and both methods.
BACKWARD_CASES = [
    pytest.param(PROBLEMS[0], "BDF", "banded", id="coupled2-BDF-banded"),
    pytest.param(PROBLEMS[1], "Adams", "dense", id="tridiag8-Adams-dense"),
    pytest.param(PROBLEMS[2], "BDF", "no_jac", id="nonlinear2-BDF-no_jac"),
]


# coupled2's 2x2 band has bandwidth 2 == neq, tripping a benign "verify a banded
# solver is appropriate" warning that is expected and irrelevant here.
@pytest.mark.filterwarnings("ignore:Bandwidth.*exceeds half:UserWarning")
@pytest.mark.parametrize("prob, method, variant", FORWARD_CASES)
def test_matches_scipy_ode(prob, method, variant):
    """solve_complex_ivp and scipy.integrate.ode('zvode') agree to tolerance."""
    assert_wrappers_agree(prob, method, variant, prob.t_eval, prob.y0)


@pytest.mark.filterwarnings("ignore:Bandwidth.*exceeds half:UserWarning")
@pytest.mark.parametrize("prob, method, variant", BACKWARD_CASES)
def test_matches_scipy_ode_backward(prob, method, variant):
    """Backward (decreasing-knot) cross-check: the H0 sign mapping must match."""
    t_eval, y0 = backward(prob)
    assert t_eval[0] > t_eval[-1]
    assert_wrappers_agree(prob, method, variant, t_eval, y0)


def test_banded_layouts_covered():
    """Both a triangular and a symmetric band layout are exercised."""
    bands = [p.band for p in PROBLEMS if p.band is not None]
    assert any(ub and not lb for lb, ub in bands)  # triangular band
    assert any(lb and ub for lb, ub in bands)  # symmetric band
