"""Cross-validation of ``solve_complex_ivp`` against ``scipy.integrate.ode``.

Both this package and SciPy's stateful ``ode`` class wrap the *same* ZVODE
Fortran core, so running identical problems through both is a near-free
regression guard for our C-layer integration loops and our option/MITER
mapping: if a refactor silently changes how an option is forwarded to ZVODE
(tolerances, method, band widths, the iteration-method flag, ...), the two
trajectories will diverge.

The comparison is run for every combination of

    {linear coupled, tridiagonal, nonlinear} problem
        x {Adams, BDF} method
            x {no-jac, dense-jac, banded-jac} Jacobian mode

(skipping banded for the dense 2x2 nonlinear system).

All problems are **holomorphic** (each f[i] is an analytic function of every
y[j] -- a hard requirement of ZVODE's complex arithmetic): the linear systems
are entire, and the nonlinear one is polynomial in ``y``.

Option mapping is aligned so both wrappers select the same ZVODE method flag
``MF = 10*METH + MITER``:

    mode      Adams (METH=1)      BDF (METH=2)
    --------  ------------------  ------------------
    no-jac    MITER=0 (MF=10)     MITER=2 (MF=22)
    dense     MITER=1 (MF=11)     MITER=1 (MF=21)
    banded    MITER=4 (MF=14)     MITER=4 (MF=24)

For the no-jac case SciPy chooses functional iteration (MITER=0) unless
``with_jacobian=True``, whereas ``solve_complex_ivp`` defaults BDF to an
internally generated Jacobian (MITER=2).  We therefore pass
``with_jacobian=(method == 'BDF')`` to SciPy so both land on the same MF.

With identical MF, tolerances, output knots and (Python) callbacks the two
drivers are in fact bit-for-bit identical today, but -- as requested -- we do
*not* assert exact equality; the tolerance below leaves headroom for benign
floating-point reordering in either driver while still catching any real
divergence in option handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

from zvode import solve_complex_ivp

# scipy.integrate.ode is the reference; skip the whole module if absent.
ode = pytest.importorskip("scipy.integrate").ode

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# Solver tolerances: tight, so both wrappers track the true solution closely
# and the comparison reflects integrator behaviour rather than discretisation.
RTOL = 1e-9
ATOL = 1e-12

# Step budget per output point.  solve_complex_ivp defaults to 1e6; SciPy's
# ode defaults to only 500, so we raise it to match and avoid spurious
# IDID=-1 failures at this tolerance.
NSTEPS = 1_000_000

# Agreement tolerance between the two wrappers.  ~100x looser than the solver
# tolerance: comfortably satisfied (the cores agree to machine precision) yet
# still a meaningful guard against option-mapping regressions.
CMP_RTOL = 1e-7
CMP_ATOL = 1e-9

# Sanity tolerance against the independent analytic / matrix-exponential
# reference, where one is available.
REF_RTOL = 1e-5
REF_ATOL = 1e-7


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """A holomorphic complex IVP and its callbacks.

    ``jac_banded`` returns ZVODE packed storage ``packed[i-j+uband, j] = J[i,j]``
    of shape ``(lband + uband + 1, n)`` -- the convention shared verbatim by
    both ``solve_complex_ivp`` and ``scipy.integrate.ode('zvode')``.
    ``reference`` returns the exact solution sampled at the knots, or ``None``
    when no closed form is used.
    """

    name: str
    fun: Callable
    jac_dense: Callable
    jac_banded: Callable | None
    lband: int | None
    uband: int | None
    y0: np.ndarray
    t_eval: np.ndarray
    reference: Callable | None
    variants: tuple[str, ...]


# --- P1: coupled 2-component linear system (upper-triangular band) ----------
#
#   dy0/dt = L1*y0 + C*y1
#   dy1/dt =          L2*y1
#
# Analytic: y1 = y0_1*exp(L2 t); y0 = A*exp(L1 t) + B*exp(L2 t),
# with B = C*y0_1/(L2-L1), A = y0_0 - B.

_L1, _L2, _C = -1 + 2j, -2 + 1j, 0.5j
_P1_Y0 = np.array([1.0 + 0j, 0.0 + 1j], dtype=np.complex128)
_P1_B = _C * _P1_Y0[1] / (_L2 - _L1)
_P1_A = _P1_Y0[0] - _P1_B


def _p1_fun(t, y):
    dy = np.empty(2, dtype=np.complex128)
    dy[0] = _L1 * y[0] + _C * y[1]
    dy[1] = _L2 * y[1]
    return dy


def _p1_jac_dense(t, y):
    return np.array([[_L1, _C], [0.0, _L2]], dtype=np.complex128)


def _p1_jac_banded(t, y):  # lband=0, uband=1
    pd = np.zeros((2, 2), dtype=np.complex128)
    pd[1, 0] = _L1  # J[0, 0]
    pd[0, 1] = _C   # J[0, 1]
    pd[1, 1] = _L2  # J[1, 1]
    return pd


def _p1_reference(t_eval):
    t = np.asarray(t_eval, dtype=float)
    y0 = _P1_A * np.exp(_L1 * t) + _P1_B * np.exp(_L2 * t)
    y1 = _P1_Y0[1] * np.exp(_L2 * t)
    return np.array([y0, y1])


# --- P2: tridiagonal complex "Schrodinger" system --------------------------
#
#   dy_k/dt = i*(y_{k-1} - 2 y_k + y_{k+1}),  Dirichlet (zero) boundaries.
#
# Linear, holomorphic; reference via the matrix exponential exp(i L t) y0.

_N = 8
_LMAT = (
    np.diag(np.full(_N, -2.0))
    + np.diag(np.ones(_N - 1), 1)
    + np.diag(np.ones(_N - 1), -1)
)
_A2 = 1j * _LMAT
_P2_Y0 = (np.linspace(1.0, -1.0, _N) + 1j * np.cos(np.arange(_N))).astype(
    np.complex128
)


def _p2_fun(t, y):
    return _A2 @ y


def _p2_jac_dense(t, y):
    return _A2.copy()


def _p2_jac_banded(t, y):  # lband=1, uband=1 (tridiagonal)
    pd = np.zeros((3, _N), dtype=np.complex128)
    for j in range(_N):
        pd[1, j] = _A2[j, j]              # main diagonal
        if j + 1 < _N:
            pd[0, j + 1] = _A2[j, j + 1]  # super-diagonal
        if j - 1 >= 0:
            pd[2, j - 1] = _A2[j, j - 1]  # sub-diagonal
    return pd


def _p2_reference(t_eval):
    from scipy.linalg import expm

    return np.column_stack([expm(_A2 * t) @ _P2_Y0 for t in t_eval])


# --- P3: nonlinear holomorphic 2-component system --------------------------
#
#   dy0/dt = i*y0 - 0.2*y0*y1
#   dy1/dt = -0.5i*y1 + 0.2*y0^2
#
# Polynomial in y => holomorphic, with a genuinely state-dependent Jacobian.
# No closed form: cross-validated solver-vs-solver only.

_P3_Y0 = np.array([0.5 + 0j, 0.3j], dtype=np.complex128)


def _p3_fun(t, y):
    dy = np.empty(2, dtype=np.complex128)
    dy[0] = 1j * y[0] - 0.2 * y[0] * y[1]
    dy[1] = -0.5j * y[1] + 0.2 * y[0] ** 2
    return dy


def _p3_jac_dense(t, y):
    return np.array(
        [[1j - 0.2 * y[1], -0.2 * y[0]], [0.4 * y[0], -0.5j]],
        dtype=np.complex128,
    )


PROBLEMS = [
    Problem(
        name="coupled2",
        fun=_p1_fun,
        jac_dense=_p1_jac_dense,
        jac_banded=_p1_jac_banded,
        lband=0,
        uband=1,
        y0=_P1_Y0,
        t_eval=np.linspace(0.0, 2.0, 9),
        reference=_p1_reference,
        variants=("no_jac", "dense", "banded"),
    ),
    Problem(
        name="tridiag8",
        fun=_p2_fun,
        jac_dense=_p2_jac_dense,
        jac_banded=_p2_jac_banded,
        lband=1,
        uband=1,
        y0=_P2_Y0,
        t_eval=np.linspace(0.0, 1.0, 9),
        reference=_p2_reference,
        variants=("no_jac", "dense", "banded"),
    ),
    Problem(
        name="nonlinear2",
        fun=_p3_fun,
        jac_dense=_p3_jac_dense,
        jac_banded=None,
        lband=None,
        uband=None,
        y0=_P3_Y0,
        t_eval=np.linspace(0.0, 2.0, 9),
        reference=None,
        variants=("no_jac", "dense"),
    ),
]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _zvode_trajectory(prob: Problem, method: str, variant: str) -> np.ndarray:
    """Trajectory at ``prob.t_eval`` via ``solve_complex_ivp`` (knot mode)."""
    kwargs = dict(method=method, rtol=RTOL, atol=ATOL, max_num_steps=NSTEPS)
    if variant == "dense":
        kwargs["jac"] = prob.jac_dense
    elif variant == "banded":
        kwargs["jac"] = prob.jac_banded
        kwargs["lband"] = prob.lband
        kwargs["uband"] = prob.uband
    sol = solve_complex_ivp(prob.fun, prob.t_eval, prob.y0, **kwargs)
    assert sol.success
    np.testing.assert_array_equal(sol.t, prob.t_eval)
    return sol.y


def _scipy_trajectory(prob: Problem, method: str, variant: str) -> np.ndarray:
    """Trajectory at ``prob.t_eval`` via the stateful ``scipy.integrate.ode``.

    The integrator is configured to land on the same ZVODE method flag as
    ``solve_complex_ivp`` for the corresponding ``variant`` (see module
    docstring), then advanced knot-by-knot to mirror knot-mode output.
    """
    integrator_kw = dict(
        method=method.lower(), rtol=RTOL, atol=ATOL, nsteps=NSTEPS
    )
    jac = None
    if variant == "dense":
        jac = prob.jac_dense
    elif variant == "banded":
        jac = prob.jac_banded
        integrator_kw["lband"] = prob.lband
        integrator_kw["uband"] = prob.uband
    else:  # no_jac: match MITER (0 for Adams, 2 for BDF)
        integrator_kw["with_jacobian"] = method == "BDF"

    r = ode(prob.fun, jac)
    r.set_integrator("zvode", **integrator_kw)
    r.set_initial_value(prob.y0, prob.t_eval[0])

    out = np.empty((len(prob.y0), len(prob.t_eval)), dtype=np.complex128)
    out[:, 0] = prob.y0
    for i, t in enumerate(prob.t_eval[1:], start=1):
        r.integrate(t)
        assert r.successful(), f"scipy ode failed (code {r.get_return_code()})"
        out[:, i] = r.y
    return out


def _cases():
    for prob in PROBLEMS:
        for method in ("Adams", "BDF"):
            for variant in prob.variants:
                yield pytest.param(
                    prob, method, variant, id=f"{prob.name}-{method}-{variant}"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# The 2x2 banded problem (coupled2) has bandwidth 2 == neq, which trips a
# benign "verify a banded solver is appropriate" UserWarning; it is expected
# here and irrelevant to the cross-validation.
@pytest.mark.filterwarnings("ignore:Bandwidth.*exceeds half:UserWarning")
@pytest.mark.parametrize("prob, method, variant", list(_cases()))
def test_matches_scipy_ode(prob: Problem, method: str, variant: str):
    """solve_complex_ivp and scipy.integrate.ode('zvode') agree to tolerance."""
    y_zvode = _zvode_trajectory(prob, method, variant)
    y_scipy = _scipy_trajectory(prob, method, variant)

    assert y_zvode.shape == y_scipy.shape == (len(prob.y0), len(prob.t_eval))
    max_diff = np.max(np.abs(y_zvode - y_scipy))
    assert np.allclose(y_zvode, y_scipy, rtol=CMP_RTOL, atol=CMP_ATOL), (
        f"{prob.name}/{method}/{variant}: trajectories diverge, "
        f"max|Δy|={max_diff:.3e}"
    )

    # Where a closed form exists, confirm *both* track the true solution, so a
    # shared bug in the common Fortran core cannot make the test pass silently.
    if prob.reference is not None:
        ref = prob.reference(prob.t_eval)
        assert np.allclose(y_zvode, ref, rtol=REF_RTOL, atol=REF_ATOL)
        assert np.allclose(y_scipy, ref, rtol=REF_RTOL, atol=REF_ATOL)


def test_problem_matrix_is_exhaustive():
    """Guard the parametrisation: jac x no-jac and dense x banded are covered."""
    variants = {v for prob in PROBLEMS for v in prob.variants}
    assert {"no_jac", "dense", "banded"} <= variants
    # At least one problem exercises each banded half-bandwidth layout.
    banded = [p for p in PROBLEMS if "banded" in p.variants]
    assert any(p.uband and not p.lband for p in banded)  # triangular band
    assert any(p.lband and p.uband for p in banded)       # symmetric band
