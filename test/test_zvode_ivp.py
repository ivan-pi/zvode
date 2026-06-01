"""Tests for the ZVODE OdeSolver class via scipy.integrate.solve_ivp.

Two analytic examples from the docs folder are used (the QME example is
excluded):

1. Complex exponential decay (docs/demo.py):
       dy/dt = -y,  y(0) = 0.5+1j,  y(t) = (0.5+1j)*exp(-t)

2. Complex oscillator (docs/example.py):
       dw/dt = -i*w^2*z,  dz/dt = i*z
       w(0) = 1/2.1,  z(0) = 1
       Solution:  z(t) = exp(it),  w(t) = 1/(exp(it) + 1.1)

The following miter options are tested:
    miter=1 – BDF + user-supplied dense Jacobian
    miter=2 – BDF + internally-generated dense Jacobian
    miter=3 – BDF + diagonal Jacobian approximation (decay problem only)
    miter=4 – BDF + user-supplied banded Jacobian
    miter=5 – BDF + internally-generated banded Jacobian

miter=3 is exercised only on the decay problem because its Jacobian is
purely diagonal (ml=0, mu=0), so the diagonal approximation is exact and
convergence at the requested tolerance is guaranteed.

Correctness is checked at both the automatically selected output points and
at a predetermined fine grid via dense output (solve_ivp dense_output=True).
"""

import itertools

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
    pd[0, 1] = -1j * w**2
    pd[1, 0] = -2j * w * z
    pd[1, 1] = 1j
    return pd


def sol_oscillator(t):
    """Analytic solution: z(t) = exp(it), w(t) = 1/(exp(it) + 1.1)."""
    z = np.exp(1j * t)
    w = 1.0 / (z + 1.1)
    return np.array([w, z], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Decay problem: miter 1 and 2 (dense / no explicit Jacobian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "miter,jac",
    [
        (1, jac_decay_dense),
        (2, None),
    ],
)
def test_decay_dense_miters(miter, jac):
    """Complex decay solved with dense miter options 1 and 2."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)

    sol = solve_ivp(
        fun_decay, t_span, y0, method=ZVODE, jac=jac, miter=miter, rtol=1e-8, atol=1e-10
    )

    assert sol.success, f"miter={miter}: {sol.message}"
    assert sol.status == 0
    assert sol.t[0] == t_span[0]
    assert sol.t[-1] == t_span[1]
    assert sol.nfev > 0
    assert sol.y.shape == (1, len(sol.t))
    assert sol.y.dtype == np.complex128

    expected = sol_decay(sol.t, y0[0])
    assert_allclose(
        sol.y[0],
        expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"miter={miter}: solution mismatch",
    )


# ---------------------------------------------------------------------------
# Decay problem: miter 4 and 5 (banded Jacobian, ml=0 mu=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "miter,jac",
    [
        (4, jac_decay_banded),
        (5, None),
    ],
)
def test_decay_banded_miters(miter, jac):
    """Complex decay with banded miter options 4 (user Jacobian) and 5 (internal FD)."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)

    sol = solve_ivp(
        fun_decay,
        t_span,
        y0,
        method=ZVODE,
        jac=jac,
        miter=miter,
        lband=0,
        uband=0,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success, f"miter={miter}: {sol.message}"
    assert sol.status == 0
    assert sol.t[-1] == t_span[1]
    assert sol.nfev > 0
    assert sol.y.dtype == np.complex128

    expected = sol_decay(sol.t, y0[0])
    assert_allclose(
        sol.y[0],
        expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"miter={miter}: solution mismatch",
    )


# ---------------------------------------------------------------------------
# Decay problem: miter 3 (diagonal Jacobian approximation)
# ---------------------------------------------------------------------------


def test_decay_diagonal_miter():
    """Complex decay solved with the diagonal Jacobian approximation (miter=3).

    The decay problem has a purely diagonal Jacobian (ml=0, mu=0), so
    ZVODE's diagonal approximation (miter=3) captures the exact Jacobian
    structure.  This makes it a clean regression test for miter=3 that
    we know must converge with high accuracy.
    """
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)

    sol = solve_ivp(fun_decay, t_span, y0, method=ZVODE, miter=3, rtol=1e-8, atol=1e-10)

    assert sol.success, f"miter=3: {sol.message}"
    assert sol.status == 0
    assert sol.t[0] == t_span[0]
    assert sol.t[-1] == t_span[1]
    assert sol.nfev > 0
    assert sol.y.shape == (1, len(sol.t))
    assert sol.y.dtype == np.complex128

    expected = sol_decay(sol.t, y0[0])
    assert_allclose(
        sol.y[0], expected, rtol=1e-5, atol=1e-8, err_msg="miter=3: solution mismatch"
    )


# ---------------------------------------------------------------------------
# Oscillator problem: miter 1, 2, 4, 5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "miter,jac,extra_kwargs",
    [
        (1, jac_oscillator_dense, {}),
        (2, None, {}),
        pytest.param(4, jac_oscillator_banded, {"lband": 0, "uband": 1}),
        pytest.param(5, None, {"lband": 0, "uband": 1}),
    ],
)
def test_oscillator_miter(miter, jac, extra_kwargs):
    """Complex oscillator trajectory checked at solver-selected output points."""
    t0 = 0.0
    t_end = 2 * np.pi
    y0 = np.array([1.0 / 2.1, 1.0], dtype=np.complex128)

    sol = solve_ivp(
        fun_oscillator,
        (t0, t_end),
        y0,
        method=ZVODE,
        jac=jac,
        miter=miter,
        rtol=1e-9,
        atol=1e-9,
        **extra_kwargs,
    )

    assert sol.success, f"miter={miter}: {sol.message}"
    assert sol.status == 0
    assert sol.t[0] == t0
    assert sol.t[-1] == t_end
    assert sol.nfev > 0
    assert sol.y.shape == (2, len(sol.t))
    assert sol.y.dtype == np.complex128

    for i, t in enumerate(sol.t):
        expected = sol_oscillator(t)
        assert_allclose(
            sol.y[:, i],
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"miter={miter}: solution mismatch at t={t:.4f}",
        )


# ---------------------------------------------------------------------------
# Solver counters
# ---------------------------------------------------------------------------


def test_solver_counters_with_jacobian():
    """With a user Jacobian (miter=1), njev should be positive."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol = solve_ivp(
        fun_decay,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        jac=jac_decay_dense,
        miter=1,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success
    assert sol.nfev > 0
    assert sol.njev > 0
    assert sol.nlu > 0


def test_solver_counters_without_jacobian():
    """Without a Jacobian (miter=2), nfev should be positive and njev may be 0."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol = solve_ivp(
        fun_decay, (0.0, 1.0), y0, method=ZVODE, miter=2, rtol=1e-8, atol=1e-10
    )

    assert sol.success
    assert sol.nfev > 0


# ---------------------------------------------------------------------------
# Dense output
# ---------------------------------------------------------------------------


def test_dense_output_decay():
    """sol(t) must match the analytic decay solution at a fine predetermined grid."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_span = (0.0, 2.0)
    t_eval = np.linspace(t_span[0], t_span[1], 50)

    sol = solve_ivp(
        fun_decay,
        t_span,
        y0,
        method=ZVODE,
        miter=2,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success
    assert sol.sol is not None

    y_interp = sol.sol(t_eval)  # shape (1, 50)
    expected = sol_decay(t_eval, y0[0])  # shape (50,)
    assert y_interp.shape == (1, len(t_eval))
    assert_allclose(
        y_interp[0],
        expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Dense output mismatch for decay problem",
    )


def test_dense_output_oscillator():
    """sol(t) must match the analytic oscillator solution at a fine predetermined grid."""
    t0 = 0.0
    t_end = 2 * np.pi
    y0 = np.array([1.0 / 2.1, 1.0], dtype=np.complex128)
    t_eval = np.linspace(t0, t_end, 80)

    sol = solve_ivp(
        fun_oscillator,
        (t0, t_end),
        y0,
        method=ZVODE,
        miter=2,
        dense_output=True,
        rtol=1e-9,
        atol=1e-9,
    )

    assert sol.success
    assert sol.sol is not None

    y_interp = sol.sol(t_eval)  # shape (2, 80)
    assert y_interp.shape == (2, len(t_eval))

    for i, t in enumerate(t_eval):
        expected = sol_oscillator(t)
        assert_allclose(
            y_interp[:, i],
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Dense output mismatch for oscillator at t={t:.4f}",
        )


# ---------------------------------------------------------------------------
# Helpers shared by the linear complex ODE system tests below
# (adapted from scipy/integrate/tests/test_banded_ode_solvers.py)
# ---------------------------------------------------------------------------

# 5x5 real matrix with lband=2, uband=1
_A_REAL = np.array(
    [
        [-0.6, 0.1, 0.0, 0.0, 0.0],
        [0.2, -0.5, 0.9, 0.0, 0.0],
        [0.1, 0.1, -0.4, 0.1, 0.0],
        [0.0, 0.3, -0.1, -0.9, -0.3],
        [0.0, 0.0, 0.1, 0.1, -0.7],
    ]
)

_A_COMPLEX = _A_REAL - 0.5j * _A_REAL  # banded: lband=2, uband=1
_A_COMPLEX_DIAG = np.diag(np.diag(_A_COMPLEX))  # diagonal: lband=0, uband=0


def _linear_exact(a, y0, t_end):
    """Exact solution y(t_end) for dy/dt = a*y via eigendecomposition."""
    lam, v = np.linalg.eig(a)
    c = np.linalg.solve(v, y0)
    return v @ (c * np.exp(lam * t_end))


def _to_zvode_banded(a, ml, mu):
    """Repack dense matrix into ZVODE banded storage: pd[i-j+mu, j] = a[i, j]."""
    n = a.shape[0]
    pd = np.zeros((ml + mu + 1, n), dtype=a.dtype)
    for i in range(n):
        for j in range(max(0, i - ml), min(n, i + mu + 1)):
            pd[i - j + mu, j] = a[i, j]
    return pd


# ---------------------------------------------------------------------------
# t_bound respected (adapted from test_tbound_respected_small_interval)
# ---------------------------------------------------------------------------


def test_tbound_respected():
    """f(t, y) must never be called with t beyond t_bound."""
    t_end = 0.5
    violations = []

    def fun(t, y):
        if t > t_end + 1e-14:
            violations.append(t)
        return -y

    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    sol = solve_ivp(fun, (0.0, t_end), y0, method=ZVODE, rtol=1e-8, atol=1e-10)
    assert sol.success
    assert not violations, f"f evaluated beyond t_bound at: {violations}"


# ---------------------------------------------------------------------------
# Backward integration
# ---------------------------------------------------------------------------


def test_backward_integration():
    """ZVODE correctly integrates backward from t=2 to t=0."""
    y_at_t2 = np.array([(0.5 + 1j) * np.exp(-2.0)], dtype=np.complex128)

    sol = solve_ivp(
        fun_decay, (2.0, 0.0), y_at_t2, method=ZVODE, miter=2, rtol=1e-8, atol=1e-10
    )

    assert sol.success, f"Backward integration failed: {sol.message}"
    assert sol.t[-1] == 0.0
    assert_allclose(sol.y[:, -1], [0.5 + 1j], rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Adams (non-stiff) method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "miter,jac",
    [
        (0, None),  # Adams + functional iteration (no Jacobian)
        (1, jac_decay_dense),  # Adams + user-supplied dense Jacobian
    ],
)
def test_adams_method(miter, jac):
    """Adams method solves the complex decay problem correctly."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)

    sol = solve_ivp(
        fun_decay,
        (0.0, 2.0),
        y0,
        method=ZVODE,
        lmm="Adams",
        jac=jac,
        miter=miter,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success, f"Adams miter={miter} failed: {sol.message}"
    assert_allclose(sol.y[0], sol_decay(sol.t, y0[0]), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Per-component (array) absolute tolerances
# ---------------------------------------------------------------------------


def test_array_atol():
    """Per-component atol array (ITOL=2) is accepted and yields correct results."""
    y0 = np.array([1.0 + 0j, 0.5 + 0.5j], dtype=np.complex128)
    atol_arr = np.array([1e-9, 1e-10])

    sol = solve_ivp(
        fun_decay, (0.0, 1.0), y0, method=ZVODE, miter=2, rtol=1e-7, atol=atol_arr
    )

    assert sol.success
    for i in range(2):
        assert_allclose(sol.y[i], sol_decay(sol.t, y0[i]), rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# t_eval: output at user-specified times
# ---------------------------------------------------------------------------


def test_t_eval():
    """solve_ivp with t_eval delivers results at exactly the requested times."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)
    t_eval = np.linspace(0.0, 2.0, 11)

    sol = solve_ivp(
        fun_decay,
        (0.0, 2.0),
        y0,
        method=ZVODE,
        miter=2,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success
    assert_allclose(sol.t, t_eval)
    assert_allclose(sol.y[0], sol_decay(t_eval, y0[0]), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# max_step and first_step
# ---------------------------------------------------------------------------


def test_max_step():
    """max_step caps internal step sizes; the solution remains correct."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)

    sol_free = solve_ivp(
        fun_decay, (0.0, 1.0), y0, method=ZVODE, miter=2, rtol=1e-8, atol=1e-10
    )

    sol_capped = solve_ivp(
        fun_decay,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        miter=2,
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol_capped.success
    assert sol_capped.nfev >= sol_free.nfev
    assert_allclose(sol_capped.y[0, -1], np.exp(-1.0), rtol=1e-5, atol=1e-8)


def test_first_step():
    """first_step limits the initial step: the first accepted step must not exceed h0."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    h0 = 1e-3

    sol = solve_ivp(
        fun_decay,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        miter=2,
        first_step=h0,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success
    # ZVODE uses H0 as the initial attempt; if rejected on error grounds it halves
    # and retries, so the actual first step satisfies h ≤ h0 but may be smaller.
    first_h = sol.t[1] - sol.t[0]
    assert first_h <= h0 + 1e-14, (
        f"First step {first_h:.3e} exceeds requested first_step {h0:.3e}"
    )
    assert_allclose(sol.y[0, -1], np.exp(-1.0), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# jsv=-1: recompute Jacobian every step
# ---------------------------------------------------------------------------


def test_jsv_negative():
    """jsv=-1 forces Jacobian recomputation each step; result is still correct."""
    y0 = np.array([0.5 + 1j], dtype=np.complex128)

    sol = solve_ivp(
        fun_decay,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        jac=jac_decay_dense,
        miter=1,
        jsv=-1,
        rtol=1e-8,
        atol=1e-10,
    )

    assert sol.success
    # sol.y has shape (n_components, n_timepoints); sol.y[0] is the full
    # trajectory of the single component across all internal steps.
    assert_allclose(sol.y[0], sol_decay(sol.t, y0[0]), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_lmm():
    """An unknown lmm raises ValueError."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="Invalid method"):
        solve_ivp(fun_decay, (0.0, 1.0), y0, method=ZVODE, lmm="Euler")


def test_miter1_without_jac_raises():
    """miter=1 without a jac callable raises ValueError."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError):
        solve_ivp(fun_decay, (0.0, 1.0), y0, method=ZVODE, miter=1)


def test_negative_rtol_raises():
    """Negative rtol raises ValueError."""
    y0 = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError):
        solve_ivp(fun_decay, (0.0, 1.0), y0, method=ZVODE, rtol=-1e-6)


def test_dense_jac_int32_overflow_guard():
    """Dense Jacobian with neq >= 46341 must raise before calling Fortran.

    The Fortran library computes neq² in 32-bit integer arithmetic; 46341² > 2³¹-1
    would silently overflow and corrupt internal array offsets.  The Python
    wrapper must detect this and raise ValueError instead.
    """
    # Use a minimal fake y0 of the offending size; the error fires during __init__
    # before any RHS evaluation, so the rhs function body doesn't matter.
    neq = 46341
    y0 = np.zeros(neq, dtype=np.complex128)

    def _rhs(t, y):
        return np.zeros_like(y)

    # miter=2: BDF with internally-generated dense Jacobian (default for stiff)
    with pytest.raises(ValueError, match="neq"):
        ZVODE(_rhs, 0.0, y0, 1.0)

    # miter=1: user-supplied dense Jacobian
    with pytest.raises(ValueError, match="neq"):
        ZVODE(
            _rhs,
            0.0,
            y0,
            1.0,
            jac=lambda t, y: np.zeros((neq, neq), dtype=np.complex128),
        )


def test_banded_jac_int32_overflow_guard():
    """Banded workspace overflow is also caught before calling Fortran."""
    # Choose ml, mu, neq such that (3*ml + mu + 1)*neq > 2**31 - 1.
    # E.g. ml=10000, mu=0, neq=71530: (30001)*71530 ≈ 2.146e9 < 2^31-1, fine.
    # ml=10000, mu=0, neq=71600: 30001*71600 ≈ 2.149e9 > 2^31-1, overflow.
    ml, mu, neq = 10000, 0, 71600
    y0 = np.zeros(neq, dtype=np.complex128)

    def _rhs(t, y):
        return np.zeros_like(y)

    with pytest.raises(ValueError, match="Banded workspace"):
        ZVODE(_rhs, 0.0, y0, 1.0, lband=ml, uband=mu)


# ---------------------------------------------------------------------------
# Complex linear system tests
# (adapted from scipy/integrate/tests/test_banded_ode_solvers.py)
#
# Both tests solve dy/dt = A*y with a 5x5 complex matrix and check accuracy
# against the analytical solution via eigendecomposition.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Bandwidth")
@pytest.mark.parametrize(
    "lmm,use_jac,banded",
    list(
        itertools.product(
            ["BDF", "Adams"],
            [False, True],
            [False, True],
        )
    ),
)
def test_complex_banded_linear_system(lmm, use_jac, banded):
    """dy/dt = A*y, A complex with lband=2 uband=1; checked against eigendecomposition."""
    a = _A_COMPLEX
    ml, mu = 2, 1
    n = a.shape[0]
    y0 = (np.arange(1, n + 1) + 1j).astype(np.complex128)

    def fun(t, y):
        return a @ y

    if use_jac and banded:
        _pd = _to_zvode_banded(a, ml, mu)

        def jac(t, y):
            return _pd.copy()

        kwargs = {"jac": jac, "miter": 4, "lband": ml, "uband": mu}
    elif use_jac:

        def jac(t, y):
            return a.copy()

        kwargs = {"jac": jac, "miter": 1}
    elif banded:
        kwargs = {"miter": 5, "lband": ml, "uband": mu}
    else:
        kwargs = {"miter": 2}

    sol = solve_ivp(
        fun,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        lmm=lmm,
        rtol=1e-9,
        atol=1e-10,
        **kwargs,
    )

    label = f"lmm={lmm}, use_jac={use_jac}, banded={banded}"
    assert sol.success, f"{label}: {sol.message}"
    assert_allclose(
        sol.y[:, -1], _linear_exact(a, y0, 1.0), rtol=1e-5, atol=1e-7, err_msg=label
    )


@pytest.mark.parametrize(
    "lmm,use_jac",
    list(
        itertools.product(
            ["BDF", "Adams"],
            [False, True],
        )
    ),
)
def test_complex_diagonal_linear_system(lmm, use_jac):
    """dy/dt = A*y, A complex diagonal (lband=0 uband=0); checked against eigendecomposition."""
    a = _A_COMPLEX_DIAG
    n = a.shape[0]
    y0 = (np.arange(1, n + 1) + 1j).astype(np.complex128)

    def fun(t, y):
        return a @ y

    if use_jac:
        _pd = _to_zvode_banded(a, 0, 0)

        def jac(t, y):
            return _pd.copy()

        kwargs = {"jac": jac, "miter": 4, "lband": 0, "uband": 0}
    else:
        kwargs = {"miter": 5, "lband": 0, "uband": 0}

    sol = solve_ivp(
        fun,
        (0.0, 1.0),
        y0,
        method=ZVODE,
        lmm=lmm,
        rtol=1e-9,
        atol=1e-10,
        **kwargs,
    )

    label = f"lmm={lmm}, use_jac={use_jac}"
    assert sol.success, f"{label}: {sol.message}"
    assert_allclose(
        sol.y[:, -1], _linear_exact(a, y0, 1.0), rtol=1e-5, atol=1e-7, err_msg=label
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
