"""Tests for the procedural solve_complex_ivp interface.

System under test: 2-component coupled complex ODE

    dy[0]/dt = LAM1*y[0] + C*y[1]
    dy[1]/dt = LAM2*y[1]

with analytic solution

    y[1](t) = y0[1] * exp(LAM2*t)
    y[0](t) = A*exp(LAM1*t) + B*exp(LAM2*t)

where B = C*y0[1]/(LAM2 - LAM1), A = y0[0] - B.
"""
import numpy as np
import pytest

from zvode import solve_complex_ivp
from zvode.solve import ZVODEStats  # not re-exported from __all__; internal use only

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

LAM1 = -1 + 2j
LAM2 = -2 + 1j
C = 0.5j
Y0 = np.array([1.0 + 0j, 0.0 + 1j])
T0 = 0.0
TF = 2.0

_B = C * Y0[1] / (LAM2 - LAM1)
_A = Y0[0] - _B

LBAND = 0
UBAND = 1   # Jacobian is upper triangular: J[1,0]=0

# Integration tolerances tight enough for 1e-5 solution accuracy
RTOL = 1e-8
ATOL = 1e-10


def exact(t):
    """Analytic solution at time(s) t, returned as shape (2, ...) array."""
    t = np.asarray(t, dtype=float)
    y1 = _A * np.exp(LAM1 * t) + _B * np.exp(LAM2 * t)
    y2 = Y0[1] * np.exp(LAM2 * t)
    return np.array([y1, y2])


def _check(t_arr, y_arr, sol_rtol=1e-5):
    ref = exact(t_arr)
    assert np.allclose(y_arr, ref, rtol=sol_rtol), (
        f"max err={np.max(np.abs(y_arr - ref)):.2e}")


# ---------------------------------------------------------------------------
# Callback definitions
# ---------------------------------------------------------------------------

# Path B: in-place, in_place=True  (defined first; scipy-style delegates below)

def fun_ip(t, y, dy):
    dy[0] = LAM1 * y[0] + C * y[1]
    dy[1] = LAM2 * y[1]


def jac_dense_ip(t, y, pd):
    pd[0, 0] = LAM1
    pd[0, 1] = C
    pd[1, 1] = LAM2


def jac_banded_ip(t, y, pd, ml, mu):
    # storage: pd[mu + i - j, j] = J[i, j]
    pd[mu, 0] = LAM1       # J[0, 0]
    pd[mu - 1, 1] = C      # J[0, 1]
    pd[mu, 1] = LAM2       # J[1, 1]


# Path A: SciPy-style, in_place=False — delegate to the in-place versions above

def fun(t, y):
    dy = np.empty(len(y), dtype=np.complex128)
    fun_ip(t, y, dy)
    return dy


def jac_dense(t, y):
    pd = np.zeros((len(y), len(y)), dtype=np.complex128)
    jac_dense_ip(t, y, pd)
    return pd


def jac_banded(t, y):
    # ZVODE banded storage: pd[mu + i - j, j] = J[i, j]
    # shape = (lband + uband + 1, n) = (2, 2)
    pd = np.zeros((LBAND + UBAND + 1, len(y)), dtype=np.complex128)
    jac_banded_ip(t, y, pd, LBAND, UBAND)
    return pd


# ---------------------------------------------------------------------------
# 1. Output modes × methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_save_steps_true(method):
    """Default mode: collect all accepted steps."""
    t, y = solve_complex_ivp(fun, [T0, TF], Y0, method=method,
                             rtol=RTOL, atol=ATOL)
    assert isinstance(t, np.ndarray)
    assert t.ndim == 1 and t[0] == T0 and t[-1] == TF   # endpoints must be exact
    assert y.ndim == 2 and y.shape == (2, len(t))
    _check(t, y)


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_save_steps_false(method):
    """Endpoint-only mode: scalar t and 1-D y."""
    t, y = solve_complex_ivp(fun, [T0, TF], Y0, method=method,
                             rtol=RTOL, atol=ATOL, save_steps=False)
    assert np.isscalar(t)
    assert t == pytest.approx(TF)
    assert y.ndim == 1 and y.shape == (2,)
    ref = exact(TF)  # shape (2,)
    assert np.allclose(y, ref, rtol=1e-5)


@pytest.mark.parametrize("method", ["Adams", "BDF"])
def test_knots_mode(method):
    """Knot mode (len(tspan) > 2): output at exactly the requested times."""
    tspan = np.linspace(T0, TF, 11)
    t, y = solve_complex_ivp(fun, tspan, Y0, method=method, rtol=RTOL, atol=ATOL)
    assert isinstance(t, np.ndarray)
    np.testing.assert_array_equal(t, tspan)
    assert y.shape == (2, 11)
    assert y.dtype == np.complex128
    _check(t, y)


# ---------------------------------------------------------------------------
# 2. Jacobian types × methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["Adams", "BDF"])
@pytest.mark.parametrize("jac_fn,jac_kwargs", [
    pytest.param(None, {}, id="no_jac"),
    pytest.param(jac_dense, {}, id="dense_jac"),
    pytest.param(jac_banded, {"lband": LBAND, "uband": UBAND}, id="banded_jac"),
])
def test_jacobian_types(method, jac_fn, jac_kwargs):
    """Dense and banded user Jacobians against the no-Jacobian baseline."""
    t, y = solve_complex_ivp(fun, [T0, TF], Y0, method=method,
                             rtol=RTOL, atol=ATOL, jac=jac_fn, **jac_kwargs)
    _check(t, y)


# ---------------------------------------------------------------------------
# 3. Callback convention (in_place=False vs in_place=True) × output modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_inplace_fun(mode):
    """in_place=True plain Python callable, no Jacobian."""
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = (mode == "steps")
    t, y = solve_complex_ivp(fun_ip, tspan, Y0, in_place=True,
                             save_steps=save, rtol=RTOL, atol=ATOL)
    if mode == "endpoint":
        ref = exact(TF)
        assert np.allclose(y, ref, rtol=1e-5)
    else:
        _check(t, y)


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_inplace_fun_dense_jac(mode):
    """in_place=True with dense Jacobian."""
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = (mode == "steps")
    t, y = solve_complex_ivp(fun_ip, tspan, Y0, in_place=True, jac=jac_dense_ip,
                             save_steps=save, rtol=RTOL, atol=ATOL)
    if mode == "endpoint":
        ref = exact(TF)
        assert np.allclose(y, ref, rtol=1e-5)
    else:
        _check(t, y)


@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_inplace_fun_banded_jac(mode):
    """in_place=True with banded Jacobian."""
    tspan = np.linspace(T0, TF, 9) if mode == "knots" else [T0, TF]
    save = (mode == "steps")
    t, y = solve_complex_ivp(fun_ip, tspan, Y0, in_place=True, jac=jac_banded_ip,
                             lband=LBAND, uband=UBAND, save_steps=save,
                             rtol=RTOL, atol=ATOL)
    if mode == "endpoint":
        ref = exact(TF)
        assert np.allclose(y, ref, rtol=1e-5)
    else:
        _check(t, y)


# ---------------------------------------------------------------------------
# 4. Backward integration × output modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["steps", "endpoint", "knots"])
def test_backward_integration(mode):
    """tspan strictly decreasing: integrate from TF back to T0."""
    y_tf = exact(TF)  # shape (2,); initial condition at t=TF
    if mode == "knots":
        tspan = np.linspace(TF, T0, 11)
    else:
        tspan = [TF, T0]
    save = (mode == "steps")

    t, y = solve_complex_ivp(fun, tspan, y_tf, save_steps=save, rtol=RTOL, atol=ATOL)

    if mode == "endpoint":
        ref = exact(T0)
        assert t == pytest.approx(T0)
        assert np.allclose(y, ref, rtol=1e-5)
    else:
        _check(t, y)


# ---------------------------------------------------------------------------
# 5. allow_overshoot
# ---------------------------------------------------------------------------

def test_allow_overshoot_false():
    """allow_overshoot=False (default): last output point must equal TF exactly."""
    t, y = solve_complex_ivp(fun, [T0, TF], Y0, allow_overshoot=False,
                             rtol=RTOL, atol=ATOL)
    assert t[-1] == pytest.approx(TF)


def test_allow_overshoot_true():
    """allow_overshoot=True: last output point may go slightly past TF."""
    t, y = solve_complex_ivp(fun, [T0, TF], Y0, allow_overshoot=True,
                             rtol=RTOL, atol=ATOL)
    assert t[-1] >= TF - 1e-12
    # Solution at TF should still be accurate regardless of overshoot
    # Find the closest output point to TF and verify the analytic match
    idx = np.argmin(np.abs(t - TF))
    _check(t[idx:idx+1], y[:, idx:idx+1])


# ---------------------------------------------------------------------------
# 6. max_num_steps exceeded → RuntimeError
# ---------------------------------------------------------------------------

def test_max_num_steps_exceeded():
    """Solver raises RuntimeError when max_num_steps is too small.

    save_steps=False asks the solver to reach TF in one shot, requiring far
    more than 2 internal steps, so the step limit is hit and RuntimeError is
    raised.  With save_steps=True the solver advances one step per call, so
    the per-output-point limit would never be reached with max_num_steps=2.
    """
    with pytest.raises(RuntimeError, match="ISTATE"):
        solve_complex_ivp(fun, [T0, TF], Y0, save_steps=False, max_num_steps=2)


# ---------------------------------------------------------------------------
# 7. ret_stats returns ZVODEStats
# ---------------------------------------------------------------------------

def test_ret_stats():
    """ret_stats=True appends a ZVODEStats object."""
    result = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL,
                               ret_stats=True)
    assert len(result) == 3
    t, y, stats = result
    assert isinstance(stats, ZVODEStats)
    assert stats.nsteps > 0
    assert stats.nfev > 0


# ---------------------------------------------------------------------------
# 8. refine > 1 (denser output via ZVINDY interpolation)
# ---------------------------------------------------------------------------

def test_refine():
    """refine=4 inserts 3 interpolated points per step; solution should match."""
    REFINE = 4
    t_base, _ = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL,
                                   refine=1)
    t_ref, y_ref = solve_complex_ivp(fun, [T0, TF], Y0, rtol=RTOL, atol=ATOL,
                                     refine=REFINE)
    # Each of the (n-1) inter-step intervals gains (refine-1) extra points.
    n_steps = len(t_base) - 1
    assert len(t_ref) == len(t_base) + n_steps * (REFINE - 1)
    _check(t_ref, y_ref)


# ---------------------------------------------------------------------------
# 9. Argument validation
# ---------------------------------------------------------------------------

def test_non_monotonic_tspan_raises():
    """Non-monotonic tspan must raise ValueError."""
    with pytest.raises(ValueError, match="monotonic"):
        solve_complex_ivp(fun, [0.0, 1.0, 0.5], Y0)


def test_non_monotonic_mixed_tspan_raises():
    """tspan with mixed sign differences must raise ValueError."""
    with pytest.raises(ValueError, match="monotonic"):
        solve_complex_ivp(fun, [0.0, 2.0, 1.0, 3.0], Y0)


def test_real_y0_warns():
    """Real y0 triggers a UserWarning (not a hard error)."""
    with pytest.warns(UserWarning, match="complex"):
        solve_complex_ivp(fun, [T0, TF], np.array([1.0, 0.0]))


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        solve_complex_ivp(fun, [T0, TF], Y0, method="RK4")


def test_invalid_refine_raises():
    with pytest.raises(ValueError, match="refine"):
        solve_complex_ivp(fun, [T0, TF], Y0, refine=0)


def test_compiled_callback_requires_in_place():
    """Compiled callbacks (numba/ctypes) are incompatible with in_place=False."""
    import ctypes

    # A minimal ctypes function pointer — address detection is enough to
    # trigger the check; the function is never actually called.
    prototype = ctypes.CFUNCTYPE(None)
    dummy = prototype(lambda: None)

    with pytest.raises(ValueError, match="in_place"):
        solve_complex_ivp(dummy, [T0, TF], Y0, in_place=False)
