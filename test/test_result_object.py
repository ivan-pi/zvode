"""Tests for the ``ZVODEResult`` design spec (docs/result-object-design.md).

Covers the new verdict fields (``success``, ``status``, ``message``), the
failure-raises-``ZVODEError``-carrying-the-partial-result behaviour, pickle
round-tripping, duck-type compatibility with ``scipy.integrate.OdeResult``,
and the pretty-printing rules.

``ZVODEResult`` is intentionally *not* part of the public ``zvode`` namespace
(a solve produces one; users never construct it), so it is imported from
``zvode.solve`` here.  ``ZVODEError`` is public because catching it by name is
the only way to reach ``exc.result``.
"""

import pickle

import numpy as np
import pytest

from zvode import solve_complex_ivp
from zvode.solve import ZVODEResult, ZVODEError

# ---------------------------------------------------------------------------
# Shared problems
# ---------------------------------------------------------------------------

# Smooth rotation: dy/dt = i*y, y(0) = 1, exact y(t) = exp(i*t).
def rot_fun(t, y):
    return 1j * y


ROT_Y0 = np.array([1.0 + 0j])
ROT_TSPAN = [0.0, 2.0 * np.pi]


# Blow-up: dy/dt = y**2, y(0) = 1, exact y = 1/(1 - t), singular at t = 1.
# Knots straddle the singularity so the solve completes a few knots and then
# fails — giving a partial trajectory longer than just the initial point.
def blowup_fun(t, y):
    # Squaring overflows as the solver probes past the t=1 singularity; the
    # overflow is expected and harmless (the step is rejected), so silence it.
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array([y[0] ** 2], dtype=np.complex128)


BLOWUP_Y0 = np.array([1.0 + 0j])
BLOWUP_KNOTS = np.linspace(0.0, 2.0, 9)

TOLS = dict(rtol=1e-8, atol=1e-10)

SUCCESS_MESSAGE = "The solver successfully reached the end of the integration interval."


def _solve_ok():
    return solve_complex_ivp(rot_fun, ROT_TSPAN, ROT_Y0, **TOLS)


def _solve_fail():
    """Run the blow-up problem and return the raised ZVODEError."""
    with pytest.raises(ZVODEError) as excinfo:
        solve_complex_ivp(blowup_fun, BLOWUP_KNOTS, BLOWUP_Y0, **TOLS)
    return excinfo.value


# ---------------------------------------------------------------------------
# 1. Verdict fields on success
# ---------------------------------------------------------------------------


def test_success_true_on_success():
    sol = _solve_ok()
    assert sol.success is True


def test_status_zero_on_success():
    sol = _solve_ok()
    assert sol.status == 0
    assert isinstance(sol.status, int)


def test_message_is_generic_success_text():
    sol = _solve_ok()
    assert sol.message == SUCCESS_MESSAGE


def test_success_is_status_nonnegative():
    """`success` is defined as `status >= 0`, not `status == 0`."""
    sol = _solve_ok()
    assert sol.success == (sol.status >= 0)


def test_message_is_str_on_success():
    sol = _solve_ok()
    assert isinstance(sol.message, str)


def test_verdict_fields_available_via_dict_and_attr():
    sol = _solve_ok()
    for key in ("success", "status", "message"):
        assert sol[key] == getattr(sol, key)


# ---------------------------------------------------------------------------
# 2. Failure raises ZVODEError carrying the partial result
# ---------------------------------------------------------------------------


def test_failure_raises_zvode_error():
    with pytest.raises(ZVODEError):
        solve_complex_ivp(blowup_fun, BLOWUP_KNOTS, BLOWUP_Y0, **TOLS)


def test_zvode_error_is_runtime_error():
    """ZVODEError subclasses RuntimeError so old `except RuntimeError` works."""
    assert issubclass(ZVODEError, RuntimeError)
    exc = _solve_fail()
    assert isinstance(exc, RuntimeError)


def test_failure_still_caught_by_runtime_error_handler():
    """The existing contract: a bare `except RuntimeError` keeps working."""
    with pytest.raises(RuntimeError, match="ISTATE"):
        solve_complex_ivp(blowup_fun, BLOWUP_KNOTS, BLOWUP_Y0, **TOLS)


def test_exception_carries_result():
    exc = _solve_fail()
    assert isinstance(exc.result, ZVODEResult)


def test_failure_result_success_false():
    exc = _solve_fail()
    assert exc.result.success is False


def test_failure_result_status_negative():
    exc = _solve_fail()
    assert exc.result.status == -1


def test_failure_result_message_contains_istate():
    exc = _solve_fail()
    assert "ISTATE" in exc.result.message


def test_exception_text_equals_result_message():
    """The exception text is the same string as the result's message."""
    exc = _solve_fail()
    assert str(exc) == exc.result.message


def test_failure_message_contains_remedy_detail():
    """The blow-up triggers repeated convergence failures (ISTATE=-5);
    the message should carry the human-readable detail, not just a code."""
    exc = _solve_fail()
    assert "convergence" in exc.result.message.lower()


# ---------------------------------------------------------------------------
# 3. Partial trajectory accumulated up to the failure point
# ---------------------------------------------------------------------------


def test_partial_trajectory_present():
    exc = _solve_fail()
    res = exc.result
    # The blow-up completes a few knots before failing at the singularity:
    # more than just the initial point, but fewer than all requested knots.
    assert 2 <= len(res.t) < len(BLOWUP_KNOTS)


def test_partial_trajectory_shapes_consistent():
    exc = _solve_fail()
    res = exc.result
    assert res.y.shape == (1, len(res.t))


def test_partial_trajectory_starts_at_initial_condition():
    exc = _solve_fail()
    res = exc.result
    assert res.t[0] == BLOWUP_KNOTS[0]
    np.testing.assert_allclose(res.y[:, 0], BLOWUP_Y0)


def test_partial_trajectory_matches_exact_solution():
    """Up to the failure point the completed knots track 1/(1 - t)."""
    exc = _solve_fail()
    res = exc.result
    exact = 1.0 / (1.0 - res.t)
    np.testing.assert_allclose(res.y[0], exact, rtol=1e-5)


def test_partial_result_has_counters():
    exc = _solve_fail()
    res = exc.result
    for key in ("nfev", "njev", "nlu", "nsteps", "nni", "ncfn", "netf"):
        assert isinstance(res[key], int)
    assert res.nfev > 0


# ---------------------------------------------------------------------------
# 4. Pickle round-trips
# ---------------------------------------------------------------------------


def test_pickle_round_trip_success():
    sol = _solve_ok()
    restored = pickle.loads(pickle.dumps(sol))
    assert isinstance(restored, ZVODEResult)
    assert restored.success == sol.success
    assert restored.status == sol.status
    assert restored.message == sol.message
    np.testing.assert_array_equal(restored.t, sol.t)
    np.testing.assert_array_equal(restored.y, sol.y)
    assert restored.nfev == sol.nfev


def test_pickle_round_trip_partial_result():
    res = _solve_fail().result
    restored = pickle.loads(pickle.dumps(res))
    assert restored.success is False
    assert restored.status == -1
    np.testing.assert_array_equal(restored.t, res.t)
    np.testing.assert_array_equal(restored.y, res.y)


# ---------------------------------------------------------------------------
# 5. Duck-type compatibility with scipy.integrate.OdeResult
# ---------------------------------------------------------------------------


def test_duck_typed_odescipy_consumer():
    """Code written against OdeResult must work verbatim on a ZVODEResult."""

    def summarise(res):
        # The canonical `if not res.success: ...` pattern plus field reads.
        if not res.success:
            return f"failed ({res.status}): {res.message}"
        return (
            f"ok t0={res.t[0]} tf={res.t[-1]} n={res.y.shape[0]} "
            f"nfev={res.nfev} njev={res.njev} nlu={res.nlu} status={res.status}"
        )

    out = summarise(_solve_ok())
    assert out.startswith("ok")


def test_strict_superset_of_odescipy_fields():
    """Every OdeResult field that exists without dense output / events
    is present (status, message, success, t, y, nfev, njev, nlu)."""
    sol = _solve_ok()
    for field in ("t", "y", "success", "status", "message", "nfev", "njev", "nlu"):
        assert field in sol


# ---------------------------------------------------------------------------
# 6. Pretty-printing
# ---------------------------------------------------------------------------


def test_repr_equals_str():
    sol = _solve_ok()
    assert repr(sol) == str(sol)


def test_repr_does_not_dump_arrays():
    """Arrays render as `[shape dtype]` placeholders, never the contents."""
    sol = _solve_ok()
    text = repr(sol)
    # The complex128 state array would contain 'j'-bearing numbers if dumped;
    # the placeholder names the dtype instead.
    assert "complex128" in text
    assert f"{sol.y.shape[0]}x{sol.y.shape[1]} complex128" in text
    # Heuristic: a dumped (1, 33) array would be far more than one line.
    assert text.count("\n") < 20


def test_repr_verdict_block_first():
    """message / success / status lead, before t / y and the counters."""
    sol = _solve_ok()
    lines = [ln.strip() for ln in repr(sol).splitlines()]
    keys = [ln.split(":")[0] for ln in lines]
    assert keys[:3] == ["message", "success", "status"]
    assert keys.index("t") < keys.index("nfev")
    assert keys.index("y") < keys.index("nfev")


def test_repr_shows_t_interval():
    """The t line carries a `first to last` interval suffix."""
    sol = _solve_ok()
    t_line = next(ln for ln in repr(sol).splitlines() if ln.strip().startswith("t:"))
    assert "to" in t_line
    assert "float64" in t_line


def test_repr_endpoint_mode_scalar_t():
    """In endpoint-only mode t is a scalar and renders with repr (no suffix)."""
    sol = solve_complex_ivp(rot_fun, ROT_TSPAN, ROT_Y0, save_steps=False, **TOLS)
    text = repr(sol)
    assert "success" in text and "status" in text
    # y is 1-D here -> placeholder names the single dimension.
    assert f"{sol.y.shape[0]} complex128" in text
