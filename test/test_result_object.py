"""Tests for the ``ZVODEResult`` design spec (docs/result-object-design.md).

Covers the new verdict fields (``success``, ``status``, ``message``), the
failure-raises-``ZVODEError``-carrying-the-partial-result behaviour, pickle
round-tripping, and duck-type compatibility with ``scipy.integrate.OdeResult``.

The spec states the printout is for humans and *not* API ("never parse it"),
so there are no assertions on the rendered format here — only a smoke test
that ``repr`` does not raise.

``ZVODEResult`` is intentionally *not* part of the public ``zvode`` namespace
(a solve produces one; users never construct it), so it is imported from
``zvode.solve``.  ``ZVODEError`` is public because catching it by name is the
only way to reach ``exc.result``.
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


def test_success_verdict_fields():
    """A successful solve reports success / status / message consistently."""
    sol = _solve_ok()
    assert sol.success is True
    assert sol.status == 0 and isinstance(sol.status, int)
    # `success` is defined as `status >= 0`, not `status == 0`.
    assert sol.success == (sol.status >= 0)
    # message is populated prose; code discriminates on status, never by
    # parsing it, so we only check it is a non-empty string.
    assert isinstance(sol.message, str) and sol.message
    # Each verdict field is reachable both as an attribute and a dict key.
    for key in ("success", "status", "message"):
        assert sol[key] == getattr(sol, key)


# ---------------------------------------------------------------------------
# 2. Failure raises ZVODEError carrying the partial result
# ---------------------------------------------------------------------------


def test_failure_raises_with_partial_result():
    """The failure-path contract, on one raised error:

    - ZVODEError subclasses RuntimeError, so old `except RuntimeError`
      handlers keep working;
    - the exception text equals the result `message` and names the ISTATE
      detail (here ISTATE=-5, repeated convergence failures);
    - the trajectory accumulated up to the failure point is preserved (it used
      to be discarded), aligned, starting at the IC, tracking the exact
      1/(1 - t) solution, with all counters tallied.
    """
    assert issubclass(ZVODEError, RuntimeError)
    exc = _solve_fail()
    assert isinstance(exc, RuntimeError)  # an `except RuntimeError` catches it

    res = exc.result
    assert isinstance(res, ZVODEResult)
    assert res.success is False
    assert res.status == -1
    assert "ISTATE" in res.message and str(exc) == res.message
    assert "convergence" in res.message.lower()

    # More than the initial point, fewer than all requested knots.
    assert 2 <= len(res.t) < len(BLOWUP_KNOTS)
    assert res.y.shape == (1, len(res.t))
    assert res.t[0] == BLOWUP_KNOTS[0]
    np.testing.assert_allclose(res.y[:, 0], BLOWUP_Y0)
    np.testing.assert_allclose(res.y[0], 1.0 / (1.0 - res.t), rtol=1e-5)
    for key in ("nfev", "njev", "nlu", "nsteps", "nni", "ncfn", "netf"):
        assert isinstance(res[key], int)
    assert res.nfev > 0


def test_documented_try_except_workflow():
    """Exercise the recovery pattern promised in the docstring / design spec:

        try:
            sol = solve_complex_ivp(fun, tspan, y0)
        except ZVODEError as exc:
            partial = exc.result   # success=False; plot partial.t, partial.y

    Uses a real try/except (not pytest.raises) so the except branch — the
    thing users actually write — is what runs.
    """
    reached_except = False
    try:
        solve_complex_ivp(blowup_fun, BLOWUP_KNOTS, BLOWUP_Y0, **TOLS)
    except ZVODEError as exc:
        reached_except = True
        partial = exc.result
        assert partial.success is False
        # The partial trajectory is usable downstream (e.g. plotting): the
        # arrays line up and the last recovered state is finite and on-curve.
        assert partial.t.ndim == 1 and partial.t.size >= 2
        assert partial.y.shape == (1, partial.t.size)
        last = partial.y[0, -1]
        assert np.isfinite(last)
        assert abs(last - 1.0 / (1.0 - partial.t[-1])) < 1e-4
    assert reached_except, "solve_complex_ivp should have raised ZVODEError"


# ---------------------------------------------------------------------------
# 3. Pickle round-trips (success and partial result)
# ---------------------------------------------------------------------------


def test_pickle_round_trip():
    for res in (_solve_ok(), _solve_fail().result):
        restored = pickle.loads(pickle.dumps(res))
        assert isinstance(restored, ZVODEResult)
        assert restored.success == res.success
        assert restored.status == res.status
        assert restored.message == res.message
        np.testing.assert_array_equal(restored.t, res.t)
        np.testing.assert_array_equal(restored.y, res.y)
        assert restored.nfev == res.nfev


# ---------------------------------------------------------------------------
# 4. Duck-type compatibility with scipy.integrate.OdeResult
# ---------------------------------------------------------------------------


def test_duck_typed_odescipy_consumer():
    """Code written against OdeResult must work verbatim on a ZVODEResult:
    the canonical `if not res.success: ...` pattern plus the field reads that
    exist without dense output / events."""
    sol = _solve_ok()
    for field in ("t", "y", "success", "status", "message", "nfev", "njev", "nlu"):
        assert field in sol

    def summarise(res):
        if not res.success:
            return f"failed ({res.status}): {res.message}"
        return f"ok n={res.y.shape[0]} nfev={res.nfev} njev={res.njev} nlu={res.nlu}"

    assert summarise(sol).startswith("ok")


# ---------------------------------------------------------------------------
# 5. Pretty-printing: smoke test only (the printout is not API)
# ---------------------------------------------------------------------------


def test_repr_does_not_raise():
    """repr must render every shape (2-D steps, 1-D endpoint, partial) without
    error and without dumping array contents; the exact layout is not API."""
    endpoint = solve_complex_ivp(rot_fun, ROT_TSPAN, ROT_Y0, save_steps=False, **TOLS)
    for res in (_solve_ok(), endpoint, _solve_fail().result):
        text = repr(res)
        assert isinstance(text, str) and text
        assert str(res) == text  # str delegates to repr
