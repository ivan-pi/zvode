"""Python-level tests for pybdf, cross-validated against scipy.integrate."""

import numpy as np
import pytest

from pybdf import BDF, solve_bdf

scipy_integrate = pytest.importorskip("scipy.integrate")
solve_ivp = scipy_integrate.solve_ivp


def rober(t, y):
    return [-0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2]


def rober_jac(t, y):
    return np.array([[-0.04, 1e4 * y[2], 1e4 * y[1]],
                     [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
                     [0.0, 6e7 * y[1], 0.0]])


ROBER_ATOL = [1e-8, 1e-10, 1e-8]


def test_scalar_decay_analytic():
    r = solve_bdf(lambda t, y: -y, (0.0, 5.0), [1.0],
                  t_eval=[5.0], rtol=1e-8, atol=1e-10)
    assert r.success
    assert abs(r.y[0, -1] - np.exp(-5.0)) < 1e-6


def test_rober_matches_scipy_user_jac():
    r = solve_bdf(rober, (0, 40), [1, 0, 0], t_eval=[40],
                  rtol=1e-6, atol=ROBER_ATOL, jac=rober_jac)
    sp = solve_ivp(rober, (0, 40), [1, 0, 0], method="BDF", t_eval=[40],
                   rtol=1e-6, atol=ROBER_ATOL, jac=rober_jac)
    assert r.success
    assert np.max(np.abs(r.y[:, -1] - sp.y[:, -1])) < 1e-6


def test_rober_finite_difference_matches_scipy():
    r = solve_bdf(rober, (0, 40), [1, 0, 0], t_eval=[40],
                  rtol=1e-6, atol=ROBER_ATOL)
    sp = solve_ivp(rober, (0, 40), [1, 0, 0], method="BDF", t_eval=[40],
                   rtol=1e-6, atol=ROBER_ATOL)
    assert np.max(np.abs(r.y[:, -1] - sp.y[:, -1])) < 1e-5


def test_jacobian_is_reused():
    # Many steps but only a handful of Jacobian evaluations.
    r = solve_bdf(rober, (0, 40), [1, 0, 0], t_eval=[40],
                  rtol=1e-6, atol=ROBER_ATOL, jac=rober_jac)
    assert r.njev < r.nsteps
    assert r.njev <= 10


def _heat_rhs(t, y):
    n = y.size
    f = -2.0 * y
    f[1:] += y[:-1]
    f[:-1] += y[1:]
    return f


def _heat_band_jac(t, y):
    n = y.size
    ab = np.zeros((3, n))
    ab[0, 1:] = 1.0     # superdiagonal
    ab[1, :] = -2.0     # diagonal
    ab[2, :-1] = 1.0    # subdiagonal
    return ab


def test_banded_matches_dense():
    n = 25
    y0 = np.sin(np.arange(1, n + 1, dtype=float))
    common = dict(rtol=1e-8, atol=1e-10, t_eval=[1.0])

    dense = solve_bdf(_heat_rhs, (0, 1), y0, **common)
    band_fd = solve_bdf(_heat_rhs, (0, 1), y0, band=(1, 1), **common)
    band_user = solve_bdf(_heat_rhs, (0, 1), y0, band=(1, 1),
                          jac=_heat_band_jac, **common)

    assert np.max(np.abs(band_fd.y[:, -1] - dense.y[:, -1])) < 1e-6
    assert np.max(np.abs(band_user.y[:, -1] - dense.y[:, -1])) < 1e-6


def test_banded_against_scipy():
    n = 25
    y0 = np.sin(np.arange(1, n + 1, dtype=float))
    r = solve_bdf(_heat_rhs, (0, 1), y0, band=(1, 1),
                  rtol=1e-8, atol=1e-10, t_eval=[1.0])
    sp = solve_ivp(_heat_rhs, (0, 1), y0, method="BDF",
                   rtol=1e-8, atol=1e-10, t_eval=[1.0])
    assert np.max(np.abs(r.y[:, -1] - sp.y[:, -1])) < 1e-6


def test_constant_jacobian():
    # Linear system y' = A y with a constant Jacobian A.
    A = np.array([[-2.0, 1.0], [1.0, -2.0]])
    r = solve_bdf(lambda t, y: A @ y, (0, 3), [1.0, 0.0], t_eval=[3.0],
                  rtol=1e-9, atol=1e-12, jac=A)
    # Analytic solution via matrix exponential.
    from scipy.linalg import expm
    exact = expm(3 * A) @ np.array([1.0, 0.0])
    assert np.max(np.abs(r.y[:, -1] - exact)) < 1e-6
    assert r.njev == 1  # constant Jacobian evaluated exactly once


def test_stepwise_output():
    solver = BDF(lambda t, y: -y, 0.0, [1.0], 2.0, rtol=1e-7, atol=1e-9)
    n = 0
    while solver.status == "running":
        solver.step()
        n += 1
    assert solver.status == "finished"
    assert abs(solver.t - 2.0) < 1e-12
    assert abs(solver.y[0] - np.exp(-2.0)) < 1e-5
    assert n > 0


def test_callback_exception_propagates():
    def bad(t, y):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        solve_bdf(bad, (0, 1), [1.0])


def test_t_eval_multiple_points():
    t_eval = np.linspace(0, 5, 11)
    r = solve_bdf(lambda t, y: -y, (0, 5), [1.0], t_eval=t_eval,
                  rtol=1e-9, atol=1e-12)
    # First stored point is the initial condition, then each t_eval value.
    assert r.t[0] == 0.0
    np.testing.assert_allclose(r.t[1:], t_eval, atol=1e-12)
    np.testing.assert_allclose(r.y[0, 1:], np.exp(-t_eval), atol=1e-6)
