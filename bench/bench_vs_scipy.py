"""
Benchmark: ZVODE vs SciPy solvers
===================================
Compares ZVODE against SciPy's built-in stiff solvers on two problems.

Implementation notes
--------------------
ZVODE and SciPy LSODA wrap compiled Fortran (ODEPACK) — they are fast
for all n.  SciPy BDF and Radau are pure-Python / NumPy implementations
and carry significant per-step overhead, especially for small n.

Complex-value support
---------------------
Only ZVODE and SciPy BDF support complex-valued y natively.
SciPy Radau and LSODA raise an error if y0 is complex, so they are
tested only on the real-valued ROBER problem.

Problem 1: tight-binding quantum chain (complex, n sites)
  Tests ability to solve complex ODEs efficiently.
  - ZVODE_BDF no-jac (auto FD, dense) vs user-banded Jacobian
  - SciPy BDF  no-jac (auto FD, dense) vs user-dense Jacobian

Problem 2: ROBER chemical kinetics (real-valued, 3 equations)
  Classic stiff benchmark; lets all solvers compete on equal footing.
  - ZVODE_BDF (wraps Fortran, treats real as complex)
  - SciPy BDF  (Python, real)
  - SciPy Radau (Python, real)
  - SciPy LSODA (wraps Fortran, real — direct ODEPACK counterpart)

Run
---
    pytest bench/bench_vs_scipy.py -v --benchmark-sort=name
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zvode import ZVODE_BDF

from problems import make_quantum_chain, rober_problem

RTOL, ATOL = 1e-6, 1e-9


def _record(benchmark, result):
    benchmark.extra_info["nfev"] = result.nfev
    benchmark.extra_info["njev"] = result.njev
    benchmark.extra_info["n_output_pts"] = len(result.t)


# ===========================================================================
# Problem 1: complex tight-binding quantum chain
# ===========================================================================

CHAIN_SIZES = [10, 50, 200]


@pytest.mark.parametrize("n", CHAIN_SIZES)
def test_zvode_bdf_auto(benchmark, n):
    """ZVODE BDF, finite-difference dense Jacobian (default, miter=2)."""
    fun, _, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


@pytest.mark.parametrize("n", CHAIN_SIZES)
def test_zvode_bdf_user_banded(benchmark, n):
    """ZVODE BDF, user-supplied banded Jacobian (miter=4, lband=uband=1)."""
    fun, _, jac_banded, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_banded, miter=4, lband=1, uband=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


@pytest.mark.parametrize("n", CHAIN_SIZES)
def test_scipy_bdf_auto(benchmark, n):
    """SciPy BDF, finite-difference Jacobian (pure-Python implementation)."""
    fun, _, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="BDF", rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


@pytest.mark.parametrize("n", CHAIN_SIZES)
def test_scipy_bdf_user_dense(benchmark, n):
    """SciPy BDF, user-supplied dense Jacobian (pure-Python implementation)."""
    fun, jac_dense, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="BDF", jac=jac_dense,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ===========================================================================
# Problem 2: ROBER chemical kinetics (real-valued, stiff)
# ===========================================================================


def test_rober_zvode_bdf_auto(benchmark):
    """ROBER: ZVODE BDF, finite-difference Jacobian (Fortran, complex arith)."""
    fun, _, y0, t_span = rober_problem(complex_y0=True)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_zvode_bdf_jac(benchmark):
    """ROBER: ZVODE BDF, user-supplied dense Jacobian (Fortran, complex arith)."""
    fun, jac, y0, t_span = rober_problem(complex_y0=True)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac, miter=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_bdf_auto(benchmark):
    """ROBER: SciPy BDF, finite-difference Jacobian (Python)."""
    fun, _, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="BDF", rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_bdf_jac(benchmark):
    """ROBER: SciPy BDF, user-supplied dense Jacobian (Python)."""
    fun, jac, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="BDF", jac=jac,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_radau_auto(benchmark):
    """ROBER: SciPy Radau, finite-difference Jacobian (Python)."""
    fun, _, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="Radau", rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_radau_jac(benchmark):
    """ROBER: SciPy Radau, user-supplied dense Jacobian (Python)."""
    fun, jac, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="Radau", jac=jac,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_lsoda_auto(benchmark):
    """ROBER: SciPy LSODA, no Jacobian (Fortran, real-valued ODEPACK)."""
    fun, _, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="LSODA", rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_rober_scipy_lsoda_jac(benchmark):
    """ROBER: SciPy LSODA, user-supplied dense Jacobian (Fortran, real-valued ODEPACK)."""
    fun, jac, y0, t_span = rober_problem()
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method="LSODA", jac=jac,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)
