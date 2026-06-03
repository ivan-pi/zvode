"""
Benchmark: Adams vs BDF and tolerance scaling
==============================================
Explores when to choose Adams (non-stiff) vs BDF (stiff), and how
wall-clock time scales with the requested tolerance.

Adams (lmm='Adams')
  Variable-order Adams–Moulton predictor-corrector, up to order 12.
  Uses functional iteration (no Jacobian / LU factorisation needed).
  Efficient for smooth, non-stiff problems.

BDF (lmm='BDF')
  Backward differentiation formulae, up to order 5.
  Requires a Jacobian (or finite-difference approximation) and LU
  factorisation at each step.  The right choice for stiff problems.

Test problems
-------------
quantum_chain  – tight-binding chain, non-stiff (purely oscillatory).
                 All eigenvalues have magnitude ~1; Adams should be
                 competitive with BDF here.
decaying_osc   – independent decaying oscillators with decay rates in
                 [1, 100]: stiffness ratio ≈ 100.  Adams must use tiny
                 steps to resolve the fast modes; BDF handles stiffness
                 with large steps.

Run
---
    pytest bench/bench_methods.py -v --benchmark-sort=name
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zvode import ZVODE_Adams, ZVODE_BDF

from problems import make_quantum_chain, make_decaying_oscillators

N = 20          # system size for all method benchmarks
RTOL = 1e-6
ATOL = 1e-9


def _record(benchmark, result):
    benchmark.extra_info["nfev"] = result.nfev
    benchmark.extra_info["njev"] = result.njev
    benchmark.extra_info["n_output_pts"] = len(result.t)


# ---------------------------------------------------------------------------
# Non-stiff problem: quantum chain — Adams should be competitive
# ---------------------------------------------------------------------------

def test_adams_chain(benchmark):
    """Adams on tight-binding chain (non-stiff): no LU per step, high-order."""
    fun, _, _, y0, t_span = make_quantum_chain(N)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_Adams, rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_bdf_chain(benchmark):
    """BDF on tight-binding chain (non-stiff): LU per step, lower order."""
    fun, jac_dense, _, y0, t_span = make_quantum_chain(N)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_dense, miter=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ---------------------------------------------------------------------------
# Stiff problem: decaying oscillators — BDF should dominate
# ---------------------------------------------------------------------------

def test_adams_stiff(benchmark):
    """Adams on stiff decaying oscillators: forces tiny steps for fast modes."""
    fun, _, y0, t_span = make_decaying_oscillators(N)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_Adams, rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


def test_bdf_stiff(benchmark):
    """BDF on stiff decaying oscillators: large steps thanks to A-stability."""
    fun, jac_dense, y0, t_span = make_decaying_oscillators(N)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_dense, miter=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ---------------------------------------------------------------------------
# Tolerance scaling: BDF on the quantum chain at three accuracy levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-9], ids=["rtol=1e-3", "rtol=1e-6", "rtol=1e-9"])
def test_bdf_tolerance(benchmark, rtol):
    """BDF cost as a function of requested tolerance (tight-binding chain)."""
    fun, jac_dense, _, y0, t_span = make_quantum_chain(N)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_dense, miter=1,
        rtol=rtol, atol=rtol * 1e-3,
    )
    _record(benchmark, result)
