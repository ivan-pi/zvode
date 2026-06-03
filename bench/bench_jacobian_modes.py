"""
Benchmark: ZVODE Jacobian modes
================================
Compares four ways to supply the Jacobian to ZVODE_BDF on the complex
tight-binding quantum chain.  The chain has a tridiagonal (banded) Jacobian,
so banded modes scale as O(n) for both evaluation and LU factorization,
while dense modes scale as O(n²) and O(n³) respectively.

Modes tested
------------
auto_dense  (miter=2) – Jacobian approximated by finite differences (dense).
                        Default when no ``jac`` is provided.
auto_banded (miter=5) – Finite-difference Jacobian, but stored and factored
                        as a band matrix.  Requires lband/uband hints.
user_dense  (miter=1) – Exact Jacobian supplied by the user as a dense matrix.
user_banded (miter=4) – Exact Jacobian supplied by the user in band storage.
                        Best for large sparse systems with known sparsity.

Run
---
    pytest bench/bench_jacobian_modes.py -v --benchmark-sort=name
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zvode import ZVODE_BDF

from problems import make_quantum_chain

SIZES = [10, 50, 200]
RTOL, ATOL = 1e-6, 1e-9


def _record(benchmark, result):
    benchmark.extra_info["nfev"] = result.nfev
    benchmark.extra_info["njev"] = result.njev
    benchmark.extra_info["n_output_pts"] = len(result.t)


# ---------------------------------------------------------------------------
# miter=2: auto-generated dense Jacobian (default, no user jac needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
def test_auto_dense(benchmark, n):
    """BDF with internally generated dense Jacobian (miter=2, default)."""
    fun, _, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ---------------------------------------------------------------------------
# miter=5: auto-generated banded Jacobian (no user jac, but bandwidths given)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
def test_auto_banded(benchmark, n):
    """BDF with internally generated banded Jacobian (miter=5, lband=uband=1)."""
    fun, _, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, miter=5, lband=1, uband=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ---------------------------------------------------------------------------
# miter=1: user-supplied exact dense Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
def test_user_dense(benchmark, n):
    """BDF with user-supplied exact dense Jacobian (miter=1)."""
    fun, jac_dense, _, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_dense, miter=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)


# ---------------------------------------------------------------------------
# miter=4: user-supplied exact banded Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
def test_user_banded(benchmark, n):
    """BDF with user-supplied exact banded Jacobian (miter=4, lband=uband=1)."""
    fun, _, jac_banded, y0, t_span = make_quantum_chain(n)
    result = benchmark(
        solve_ivp, fun, t_span, y0,
        method=ZVODE_BDF, jac=jac_banded, miter=4, lband=1, uband=1,
        rtol=RTOL, atol=ATOL,
    )
    _record(benchmark, result)
