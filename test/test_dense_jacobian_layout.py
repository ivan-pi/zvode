"""Dense user-Jacobian copy path: C-order vs F-order vs list-of-lists.

Exercises the MITER=1 (mf=21: BDF with a user-supplied dense Jacobian) path,
where the array returned by ``jac(t, y)`` is funnelled through the F-contiguous
coercion + column copy in ``src/_zvode.c`` (the ``jac_adaptor`` /
``PyArray_FROM_OTF(..., NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST)`` block).

PD is a Fortran (column-major) array handed to ZGETRF, so a conventional
row-major Jacobian must be transposed into it.  These tests pin two properties
of that copy:

* **Layout invariance** — whether the callback returns a C-contiguous array,
  an F-contiguous array, or a plain list of lists, the data copied into PD must
  be identical, so the whole integration must be byte-for-byte reproducible.
* **The transpose is actually applied** — the test Jacobian is deliberately
  asymmetric (``J[0,1] = 2e3`` while ``J[1,0] = K1 + i*W``).  With the correct
  (transposed) exact Jacobian, Newton converges cleanly in a few hundred steps
  with zero corrector-convergence failures.  A wrong transpose still reaches the
  right answer (the residual comes from ``fun``) but at ~770x the step count and
  with >100k convergence failures, which the ``nsteps`` / ``ncfn`` bounds catch.

Problem: a complex-valued, stiff Robertson-type 3-species system.  An imaginary
rotation ``W`` makes the dynamics and the Jacobian genuinely complex rather than
a real system stored in complex arrays.
"""

import numpy as np
import pytest

from zvode import solve_complex_ivp

# ---------------------------------------------------------------------------
# Complex Robertson-type stiff problem
#   y0' = -K1 y0 + K3 y1 y2 - i W y0
#   y1' =  K1 y0 - K3 y1 y2 - K2 y1^2 + i W y0
#   y2' =  K2 y1^2
# ---------------------------------------------------------------------------

K1, K2, K3 = 0.04, 3.0e7, 1.0e4
W = 0.3  # imaginary coupling: genuinely complex dynamics and Jacobian
Y0 = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j])
TSPAN = [0.0, 40.0]
TOLS = dict(rtol=1e-8, atol=1e-10)


def fun(t, y):
    y0, y1, y2 = y
    f = np.empty(3, dtype=np.complex128)
    f[0] = -K1 * y0 + K3 * y1 * y2 - 1j * W * y0
    f[1] = K1 * y0 - K3 * y1 * y2 - K2 * y1 * y1 + 1j * W * y0
    f[2] = K2 * y1 * y1
    return f


def _jac_rows(t, y):
    """Exact Jacobian J[i, j] = df(i)/dy(j) as a list of rows (row-major)."""
    y0, y1, y2 = y
    return [
        [-K1 - 1j * W, K3 * y2, K3 * y1],
        [K1 + 1j * W, -K3 * y2 - 2 * K2 * y1, -K3 * y1],
        [0.0, 2 * K2 * y1, 0.0],
    ]


def jac_C(t, y):
    """C-contiguous (row-major) ndarray — the common case; copy transposes it."""
    return np.ascontiguousarray(np.array(_jac_rows(t, y), dtype=np.complex128))


def jac_F(t, y):
    """F-contiguous ndarray — taken as a zero-copy view by the adaptor."""
    return np.asfortranarray(np.array(_jac_rows(t, y), dtype=np.complex128))


def jac_list(t, y):
    """Plain Python list of lists — coerced and cast by PyArray_FROM_OTF."""
    return _jac_rows(t, y)


_LAYOUTS = {"C": jac_C, "F": jac_F, "list": jac_list}

# Reference run via the F-contiguous (zero-copy) path, plus the no-Jacobian
# finite-difference baseline for an independent correctness anchor.  Computed
# once at import; each is a fast (~480-step) solve.
_REF = solve_complex_ivp(fun, TSPAN, Y0, method="BDF", jac=jac_F, **TOLS)
_FD_FINAL = solve_complex_ivp(fun, TSPAN, Y0, method="BDF", **TOLS).y[:, -1]

_STATS = ("nsteps", "nfev", "njev", "nlu", "nni", "ncfn", "netf")


@pytest.mark.parametrize("layout", ["C", "F", "list"])
def test_dense_jacobian_layout(layout):
    """C/F/list dense Jacobian returns are equivalent and correctly transposed."""
    r = solve_complex_ivp(
        fun, TSPAN, Y0, method="BDF", jac=_LAYOUTS[layout], **TOLS
    )

    # The exact (transposed) Jacobian is genuinely being used: Newton converges
    # cleanly.  A wrong transpose balloons to >300k steps / >100k ncfn here.
    assert r.success
    assert r.ncfn == 0
    assert r.nsteps < 5000
    assert np.allclose(r.y[:, -1], _FD_FINAL, rtol=1e-5)

    # Layout invariance: identical to the zero-copy F-contiguous reference,
    # down to the trajectory samples and every integer counter.
    assert np.array_equal(r.t, _REF.t)
    assert np.array_equal(r.y, _REF.y)
    for k in _STATS:
        assert r[k] == _REF[k], f"{k}: {r[k]} != {_REF[k]} for layout {layout}"
