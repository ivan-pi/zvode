"""
Work-precision benchmark: zvode vs SciPy on a stiff complex ODE system.

We integrate a small (3-component) complex-valued, non-linearly coupled,
stiff ODE and compare the cost (wall-clock time) required to reach a given
accuracy.  Four solver paths are compared:

  1. ``zvode.solve_complex_ivp``        — zvode's low-overhead procedural API
  2. ``zvode.ZVODE_BDF`` via solve_ivp  — zvode's scipy-compatible OdeSolver
  3. SciPy ``solve_ivp(method="BDF")``  — SciPy's pure-Python BDF integrator
  4. SciPy ``scipy.integrate.ode("zvode", method="bdf")`` — the classic,
     lowest-overhead Fortran ZVODE wrapper shipped with SciPy

Paths 1, 2 and 4 all drive the *same* Fortran ZVODE core, so they solve the
problem with (nearly) identical step sequences; the spread between them is
pure interface overhead.  Path 3 is a different algorithm implemented in
Python and serves as the "pure-Python OdeSolver" reference.

The system is small, so the integrators run in the *latency-bound* regime:
per-step fixed costs and Python call-back overhead dominate over the actual
linear-algebra work.  This is exactly the regime where keeping the step
loop in Fortran (paths 1, 2, 4) pays off relative to a Python step loop
(path 3), and where a thin wrapper (path 1) beats a heavier one (path 2).

Test problem
------------
A complex autocatalytic kinetics system with conserved total mass.  The
right-hand side is a polynomial (hence holomorphic / complex-analytic), the
``y2*y3`` term provides non-linear coupling, the large rate ``a`` makes it
stiff, and the conservation law ``y1 + y2 + y3 = const`` keeps the solution
bounded:

    y1' = -a*y1 + b*y2*y3
    y2' =  a*y1 - b*y2*y3 - c*y2
    y3' =  c*y2

Complex coefficients (a, c) and a complex initial condition make the state
genuinely complex.  Conservation of ``y1 + y2 + y3`` doubles as a correctness
check for every solver.

Run it::

    python docs/bench_work_precision.py

It writes ``docs/work_precision.png`` and prints a results table together
with the exact library versions used (stamped onto the figure for
reproducibility).
"""

import importlib.metadata
import platform
import subprocess
import timeit
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import ode, solve_ivp

from zvode import ZVODE_BDF, solve_complex_ivp

# solve_ivp always forwards vectorized= to the OdeSolver; ZVODE_BDF ignores it
# and warns once per call.  It is harmless here and only clutters the output.
warnings.filterwarnings("ignore", message=".*vectorized.*")

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
a = 1000.0 + 200.0j   # fast (stiff) forward rate, with oscillation
b = 1000.0            # non-linear back-reaction rate
c = 1.0 + 50.0j       # slow decay, with oscillation

t0, tf = 0.0, 3.0
y0 = np.array([1.0 + 0.0j, 0.0 + 0.2j, 0.0 + 0.0j], dtype=np.complex128)


def rhs(t, y):
    y1, y2, y3 = y
    return np.array(
        [
            -a * y1 + b * y2 * y3,
            a * y1 - b * y2 * y3 - c * y2,
            c * y2,
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Solver drivers — each returns the state vector at t = tf
#
# Output mode is each interface's idiomatic, lowest-overhead one:
#
#   * solve_complex_ivp and both solve_ivp drivers collect the full trajectory
#     (every accepted step).  This is solve_ivp's standard mode, and it is the
#     *cheapest* one for it: forcing endpoint-only via t_eval=(tf,) is actually
#     slower, because solve_ivp then runs an np.searchsorted per step.
#     solve_complex_ivp collects in C, so trajectory vs. endpoint-only makes
#     essentially no difference.  The three lines therefore have matching
#     output semantics and are directly comparable.
#
#   * ode("zvode") integrates straight to tf in a single Fortran call.  That is
#     its natural low-overhead mode; making it collect a trajectory would
#     require a manual Python step loop, defeating its role as the bare-core
#     reference.  Its per-step Fortran storage is not exposed to Python.
# ---------------------------------------------------------------------------
def run_solve_complex_ivp(tol):
    sol = solve_complex_ivp(rhs, (t0, tf), y0, method="BDF", rtol=tol, atol=tol)
    return sol.y[:, -1]


def run_zvode_bdf(tol):
    sol = solve_ivp(rhs, (t0, tf), y0, method=ZVODE_BDF, rtol=tol, atol=tol)
    return sol.y[:, -1]


def run_scipy_bdf(tol):
    sol = solve_ivp(rhs, (t0, tf), y0, method="BDF", rtol=tol, atol=tol)
    return sol.y[:, -1]


def run_scipy_ode_zvode(tol):
    # with_jacobian=True selects the stiff Newton path with an internally
    # generated Jacobian (MF=22).  Without it SciPy's ode("zvode") defaults to
    # functional iteration (MITER=0), which is not a fair stiff baseline: it
    # caps the order and takes far more (tiny) steps.
    r = ode(rhs).set_integrator(
        "zvode", method="bdf", with_jacobian=True,
        rtol=tol, atol=tol, nsteps=1_000_000,
    )
    r.set_initial_value(y0, t0)
    return r.integrate(tf)


CONFIGS = [
    # label                         driver                  color         marker  ls
    ("zvode.solve_complex_ivp",     run_solve_complex_ivp,  "tab:green",  "o", "-"),
    ("zvode.ZVODE_BDF (solve_ivp)", run_zvode_bdf,          "tab:blue",   "D", "-"),
    ("SciPy solve_ivp BDF",         run_scipy_bdf,          "tab:orange", "s", "--"),
    ("SciPy ode('zvode')",          run_scipy_ode_zvode,    "tab:red",    "^", ":"),
]

# ---------------------------------------------------------------------------
# Reference solution (tight tolerance) for error measurement
# ---------------------------------------------------------------------------
ref = solve_ivp(rhs, (t0, tf), y0, method=ZVODE_BDF, rtol=1e-12, atol=1e-13)
assert ref.success, ref.message
u_ref = ref.y[:, -1]

# Cross-check with SciPy's *independent* BDF so the reference is not biased
# toward the ZVODE-backed solvers (which share the Fortran core).
ref_scipy = solve_ivp(rhs, (t0, tf), y0, method="BDF", rtol=1e-12, atol=1e-13)
assert np.linalg.norm(ref_scipy.y[:, -1] - u_ref) < 1e-9, "reference solvers disagree"

# ---------------------------------------------------------------------------
# Tolerance sweep
# ---------------------------------------------------------------------------
tols = np.logspace(-3, -11, 9)
N_REPEAT = 10  # timing repeats; the minimum is reported (least system noise)

results = {}
print(f"\nIntegrating  y' = f(t, y)  on t in [{t0}, {tf}],  n = {y0.size} "
      f"complex components\n")
for label, driver, *_ in CONFIGS:
    wall_ms = np.empty(tols.size)
    errors = np.empty(tols.size)
    print(f"{label}")
    for i, tol in enumerate(tols):
        y_end = driver(tol)                                   # warm-up
        ts = timeit.repeat(lambda: driver(tol), number=1, repeat=N_REPEAT)
        wall_ms[i] = min(ts) * 1e3
        errors[i] = np.linalg.norm(y_end - u_ref)
        drift = abs(np.sum(y_end) - np.sum(y0))               # conservation check
        print(f"    tol={tol:.0e}   t={wall_ms[i]:7.3f} ms   "
              f"err={errors[i]:.2e}   drift={drift:.1e}")
    results[label] = dict(wall_ms=wall_ms, errors=errors)

# ---------------------------------------------------------------------------
# Overhead summary, relative to the classic SciPy ode('zvode') interface
# ---------------------------------------------------------------------------
base = results["SciPy ode('zvode')"]["wall_ms"]
print("\nMedian wall-clock ratio (lower is less overhead):")
for label, *_ in CONFIGS:
    ratio = results[label]["wall_ms"] / base
    print(f"    {label:30s} {np.median(ratio):5.2f}x  vs  SciPy ode('zvode')")

# ---------------------------------------------------------------------------
# Provenance stamp
# ---------------------------------------------------------------------------
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    commit = "unknown"

stamp = (
    f"zvode {importlib.metadata.version('zvode')} (commit {commit})  |  "
    f"SciPy {scipy.__version__}  |  NumPy {np.__version__}  |  "
    f"Python {platform.python_version()}  |  {platform.platform()}"
)
print("\n" + stamp)

# ---------------------------------------------------------------------------
# Work-precision figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for label, _driver, color, marker, ls in CONFIGS:
    r = results[label]
    axes[0].loglog(r["errors"], r["wall_ms"], ls, marker=marker,
                   label=label, color=color, lw=1.5)
    axes[1].loglog(tols, r["wall_ms"], ls, marker=marker,
                   label=label, color=color, lw=1.5)

axes[0].set_xlabel(r"end-point error  $\|y(t_f) - y_\mathrm{ref}\|_2$")
axes[0].set_ylabel("wall-clock time  (ms)")
axes[0].set_title("Work-precision  (time vs error)")
axes[0].legend(fontsize=9)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].set_xlabel("tolerance  (rtol = atol)")
axes[1].set_ylabel("wall-clock time  (ms)")
axes[1].set_title("Cost vs tolerance")
axes[1].legend(fontsize=9)
axes[1].grid(True, which="both", alpha=0.3)

fig.suptitle(
    "Stiff complex kinetics  (3 coupled complex components),  "
    rf"$t \in [{t0:g}, {tf:g}]$",
    fontsize=12,
)
fig.text(0.5, -0.02, stamp, ha="center", va="top", fontsize=7, color="0.4")
plt.tight_layout()

out = Path(__file__).resolve().parent / "work_precision.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
