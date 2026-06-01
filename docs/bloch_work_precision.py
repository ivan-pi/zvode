"""
Work-precision diagram for the Bloch equations: ZVODE-BDF vs SciPy BDF.

The Bloch equations model a two-level open quantum system (Rabi oscillations
with spontaneous emission).  With rapid Rabi oscillations (Ω=100) and slow
decay (Γ=1) the system is stiff, yet it has only 4 complex components.  At
this small size the solver is in the *latency-bound* regime: per-step fixed
costs and Python-call overhead dominate over linear-algebra work.

ZVODE keeps its internal step-control, error-checking, and LU factorisation
loop in Fortran, calling back to Python only for RHS evaluations.  SciPy BDF
implements those same operations in Python.  The difference shows up most
clearly at loose tolerances, where the step count is small and per-step
overhead is the dominant cost.

Reference:
  https://discourse.julialang.org/t/how-can-i-solve-complex-valued-odes/110581
"""

import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from zvode import ZVODE_BDF

# ---------------------------------------------------------------------------
# Problem definition: Bloch equations (4-component complex ODE)
# ---------------------------------------------------------------------------
Omega = 100.0   # Rabi frequency
Delta = 0.0     # detuning
Gamma = 1.0     # spontaneous-emission rate
gamma = Gamma / 2.0

t_span = (0.0, 7.0)
u0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)


def bloch_rhs(t, u):
    du = np.empty(4, dtype=np.complex128)
    diff = u[2] - u[3]
    du[0] =  1j * Omega * diff + Gamma * u[1]
    du[1] = -1j * Omega * diff - Gamma * u[1]
    du[2] = -(gamma + 1j * Delta) * u[2] - 1j * Omega * (u[1] - u[0])
    du[3] = np.conj(du[2])
    return du


# ---------------------------------------------------------------------------
# Reference solution (tight tolerance)
# ---------------------------------------------------------------------------
ref = solve_ivp(
    bloch_rhs, t_span, u0.copy(),
    method=ZVODE_BDF, rtol=1e-13, atol=1e-13, dense_output=False,
)
assert ref.success, f"Reference solve failed: {ref.message}"
u_ref = ref.y[:, -1]

# ---------------------------------------------------------------------------
# Tolerance sweep
# ---------------------------------------------------------------------------
tols = np.logspace(-2, -10, 17)
N_REPEAT = 15   # timing repeats; minimum is reported

solvers = [
    ("ZVODE-BDF", ZVODE_BDF,  "tab:blue",   "o"),
    ("SciPy BDF", "BDF",      "tab:orange", "s"),
]

results = {}
for label, method, _color, _marker in solvers:
    wall_ms = np.empty(len(tols))
    errors  = np.empty(len(tols))
    for i, tol in enumerate(tols):
        kw = dict(rtol=tol, atol=tol, dense_output=False)
        # warmup run (avoids import / JIT artefacts)
        sol = solve_ivp(bloch_rhs, t_span, u0.copy(), method=method, **kw)
        # timed runs
        ts = timeit.repeat(
            lambda: solve_ivp(bloch_rhs, t_span, u0.copy(), method=method, **kw),
            number=1, repeat=N_REPEAT,
        )
        wall_ms[i] = min(ts) * 1e3
        errors[i]  = np.linalg.norm(sol.y[:, -1] - u_ref)
        print(f"  {label:10s}  tol={tol:.0e}  t={wall_ms[i]:.2f} ms  err={errors[i]:.2e}")
    results[label] = (wall_ms, errors)

# ---------------------------------------------------------------------------
# Speedup summary
# ---------------------------------------------------------------------------
z_times = results["ZVODE-BDF"][0]
b_times = results["SciPy BDF"][0]
speedup  = b_times / z_times
print(f"\nSpeedup (SciPy BDF / ZVODE-BDF): min={speedup.min():.1f}×  "
      f"max={speedup.max():.1f}×  median={np.median(speedup):.1f}×")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for label, method, color, marker in solvers:
    wall_ms, errors = results[label]
    fmt = f"{marker}-"
    axes[0].loglog(tols,   wall_ms, fmt, label=label, color=color, lw=1.5)
    axes[1].loglog(errors, wall_ms, fmt, label=label, color=color, lw=1.5)

# --- Left panel: time vs tolerance ---
ax = axes[0]
ax.invert_xaxis()
ax.set_xlabel("Tolerance  (rtol = atol)")
ax.set_ylabel("Wall-clock time  (ms)")
ax.set_title("Time vs tolerance")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# --- Right panel: classic work-precision ---
ax = axes[1]
ax.set_xlabel(r"$\|u(7) - u_\mathrm{ref}\|_2$")
ax.set_ylabel("Wall-clock time  (ms)")
ax.set_title("Work-precision  (time vs error)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(
    r"Bloch equations  ($\Omega=100$, $\Delta=0$, $\Gamma=1$),  $t \in [0, 7]$"
    "\nZVODE-BDF vs SciPy BDF — 4-component complex ODE, latency-bound regime",
    fontsize=11,
)
plt.tight_layout()

out_path = "docs/bloch_work_precision.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
plt.show()
