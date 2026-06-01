"""
Work-precision diagram for the Bloch equations: ZVODE-BDF vs SciPy BDF,
with and without a user-supplied analytic Jacobian.

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

Jacobian notes
--------------
f₀, f₁, f₂ are analytic in u.  f₃ = conj(f₂) is anti-analytic, so the
Jacobian row J[3,:] is obtained as conj(J[2,:]) — the same result that
ZVODE's finite-difference scheme (real perturbations) produces naturally.
The column J[:,3] is zero for f₀, f₁, f₂; J[3,3] = 0 because f₃ does not
depend on u₃.

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


def bloch_jac(t, u):
    J = np.zeros((4, 4), dtype=np.complex128)
    # df0/duj
    J[0, 1] =  Gamma
    J[0, 2] =  1j * Omega
    J[0, 3] = -1j * Omega
    # df1/duj
    J[1, 1] = -Gamma
    J[1, 2] = -1j * Omega
    J[1, 3] =  1j * Omega
    # df2/duj  (analytic)
    J[2, 0] =  1j * Omega
    J[2, 1] = -1j * Omega
    J[2, 2] = -(gamma + 1j * Delta)
    # df3/duj = conj(df2/duj)  (real-perturbation finite-diff sense)
    J[3, 0] = -1j * Omega
    J[3, 1] =  1j * Omega
    J[3, 2] = -(gamma - 1j * Delta)
    # J[3, 3] = 0: f3 does not depend on u3
    return J


# ---------------------------------------------------------------------------
# Reference solution (tight tolerance, analytic Jacobian)
# ---------------------------------------------------------------------------
ref = solve_ivp(
    bloch_rhs, t_span, u0.copy(),
    method=ZVODE_BDF, jac=bloch_jac, rtol=1e-13, atol=1e-13, dense_output=False,
)
assert ref.success, f"Reference solve failed: {ref.message}"
u_ref = ref.y[:, -1]

# ---------------------------------------------------------------------------
# Solver configurations: (label, method, jac, color, marker, linestyle)
# ---------------------------------------------------------------------------
CONFIGS = [
    ("ZVODE-BDF",           ZVODE_BDF, None,       "tab:blue",   "o", "--"),
    ("ZVODE-BDF + jac",     ZVODE_BDF, bloch_jac,  "tab:blue",   "o", "-"),
    ("SciPy BDF",           "BDF",     None,       "tab:orange", "s", "--"),
    ("SciPy BDF + jac",     "BDF",     bloch_jac,  "tab:orange", "s", "-"),
]

# ---------------------------------------------------------------------------
# Tolerance sweep
# ---------------------------------------------------------------------------
tols    = np.logspace(-10, -2, 17)
N_REPEAT = 10   # timing repeats; minimum is reported

results = {}
for label, method, jac, _c, _m, _ls in CONFIGS:
    wall_ms = np.empty(len(tols))
    errors  = np.empty(len(tols))
    nfev    = np.empty(len(tols), dtype=int)
    njev    = np.empty(len(tols), dtype=int)
    nlu     = np.empty(len(tols), dtype=int)
    for i, tol in enumerate(tols):
        kw = dict(rtol=tol, atol=tol, jac=jac, dense_output=False)
        sol = solve_ivp(bloch_rhs, t_span, u0.copy(), method=method, **kw)  # warmup
        ts  = timeit.repeat(
            lambda: solve_ivp(bloch_rhs, t_span, u0.copy(), method=method, **kw),
            number=1, repeat=N_REPEAT,
        )
        wall_ms[i] = min(ts) * 1e3
        errors[i]  = np.linalg.norm(sol.y[:, -1] - u_ref)
        nfev[i], njev[i], nlu[i] = sol.nfev, sol.njev, sol.nlu
        print(f"  {label:20s}  tol={tol:.0e}  t={wall_ms[i]:8.2f} ms  err={errors[i]:.2e}"
              f"  nfev={nfev[i]}  njev={njev[i]}  nlu={nlu[i]}")
    results[label] = dict(wall_ms=wall_ms, errors=errors, nfev=nfev, njev=njev, nlu=nlu)

# ---------------------------------------------------------------------------
# Speedup summary
# ---------------------------------------------------------------------------
print()
for label_num, label_den in [
    ("ZVODE-BDF",     "ZVODE-BDF + jac"),
    ("SciPy BDF",     "SciPy BDF + jac"),
    ("SciPy BDF",     "ZVODE-BDF"),
    ("SciPy BDF + jac", "ZVODE-BDF + jac"),
]:
    ratio = results[label_num]["wall_ms"] / results[label_den]["wall_ms"]
    print(f"  {label_num:20s} / {label_den:20s}: "
          f"median={np.median(ratio):.2f}×  range=[{ratio.min():.2f}, {ratio.max():.2f}]×")

# ---------------------------------------------------------------------------
# Figure 1: work-precision (time vs tolerance and time vs error)
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

for label, method, jac, color, marker, ls in CONFIGS:
    r = results[label]
    axes[0].loglog(tols,         r["wall_ms"], ls, marker=marker, label=label, color=color, lw=1.5)
    axes[1].loglog(r["errors"],  r["wall_ms"], ls, marker=marker, label=label, color=color, lw=1.5)

ax = axes[0]
ax.set_xlabel("Tolerance  (rtol = atol)")
ax.set_ylabel("Wall-clock time  (ms)")
ax.set_title("Time vs tolerance")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.set_xlabel(r"$\|u(7) - u_\mathrm{ref}\|_2$")
ax.set_ylabel("Wall-clock time  (ms)")
ax.set_title("Work-precision  (time vs error)")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

fig1.suptitle(
    r"Bloch equations  ($\Omega=100$, $\Delta=0$, $\Gamma=1$),  $t \in [0, 7]$"
    "\nZVODE-BDF vs SciPy BDF — with (—) and without (- -) analytic Jacobian",
    fontsize=11,
)
plt.tight_layout()
fig1.savefig("docs/bloch_work_precision.png", dpi=150, bbox_inches="tight")
print("\nSaved docs/bloch_work_precision.png")

# ---------------------------------------------------------------------------
# Figure 2: evaluation counts vs tolerance (analytic Jacobian only)
# ---------------------------------------------------------------------------
JAC_CONFIGS = [(l, c, m) for l, _meth, jac, c, m, _ls in CONFIGS if jac is not None]

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

stat_info = [
    ("nfev", "RHS evaluations"),
    ("njev", "Jacobian evaluations"),
    ("nlu",  "LU decompositions"),
]

for ax, (key, ylabel) in zip(axes2, stat_info):
    for label, color, marker in JAC_CONFIGS:
        ax.loglog(tols, results[label][key], "-", marker=marker,
                  label=label, color=color, lw=1.5)
    ax.set_xlabel("Tolerance  (rtol = atol)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig2.suptitle(
    r"Bloch equations  ($\Omega=100$, $\Delta=0$, $\Gamma=1$),  $t \in [0, 7]$"
    "\nEvaluation counts — analytic Jacobian case",
    fontsize=11,
)
plt.tight_layout()
fig2.savefig("docs/bloch_eval_counts.png", dpi=150, bbox_inches="tight")
print("Saved docs/bloch_eval_counts.png")

plt.show()
