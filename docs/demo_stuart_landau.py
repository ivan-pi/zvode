"""
Demo: Stuart-Landau equation — supercritical Hopf bifurcation normal form.

    dA/dt = (μ + iω₀)A - (1 + iγ)|A|²A,   A ∈ ℂ,   t ∈ [0, 8]

Parameters: μ = 1.0 (bifurcation parameter), ω₀ = 2.0 (natural frequency),
γ = 0.5 (non-isochronous coefficient).

With μ > 0, every non-zero initial condition spirals onto a stable limit cycle
of radius √μ.  Writing A = R·exp(iφ) separates the dynamics into a logistic
equation for R² and a quadrature for φ, yielding the closed-form solution

    R(t) = √μ / √(1 + C·exp(-2μt)),   C = μ/|A₀|² - 1
    φ(t) = φ₀ + (ω₀ - γμ)t + (γ/2)·ln[(1 + C) / (1 + C·exp(-2μt))]
    A(t) = R(t)·exp(iφ(t))

On the limit cycle R = √μ the phase advances at the shifted frequency
ω_lc = ω₀ - γμ.  Because dφ/dt = ω₀ - γR², trajectories that start far
outside (R₀ >> √μ) initially rotate at a lower — or even negative —
instantaneous frequency before settling to ω_lc.

The demo integrates five trajectories (three inside, two outside the limit
cycle), checks them against the exact solution, and plots the phase portrait
in the complex plane together with the amplitude relaxation curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from zvode import ZVODE


# --- parameters -----------------------------------------------------------
mu = 1.0      # bifurcation parameter  (μ > 0: stable limit cycle exists)
omega0 = 2.0  # natural frequency
gamma = 0.5   # non-isochronous (frequency-amplitude coupling) coefficient

T = 8.0
t_eval = np.linspace(0.0, T, 800)


def f(t, y):
    A = y[0]
    return [(mu + 1j * omega0) * A - (1.0 + 1j * gamma) * abs(A) ** 2 * A]


def exact(t, A0):
    """Closed-form solution for given scalar initial condition A0."""
    R0 = abs(A0)
    phi0 = np.angle(A0)
    C = mu / R0**2 - 1.0
    e = np.exp(-2.0 * mu * t)
    R = np.sqrt(mu / (1.0 + C * e))
    phi = phi0 + (omega0 - gamma * mu) * t + (gamma / 2.0) * np.log(
        (1.0 + C) / (1.0 + C * e)
    )
    return R * np.exp(1j * phi)


# --- initial conditions: three inside, two outside the limit cycle --------
R_lc = np.sqrt(mu)           # limit-cycle radius
omega_lc = omega0 - gamma * mu  # limit-cycle frequency

A0_list = [
    0.20 + 0.00j,                       # inside, positive real axis
    0.35 * np.exp(1j * 0.90 * np.pi),  # inside, upper left
    0.25 * np.exp(1j * 1.40 * np.pi),  # inside, lower left
    2.20 + 0.00j,                       # outside, positive real axis
    1.60 * np.exp(1j * 0.40 * np.pi),  # outside, upper right
]

# --- integrate all trajectories -------------------------------------------
y0_arrays = [np.array([A0], dtype=np.complex128) for A0 in A0_list]
sols = [
    solve_ivp(f, (0.0, T), y0, method=ZVODE, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    for y0 in y0_arrays
]

# --- verify accuracy for the first trajectory -----------------------------
A_num = sols[0].y[0]
A_ref = exact(t_eval, A0_list[0])
err = np.max(np.abs(A_num - A_ref))

print(f"nfev={sols[0].nfev}, njev={sols[0].njev}, nlu={sols[0].nlu}")
print(f"Max |error| vs exact (A₀ = {A0_list[0]}): {err:.2e}")
print(f"Limit-cycle radius  √μ     = {R_lc:.6f}")
print(f"Limit-cycle frequency ω_lc = {omega_lc:.6f}")
print(
    f"Initial instantaneous frequency for A₀ = 2.20: "
    f"{omega0 - gamma * 2.20**2:.4f}  (< 0 → starts rotating backward)"
)

# --- plot -----------------------------------------------------------------
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
fig, (ax_phase, ax_amp) = plt.subplots(1, 2, figsize=(13, 5))

# left panel: phase portrait in the complex plane --------------------------
theta = np.linspace(0, 2 * np.pi, 300)
ax_phase.plot(
    R_lc * np.cos(theta),
    R_lc * np.sin(theta),
    "k--",
    lw=1.5,
    label=f"Limit cycle $|A| = \\sqrt{{\\mu}} = {R_lc:.2f}$",
    zorder=2,
)
for sol, A0, color in zip(sols, A0_list, colors):
    A = sol.y[0]
    ax_phase.plot(A.real, A.imag, color=color, lw=1.2, alpha=0.9)
    ax_phase.plot(A.real[0], A.imag[0], "o", color=color, ms=7, zorder=5)

ax_phase.set_aspect("equal")
ax_phase.set_title("Phase portrait (complex plane)")
ax_phase.set_xlabel("Re $A$")
ax_phase.set_ylabel("Im $A$")
ax_phase.legend(loc="upper right", fontsize=9)
ax_phase.grid(True, alpha=0.4)

# right panel: amplitude |A(t)| vs time -----------------------------------
ax_amp.axhline(
    R_lc,
    color="k",
    ls="--",
    lw=1.5,
    label=f"$|A|_{{\\infty}} = \\sqrt{{\\mu}} = {R_lc:.2f}$",
)
for sol, A0, color in zip(sols, A0_list, colors):
    A_num_i = sol.y[0]
    A_ex_i = exact(t_eval, A0)
    ax_amp.plot(t_eval, np.abs(A_num_i), color=color, lw=2)
    ax_amp.plot(t_eval, np.abs(A_ex_i), ":", color="k", lw=1.0, alpha=0.5)

ax_amp.plot([], [], "k-", lw=2, label="ZVODE (numerical)")
ax_amp.plot([], [], "k:", lw=1.2, alpha=0.6, label="Exact")
ax_amp.set_xlabel("$t$")
ax_amp.set_ylabel("$|A(t)|$")
ax_amp.set_title("Amplitude relaxation to the limit cycle")
ax_amp.legend()
ax_amp.grid(True, alpha=0.4)

plt.suptitle(
    r"Stuart-Landau: $\dot{A} = (\mu + i\omega_0)A - (1+i\gamma)|A|^2 A$"
    + f"\n$\\mu={mu},\\; \\omega_0={omega0},\\; \\gamma={gamma}$"
)
plt.tight_layout()
plt.show()
