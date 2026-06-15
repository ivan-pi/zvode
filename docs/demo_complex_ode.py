"""
Demo / tutorial: a driven scalar complex ODE solved with ZVODE.

    y'(t) = 10 · exp(2πi·t) · y(t),   y(0) = 1,   t ∈ [0, 5]

This is the example used in the Wolfram Language documentation on
visualizing solutions of complex ODEs, where it is solved with
``NDSolveValue``.

Attribution / original example:
  "Solutions of Complex ODEs", Wolfram Language 12 documentation,
  https://www.wolfram.com/language/12/complex-visualization/solutions-of-complex-odes.html

The right-hand side is holomorphic in ``y`` (it is simply a
time-dependent complex coefficient times ``y``), so ZVODE is a natural
fit: it integrates the complex state directly, with no need to split the
problem into real and imaginary parts.

Analytic solution
-----------------
Separating variables, ``y'/y = 10 e^{2πi t}``, and integrating gives

    y(t) = exp( (10 / (2πi)) · (e^{2πi t} − 1) ).

Because ``e^{2πi t} = 1`` whenever ``t`` is an integer, the solution
returns to ``y = 1`` at every integer time; in particular ``y(5) = 1``.
We use this closed form only to check the numerical accuracy.

The script produces two figures, mirroring the Wolfram example:

  1. the real part, imaginary part, and modulus ``|y|`` versus the real
     variable ``t``,
  2. the trajectory drawn parametrically in the complex plane, with
     arrows indicating the direction of travel.

Run with::

    python docs/demo_complex_ode.py
"""

import numpy as np
import matplotlib.pyplot as plt

from zvode import solve_complex_ivp


# ---------------------------------------------------------------------
# 1. Right-hand side and (optional) analytic reference
# ---------------------------------------------------------------------
def rhs(t, y):
    # y'(t) = 10 e^{2πi t} y(t)
    return [10.0 * np.exp(2j * np.pi * t) * y[0]]


def exact(t):
    # y(t) = exp( (10 / (2πi)) (e^{2πi t} − 1) )
    t = np.asarray(t, dtype=float)
    return np.exp((10.0 / (2j * np.pi)) * (np.exp(2j * np.pi * t) - 1.0))


# ---------------------------------------------------------------------
# 2. Solve with ZVODE
# ---------------------------------------------------------------------
# The problem is non-stiff (a smooth, oscillatory coefficient), so the
# Adams method is the appropriate choice. We request output on a fine,
# uniform grid so the parametric curve is smooth.
t_eval = np.linspace(0.0, 5.0, 501)

sol = solve_complex_ivp(
    fun=rhs,
    tspan=t_eval,
    y0=[1.0 + 0.0j],
    method="Adams",
    rtol=1e-10,
    atol=1e-12,
)

y = sol.y[0]

err = np.max(np.abs(y - exact(t_eval)))
print(f"nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")
print(f"Max absolute error vs. analytic solution: {err:.2e}")
print(f"y(5) = {y[-1]:.6f}  (analytic value: 1)")

# ---------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------
title = r"$y' = 10\,e^{2\pi i t}\,y,\quad y(0) = 1,\quad t \in [0, 5]$"

# Figure 1 — real part, imaginary part, and modulus versus t.
fig1, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_eval, y.real, label=r"$\mathrm{Re}\,y$")
ax.plot(t_eval, y.imag, "--", label=r"$\mathrm{Im}\,y$")
ax.plot(t_eval, np.abs(y), color="tab:purple", label=r"$|y|$")
ax.set_title("Real part, imaginary part, and modulus\n" + title)
ax.set_xlabel("$t$")
ax.set_ylabel("$y$")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.4)
fig1.tight_layout()

# Figure 2 — trajectory in the complex plane, with direction arrows.
fig2, ax_c = plt.subplots(figsize=(6, 6))
ax_c.plot(y.real, y.imag, color="tab:blue", linewidth=2)

# The solution is periodic (period 1), so one loop suffices to read off
# the direction of travel.  Drop a few arrows along the first period; a
# fixed-size arrowhead is drawn even though successive points are close.
period = len(t_eval) // 5  # samples per unit-time period
for frac in (0.15, 0.45, 0.78):
    i = int(frac * period)
    ax_c.annotate(
        "",
        xy=(y.real[i + 1], y.imag[i + 1]),
        xytext=(y.real[i], y.imag[i]),
        arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2, mutation_scale=22),
    )

ax_c.plot(y.real[0], y.imag[0], "o", color="tab:green", zorder=5, label="start $y(0)=1$")
ax_c.set_aspect("equal")
ax_c.set_title("Trajectory in the complex plane\n" + title)
ax_c.set_xlabel(r"$\mathrm{Re}\,y$")
ax_c.set_ylabel(r"$\mathrm{Im}\,y$")
ax_c.legend(loc="upper right")
ax_c.grid(True, alpha=0.4)
fig2.tight_layout()

plt.show()
