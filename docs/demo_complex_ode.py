"""
Demo / tutorial: a driven scalar complex ODE solved with ZVODE.

    y'(t) = 10 · exp(2πi·t) · y(t),   y(0) = 1,   t ∈ [0, 5]

This is the example used in the Wolfram Language documentation on
visualizing solutions of complex ODEs, where it is solved with
``NDSolveValue``:

    z = NDSolveValue[{y'[t] == 10 E^(2 Pi I t) y[t], y[0] == 1},
                     y, {t, 0, 5}];

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

The script produces three plots, mirroring the Wolfram example:

  1. real and imaginary parts of ``y`` versus the real variable ``t``,
  2. the modulus ``|y|`` versus ``t``,
  3. the trajectory drawn parametrically in the complex plane, coloured
     by ``t``.

Run with::

    python docs/demo_complex_ode.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
ax_ri = fig.add_subplot(gs[0, 0])   # real & imaginary parts vs t
ax_abs = fig.add_subplot(gs[1, 0])  # |y| vs t
ax_c = fig.add_subplot(gs[:, 1])    # parametric curve in complex plane

# (1) Real and imaginary parts vs t
ax_ri.plot(t_eval, y.real, label=r"$\mathrm{Re}\,y$")
ax_ri.plot(t_eval, y.imag, "--", label=r"$\mathrm{Im}\,y$")
ax_ri.set_title("Real and imaginary parts")
ax_ri.set_xlabel("$t$")
ax_ri.legend(loc="upper right")
ax_ri.grid(True, alpha=0.4)

# (2) Modulus |y| vs t
ax_abs.plot(t_eval, np.abs(y), color="tab:purple")
ax_abs.set_title(r"Modulus $|y|$")
ax_abs.set_xlabel("$t$")
ax_abs.set_ylabel("$|y|$")
ax_abs.grid(True, alpha=0.4)

# (3) Parametric trajectory in the complex plane, coloured by t
points = np.column_stack([y.real, y.imag]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap="viridis", linewidth=2)
lc.set_array(t_eval)
ax_c.add_collection(lc)
fig.colorbar(lc, ax=ax_c, label="$t$")

ax_c.plot(y.real[0], y.imag[0], "o", color="tab:green", zorder=5, label="$t=0$")
ax_c.plot(y.real[-1], y.imag[-1], "s", color="tab:red", zorder=5, label="$t=5$")
ax_c.autoscale()
ax_c.set_aspect("equal")
ax_c.set_title("Trajectory in the complex plane")
ax_c.set_xlabel(r"$\mathrm{Re}\,y$")
ax_c.set_ylabel(r"$\mathrm{Im}\,y$")
ax_c.legend(loc="upper right")
ax_c.grid(True, alpha=0.4)

fig.suptitle(r"$y' = 10\,e^{2\pi i t}\,y,\quad y(0) = 1,\quad t \in [0, 5]$")
fig.tight_layout()
plt.show()
