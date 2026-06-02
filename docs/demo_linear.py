"""
Demo: non-autonomous linear complex ODE solved with ZVODE.

    y'(t) = t*y(t) + 2i,   y(0) = 1+i,   t in [0, 2]

Analytic solution (variation of constants / integrating factor exp(-t²/2)):

    y(t) = exp(t²/2) * [ (1+i) + i*sqrt(2π) * erf(t/sqrt(2)) ]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf

from zvode import ZVODE


def f(t, y):
    return t * y + 2j


def jac(t, y):
    return [[t]]


def exact(t):
    t = np.asarray(t)
    return np.exp(t**2 / 2) * ((1 + 1j) + 1j * np.sqrt(2 * np.pi) * erf(t / np.sqrt(2)))


y0 = np.array([1.0 + 1j], dtype=np.complex128)
t_eval = np.linspace(0.0, 2.0, 201)

sol = solve_ivp(
    f,
    (0.0, 2.0),
    y0,
    method=ZVODE,
    jac=jac,
    miter=1,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

y_num = sol.y[0]
y_ref = exact(t_eval)
err = np.abs(y_num - y_ref)

print(f"nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")
print(f"Max absolute error: {np.max(err):.2e}")

from matplotlib.collections import LineCollection

fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

# Left: real and imaginary parts vs time
ax_t.plot(t_eval, y_num.real, label="Re $y$")
ax_t.plot(t_eval, y_num.imag, "--", label="Im $y$")
ax_t.set_title("Components vs. time")
ax_t.set_xlabel("$t$")
ax_t.legend()
ax_t.grid(True, alpha=0.4)

# Right: trajectory in the complex plane, coloured by time
points = np.column_stack([y_num.real, y_num.imag]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap="viridis", linewidth=2)
lc.set_array(t_eval)
ax_c.add_collection(lc)
fig.colorbar(lc, ax=ax_c, label="$t$")

ax_c.plot(y_num.real[0], y_num.imag[0], "o", color="tab:green", zorder=5, label="$t=0$")
ax_c.plot(y_num.real[-1], y_num.imag[-1], "s", color="tab:red", zorder=5, label="$t=2$")
ax_c.autoscale()
ax_c.set_aspect("equal")
ax_c.set_title("Trajectory in the complex plane")
ax_c.set_xlabel("Re $y$")
ax_c.set_ylabel("Im $y$")
ax_c.legend()
ax_c.grid(True, alpha=0.4)

plt.suptitle(r"$y' = t\,y + 2i,\quad y(0) = 1+i$")
plt.tight_layout()
plt.show()
