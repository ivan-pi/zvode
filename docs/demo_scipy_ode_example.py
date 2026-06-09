"""
Demo: 2-ODE complex system from the SciPy ode docs, solved with ZVODE.

    dw/dt = iα·w + z,   dz/dt = -α·z²
    w(0) = i,   z(0) = 2,   t ∈ [0, 10],   α = 2

Problem taken from the bottom of:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

The script reproduces the step-by-step integration loop shown in the SciPy
docs using scipy.integrate.ode, then solves the same problem with the ZVODE
wrapper via solve_ivp, and cross-checks the two results via dense output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import ode, solve_ivp

from zvode import ZVODE


def f(t, y, alpha):
    w, z = y
    return [1j * alpha * w + z, -alpha * z**2]


def jac(t, y, alpha):
    w, z = y
    return [[1j * alpha, 1], [0, -alpha * 2 * z]]


alpha = 2.0
y0 = np.array([1.0j, 2.0], dtype=np.complex128)
t0, t1 = 0.0, 10.0

# ---- Reproduce the SciPy docs step-by-step loop -------------------------

r = (
    ode(f, jac)
    .set_integrator("zvode", method="bdf", with_jacobian=True, rtol=1e-8, atol=1e-10)
    .set_initial_value(y0, t0)
    .set_f_params(alpha)
    .set_jac_params(alpha)
)

print("scipy.integrate.ode  (step-by-step, reproducing the SciPy docs example):")
print(f"{'t':>4}   {'w':^26}   {'z':^26}")

_t_list = [t0]
_y_list = [y0.copy()]
dt = 1.0
while r.successful() and r.t < t1 - 0.5 * dt:
    r.integrate(r.t + dt)
    w, z = r.y
    print(f"{r.t:4g}   {w.real:+.6f}{w.imag:+.6f}j   {z.real:+.6f}{z.imag:+.6f}j")
    _t_list.append(r.t)
    _y_list.append(r.y.copy())

t_ode = np.array(_t_list)
y_ode = np.array(_y_list).T  # shape (2, n_steps)

# ---- Same problem via ZVODE + solve_ivp ---------------------------------

t_eval = np.linspace(t0, t1, 201)

sol = solve_ivp(
    f,
    (t0, t1),
    y0,
    method=ZVODE,
    args=(alpha,),
    jac=jac,
    miter=1,
    t_eval=t_eval,
    dense_output=True,
    rtol=1e-8,
    atol=1e-10,
)

print(f"\nZVODE (solve_ivp):  nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

# ---- Cross-check: max difference at the integer-step times ---------------

y_at_ode = sol.sol(t_ode)  # shape (2, n_steps), evaluated via dense output
max_err_w = np.max(np.abs(y_at_ode[0] - y_ode[0]))
max_err_z = np.max(np.abs(y_at_ode[1] - y_ode[1]))
print(f"Max |ZVODE - scipy ode|:  w: {max_err_w:.2e},  z: {max_err_z:.2e}")

# ---- Plot ---------------------------------------------------------------

# z(0) = 2 is real and z' = -α·z² preserves the real line, so z stays real.
# The complex-plane trajectory of z is therefore trivial (a segment on the
# real axis) and is omitted from the right panel.

w_num, z_num = sol.y[0], sol.y[1]
w_ode, z_ode = y_ode[0], y_ode[1]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Re w, Im w, and z (real) vs time
ax_t.plot(t_eval, w_num.real, color=colors[0], label=r"$\mathrm{Re}\,w$")
ax_t.plot(t_eval, w_num.imag, "--", color=colors[0], label=r"$\mathrm{Im}\,w$")
ax_t.plot(t_eval, z_num.real, color=colors[1], label=r"$z$  (real-valued)")
ax_t.plot(t_ode, w_ode.real, "o", ms=4, color=colors[0])
ax_t.plot(t_ode, w_ode.imag, "o", ms=4, color=colors[0])
ax_t.plot(t_ode, z_ode.real, "o", ms=4, color=colors[1])
proxy = Line2D([0], [0], color="gray", marker="o", ms=4, ls="none", label="scipy ode")
handles, labels = ax_t.get_legend_handles_labels()
ax_t.legend(handles=handles + [proxy], labels=labels + ["scipy ode"])
ax_t.set_xlabel("$t$")
ax_t.set_title("Components vs. time")
ax_t.grid(True, alpha=0.4)

# Right: trajectory of w in the complex plane
ax_c.plot(w_num.real, w_num.imag, color=colors[0], label=r"$w$ (ZVODE)")
ax_c.plot(w_ode.real, w_ode.imag, "o", ms=5, color=colors[0], label="scipy ode")
ax_c.plot(
    w_num.real[0], w_num.imag[0], "^", ms=8, color="tab:green", zorder=5, label="$t=0$"
)
ax_c.plot(
    w_num.real[-1], w_num.imag[-1], "s", ms=6, color="tab:red", zorder=5, label="$t=10$"
)
ax_c.set_xlabel(r"$\mathrm{Re}\,w$")
ax_c.set_ylabel(r"$\mathrm{Im}\,w$")
ax_c.set_title(r"Trajectory of $w$ in the complex plane")
ax_c.legend()
ax_c.grid(True, alpha=0.4)

plt.suptitle(
    r"$\dot{w} = i\alpha w + z,\quad \dot{z} = -\alpha z^2,\quad"
    r"\alpha = 2,\quad w(0) = i,\quad z(0) = 2$"
)
plt.tight_layout()
plt.show()
