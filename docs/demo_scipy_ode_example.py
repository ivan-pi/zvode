"""
Demo: SciPy ode docs example solved with ZVODE.

Problem from the bottom of:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

    dy[0]/dt = 1j*arg1*y[0] + y[1]
    dy[1]/dt = -arg1*y[1]^2

    y[0](0) = 1j,  y[1](0) = 2.0,  arg1 = 2.0,  t in [0, 10]

The script first reproduces the step-by-step integration loop shown in the
SciPy docs using scipy.integrate.ode, then solves the same problem with the
ZVODE wrapper via solve_ivp, and shows that the two match.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, solve_ivp

from zvode import ZVODE


def f(t, y, arg1):
    return [1j * arg1 * y[0] + y[1], -arg1 * y[1] ** 2]


def jac(t, y, arg1):
    return [[1j * arg1, 1], [0, -arg1 * 2 * y[1]]]


arg1 = 2.0
y0 = np.array([1.0j, 2.0], dtype=np.complex128)
t0, t1 = 0.0, 10.0

# ---- Reproduce the SciPy docs step-by-step loop -------------------------

r = (
    ode(f, jac)
    .set_integrator("zvode", method="bdf", with_jacobian=True, rtol=1e-8, atol=1e-10)
    .set_initial_value(y0, t0)
    .set_f_params(arg1)
    .set_jac_params(arg1)
)

print("scipy.integrate.ode  (step-by-step, reproducing the SciPy docs example):")
print(f"{'t':>4}   {'y[0]':^26}   {'y[1]':^26}")

t_ode = [t0]
y_ode = [y0.copy()]
dt = 1.0
while r.successful() and r.t < t1 - 0.5 * dt:
    r.integrate(r.t + dt)
    print(
        f"{r.t:4g}   "
        f"{r.y[0].real:+.6f}{r.y[0].imag:+.6f}j   "
        f"{r.y[1].real:+.6f}{r.y[1].imag:+.6f}j"
    )
    t_ode.append(r.t)
    y_ode.append(r.y.copy())

t_ode = np.array(t_ode)
y_ode = np.array(y_ode).T  # shape (2, n_steps)

# ---- Same problem via ZVODE + solve_ivp ---------------------------------

t_eval = np.linspace(t0, t1, 201)

sol = solve_ivp(
    f,
    (t0, t1),
    y0,
    method=ZVODE,
    args=(arg1,),
    jac=jac,
    miter=1,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)

print(f"\nZVODE (solve_ivp):  nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

# ---- Cross-check: max difference at the integer-step times ---------------

# Interpolate the dense solve_ivp output at the coarse scipy-ode time points.
y0_at_ode = np.interp(t_ode, sol.t, sol.y[0].real) + 1j * np.interp(
    t_ode, sol.t, sol.y[0].imag
)
y1_at_ode = np.interp(t_ode, sol.t, sol.y[1].real) + 1j * np.interp(
    t_ode, sol.t, sol.y[1].imag
)

max_err0 = np.max(np.abs(y0_at_ode - y_ode[0]))
max_err1 = np.max(np.abs(y1_at_ode - y_ode[1]))
print(f"Max |ZVODE - scipy ode|:  y[0]: {max_err0:.2e},  y[1]: {max_err1:.2e}")

# ---- Plot ---------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for i in range(2):
    axes[i, 0].plot(sol.t, sol.y[i].real, label="ZVODE (solve_ivp)")
    axes[i, 0].plot(t_ode, y_ode[i].real, "o", ms=5, label="scipy ode (zvode)")
    axes[i, 0].set_title(f"y[{i}].real")
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.4)

    axes[i, 1].plot(sol.t, sol.y[i].imag, label="ZVODE (solve_ivp)")
    axes[i, 1].plot(t_ode, y_ode[i].imag, "o", ms=5, label="scipy ode (zvode)")
    axes[i, 1].set_title(f"y[{i}].imag")
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.4)

for ax in axes[-1]:
    ax.set_xlabel("t")

plt.suptitle(
    "SciPy ode docs example  "
    r"($\dot{y}_0 = i\alpha y_0 + y_1,\;$"
    r"$\dot{y}_1 = -\alpha y_1^2,\;$"
    r"$\alpha = 2$)"
)
plt.tight_layout()
plt.show()
