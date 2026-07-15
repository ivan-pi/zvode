"""
Demo: 2-ODE complex system from the ZVODE source file (zvode.f).

    dw/dt = -i·w²·z,   dz/dt = i·z
    w(0) = 1/2.1,   z(0) = 1,   t ∈ [0, 2π]

Analytic solution: w = 1/(z + 1.1),  z = exp(i·t).
As z traces the unit circle, w traces a circle of radius 10/2.1
centered at 11/2.1.

Uses MF = 21 (BDF with user-supplied Jacobian). The imaginary unit i is
passed as an extra argument to demonstrate the `args` parameter of solve_ivp.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from zvode import ZVODE


def fun(t, y, rpar):
    ydot = np.empty_like(y)
    ydot[0] = -rpar * y[0] ** 2 * y[1]
    ydot[1] = rpar * y[1]
    return ydot


def jac(t, y, rpar):
    J = np.zeros((2, 2), dtype=np.complex128)
    J[0, 0] = -2.0 * rpar * y[0] * y[1]
    J[0, 1] = -rpar * y[0] ** 2
    J[1, 1] = rpar
    return J


t0 = 0.0
y0 = np.array([1.0 / 2.1, 1.0], dtype=np.complex128)
rpar = complex(0.0, 1.0)

dtout = 0.1570796326794896
t_eval = dtout * np.arange(1, 41)

sol = solve_ivp(
    fun,
    (t0, t_eval[-1]),
    y0,
    method=ZVODE,
    t_eval=t_eval,
    args=(rpar,),
    rtol=1.0e-9,
    atol=1.0e-8,
    jac=jac,
    miter=1,
)

w = sol.y[0]
z = sol.y[1]

print(f"No. f-s = {sol.nfev}, No. J-s = {sol.njev}, No. LU-s = {sol.nlu}")
print("\n   t           w                          z")
for i in range(len(sol.t)):
    print(
        f"{sol.t[i]:8.5f}     {w[i].real:9.7f}  {w[i].imag:9.7f}      {z[i].real:9.7f}  {z[i].imag:9.7f}"
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(w.real, w.imag, "bo-", label="$w$ trajectory")
ax1.plot(11 / 2.1, 0, "rx", label="Expected center")
ax1.set_aspect("equal")
ax1.set_title("Complex plane trajectory of $w$")
ax1.set_xlabel("Real part")
ax1.set_ylabel("Imaginary part")
ax1.legend()
ax1.grid(True, alpha=0.5)

ax2.plot(z.real, z.imag, "gv-", label="$z$ trajectory")
ax2.plot(0, 0, "rx", label="Expected center")
ax2.set_aspect("equal")
ax2.set_title("Complex plane trajectory of $z$")
ax2.set_xlabel("Real part")
ax2.set_ylabel("Imaginary part")
ax2.legend()
ax2.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()
