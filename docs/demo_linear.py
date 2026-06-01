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

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(t_eval, y_ref.real, label="exact")
axes[0].plot(t_eval, y_num.real, "--", label="ZVODE")
axes[0].set_title("Real part")
axes[0].set_xlabel("t")
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].plot(t_eval, y_ref.imag, label="exact")
axes[1].plot(t_eval, y_num.imag, "--", label="ZVODE")
axes[1].set_title("Imaginary part")
axes[1].set_xlabel("t")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

axes[2].semilogy(t_eval, err)
axes[2].set_title("Absolute error  $|y_{num} - y_{exact}|$")
axes[2].set_xlabel("t")
axes[2].grid(True, alpha=0.4)

plt.suptitle(r"$y' = t\,y + 2i,\quad y(0) = 1+i$")
plt.tight_layout()
plt.show()
