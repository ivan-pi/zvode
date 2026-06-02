"""
Demo: minimal usage of ZVODE with solve_ivp.

    y'(t) = -i·y,   y(0) = 1,   t ∈ [0, 10]

Analytic solution: y(t) = exp(-i·t)  (counter-clockwise rotation on the
unit circle in the complex plane).
"""

import numpy as np
from scipy.integrate import solve_ivp
from zvode import ZVODE

print("Import complete; calling solve_ivp")

sol = solve_ivp(
    fun=lambda t, y: -1j * y,
    t_span=(0.0, 10.0),
    y0=np.array([1.0 + 0.0j], dtype=np.complex128),
    method=ZVODE,
    jac=lambda t, y: -1j,
)

print(sol)
