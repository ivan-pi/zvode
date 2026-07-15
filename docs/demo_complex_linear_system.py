"""
Demo: complex linear system y' = A·y, two ways.

    y'(t) = A·y(t),   t ∈ [0, 25]

Two formulations are solved:

1. y as a 3-component state vector.
2. Y as a 3×3 matrix (flattened to 9 components), with all columns
   evolving simultaneously as independent IVPs.

Exact solution in both cases: y(t) = expm(A·t) · y0  (matrix exponential).

For the matrix IVP the Jacobian of flatten(A·Y) w.r.t. flatten(Y)
(row-major) is the 9×9 Kronecker product A ⊗ I₃.

The matrix A and initial conditions are adapted from the SciPy solve_ivp docs:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from zvode import ZVODE

A = np.array(
    [
        [-0.25 + 0.14j, 0, 0.33 + 0.44j],
        [0.25 + 0.58j, -0.2 + 0.14j, 0],
        [0, 0.2 + 0.4j, -0.1 + 0.97j],
    ]
)

t_span = (0.0, 25.0)
t_eval = np.linspace(0.0, 25.0, 101)

# ---- Case 1: 3-component vector IVP ------------------------------------

y0_vec = np.array([10 + 0j, 20 + 0j, 30 + 0j])


def deriv_vec(t, y):
    return A @ y


sol_vec = solve_ivp(
    deriv_vec,
    t_span,
    y0_vec,
    method=ZVODE,
    jac=lambda t, y: A,
    miter=1,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

print("Vector IVP")
print("  y[:, 0] :", sol_vec.y[:, 0])
print("  y[:, -1]:", sol_vec.y[:, -1])

ref_vec = np.column_stack([expm(A * t) @ y0_vec for t in t_eval])
err_vec = np.max(np.abs(sol_vec.y - ref_vec))
print(f"  Max absolute error vs expm: {err_vec:.2e}\n")

# ---- Case 2: 3x3 matrix IVP (flattened to 9 components) ---------------

y0_mat = np.array(
    [
        [2 + 0j, 3 + 0j, 4 + 0j],
        [5 + 0j, 6 + 0j, 7 + 0j],
        [9 + 0j, 34 + 0j, 78 + 0j],
    ]
)
J_mat = np.kron(A, np.eye(3))  # Jacobian of flatten(A @ Y) w.r.t. flatten(Y)


def deriv_mat(t, y):
    return (A @ y.reshape(3, 3)).flatten()


sol_mat = solve_ivp(
    deriv_mat,
    t_span,
    y0_mat.flatten(),
    method=ZVODE,
    jac=lambda t, y: J_mat,
    miter=1,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

print("Matrix IVP")
print("  y[:, 0].reshape(3, 3):\n", sol_mat.y[:, 0].reshape(3, 3))
print("  y[:, -1].reshape(3, 3):\n", sol_mat.y[:, -1].reshape(3, 3))

ref_mat = np.column_stack([(expm(A * t) @ y0_mat).flatten() for t in t_eval])
err_mat = np.max(np.abs(sol_mat.y - ref_mat))
print(f"  Max absolute error vs expm: {err_mat:.2e}")

# ---- Plot --------------------------------------------------------------

# Figure 1: vector IVP — all 3 components together
fig1, ax1 = plt.subplots(figsize=(8, 5))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k in range(3):
    c = colors[k]
    ax1.plot(t_eval, sol_vec.y[k].real, color=c, label=f"$y_{k}$ re")
    ax1.plot(t_eval, sol_vec.y[k].imag, "--", color=c, label=f"$y_{k}$ im")
ax1.set_title(r"Vector IVP — $y' = Ay$, $y_0 = [10, 20, 30]^T$")
ax1.set_xlabel("$t$")
ax1.legend(ncol=3)
ax1.grid(True, alpha=0.4)
fig1.tight_layout()

# Figure 2: matrix IVP — all 9 components together
fig2, ax2 = plt.subplots(figsize=(10, 6))
y_mat = sol_mat.y.reshape(3, 3, -1)  # shape (row, col, n_t)
color_idx = 0
for row in range(3):
    for col in range(3):
        c = colors[color_idx % len(colors)]
        ax2.plot(t_eval, y_mat[row, col].real, color=c, label=f"$Y_{{{row},{col}}}$ re")
        ax2.plot(
            t_eval, y_mat[row, col].imag, "--", color=c, label=f"$Y_{{{row},{col}}}$ im"
        )
        color_idx += 1
ax2.set_title(r"Matrix IVP — $Y' = AY$, $Y_0 = [[2,3,4],[5,6,7],[9,34,78]]$")
ax2.set_xlabel("$t$")
ax2.legend(ncol=3, fontsize=8)
ax2.grid(True, alpha=0.4)
fig2.tight_layout()

plt.show()
