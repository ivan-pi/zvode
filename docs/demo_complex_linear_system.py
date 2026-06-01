"""
Demo: complex linear system y' = A*y from the SciPy solve_ivp docs.

Two cases from:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

1. y as a 3-component state vector
2. y as a flattened 3x3 matrix (each column evolves as an independent IVP)

Exact solution in both cases: y(t) = expm(A*t) @ y0  (matrix exponential).

For the flattened matrix IVP the Jacobian of flatten(A @ Y) with respect to
flatten(Y) (row-major) is the 9x9 Kronecker product  A ⊗ I_3.
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

ref_mat = np.column_stack(
    [(expm(A * t) @ y0_mat).flatten() for t in t_eval]
)
err_mat = np.max(np.abs(sol_mat.y - ref_mat))
print(f"  Max absolute error vs expm: {err_mat:.2e}")

# ---- Plot --------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Top row: one subplot per component of the vector IVP
for k in range(3):
    axes[0, k].plot(t_eval, sol_vec.y[k].real, label="real")
    axes[0, k].plot(t_eval, sol_vec.y[k].imag, "--", label="imag")
    axes[0, k].set_title(f"Vector IVP — $y_{k}$")
    axes[0, k].set_xlabel("$t$")
    axes[0, k].legend()
    axes[0, k].grid(True, alpha=0.4)

# Bottom row: one subplot per row of Y (grouping the three columns together)
y_mat = sol_mat.y.reshape(3, 3, -1)  # shape (row, col, n_t)
for row in range(3):
    for col in range(3):
        axes[1, row].plot(t_eval, y_mat[row, col].real, label=f"$Y_{{{row},{col}}}$.re")
        axes[1, row].plot(t_eval, y_mat[row, col].imag, "--", label=f"$Y_{{{row},{col}}}$.im")
    axes[1, row].set_title(f"Matrix IVP — row {row}")
    axes[1, row].set_xlabel("$t$")
    axes[1, row].legend(fontsize=7)
    axes[1, row].grid(True, alpha=0.4)

plt.suptitle(r"$y' = Ay$ with complex $3\times3$ matrix $A$")
plt.tight_layout()
plt.show()
