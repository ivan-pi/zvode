"""
Demo: banded Jacobian for a coupled complex chain system.

We solve  y' = A y  where  A  is a constant complex n x n matrix with
lower bandwidth  lband = 2  and upper bandwidth  uband = 1  (n = 5).

Full Jacobian (symbolic):

    a  d  .  .  .
    b  a  d  .  .
    c  b  a  d  .
    .  c  b  a  d
    .  .  c  b  a

    a = alpha = -1+2j   main diagonal
    b = beta  = 0.5     first  subdiagonal  (lband >= 1)
    c = gamma = 0.25    second subdiagonal  (lband >= 2)
    d = delta = 0.5j    superdiagonal       (uband >= 1)
    . = 0

ZVODE stores the banded Jacobian in a compact array  pd  of shape
(lband + uband + 1, n).  Each column of  pd  corresponds to a column
of  J, keeping only the entries that lie within the band:

    pd[i - j + uband, j]  =  J[i, j]

For this problem (lband=2, uband=1) the storage array has shape (4, 5):

    *  d  d  d  d       row 0: J[j-1, j] = delta   (superdiagonal)
    a  a  a  a  a       row 1: J[j,   j] = alpha   (main diagonal)
    b  b  b  b  *       row 2: J[j+1, j] = beta    (first subdiagonal)
    c  c  c  *  *       row 3: J[j+2, j] = gamma   (second subdiagonal)

    * = unused (can be any value, typically 0)

The exact solution  y(t) = expm(A t) y0  is used to verify accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from zvode import ZVODE

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------

n = 5
alpha = -1.0 + 2j    # main diagonal
beta  =  0.5 + 0j    # first subdiagonal   (lband = 1)
gamma =  0.25 + 0j   # second subdiagonal  (lband = 2)
delta =  0.0 + 0.5j  # superdiagonal       (uband = 1)

lband = 2
uband = 1

# ---------------------------------------------------------------------------
# Build the full (dense) matrix A – only needed for the exact reference
# ---------------------------------------------------------------------------

A = np.diag([alpha] * n)
A += np.diag([delta] * (n - 1),  1)   # superdiagonal
A += np.diag([beta]  * (n - 1), -1)   # first subdiagonal
A += np.diag([gamma] * (n - 2), -2)   # second subdiagonal

# ---------------------------------------------------------------------------
# Right-hand side  f(t, y) = A y  (vectorized, no temporary matrix multiply)
# ---------------------------------------------------------------------------

def fun(t, y):
    dy = alpha * y
    dy[:-1] += delta * y[1:]    # coupling from y_{k+1}
    dy[1:]  += beta  * y[:-1]   # coupling from y_{k-1}
    dy[2:]  += gamma * y[:-2]   # coupling from y_{k-2}
    return dy

# ---------------------------------------------------------------------------
# Banded Jacobian:  return pd of shape (lband + uband + 1, n)
#
#   pd[i - j + uband, j]  =  dF_i / dy_j
#
# For this problem J = A is constant, so pd does not depend on t or y.
# ---------------------------------------------------------------------------

def jac_banded(t, y):
    pd = np.zeros((lband + uband + 1, n), dtype=complex)
    pd[0, 1:]  = delta    # J[j-1, j] = delta,  j = 1 .. n-1  (superdiag)
    pd[1, :]   = alpha    # J[j,   j] = alpha,  j = 0 .. n-1  (main diag)
    pd[2, :-1] = beta     # J[j+1, j] = beta,   j = 0 .. n-2  (subdiag 1)
    pd[3, :-2] = gamma    # J[j+2, j] = gamma,  j = 0 .. n-3  (subdiag 2)
    return pd

# ---------------------------------------------------------------------------
# Initial conditions and output grid
# ---------------------------------------------------------------------------

y0 = np.array([1.0, 0.5 + 0.5j, -1j, 0.25, 1.0 - 0.5j], dtype=complex)

t_span = (0.0, 4.0)
t_eval = np.linspace(0.0, 4.0, 401)

# ---------------------------------------------------------------------------
# Integrate with ZVODE (BDF, user-supplied banded Jacobian)
# ---------------------------------------------------------------------------

sol = solve_ivp(
    fun,
    t_span,
    y0,
    method=ZVODE,
    jac=jac_banded,
    lband=lband,
    uband=uband,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)

# ---------------------------------------------------------------------------
# Exact solution via matrix exponential
# ---------------------------------------------------------------------------

y_exact = np.array([expm(A * t) @ y0 for t in t_eval]).T

max_err = np.max(np.abs(sol.y - y_exact))
print(f"nsteps={sol.t.size - 1}, nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")
print(f"Max absolute error vs expm: {max_err:.2e}")

# ---------------------------------------------------------------------------
# Plot: real and imaginary parts of each component
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n), sharex=True)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for k in range(n):
    c = colors[k % len(colors)]
    for col, attr, label in [(0, "real", "Re"), (1, "imag", "Im")]:
        ax = axes[k, col]
        ax.plot(t_eval, getattr(sol.y[k], attr), color=c, label="ZVODE")
        ax.plot(t_eval, getattr(y_exact[k], attr), "--", color="k", lw=0.8,
                label="expm", alpha=0.6)
        ax.set_ylabel(rf"{label} $y_{k}$")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)

for col, title in [(0, "Real part"), (1, "Imaginary part")]:
    axes[-1, col].set_xlabel("$t$")
    axes[0, col].set_title(title)

plt.suptitle(
    r"$y' = Ay$,  banded $A$ with $\ell_b = 2$, $u_b = 1$  ($n = 5$)",
    fontsize=12,
)
plt.tight_layout()
plt.show()
