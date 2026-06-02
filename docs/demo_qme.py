"""
Demo: Lindblad master equation for a driven, dissipative two-level system.

    dρ/dt = -i[H, ρ] + γ·(L·ρ·L† - ½{L†L, ρ})
    ρ(0) = |1⟩⟨1|,   t ∈ [0, 15]

Parameters: Rabi frequency ω = 2.0, spontaneous emission rate γ = 0.5.
The Hamiltonian H = ½ω·σₓ drives Rabi oscillations; the jump operator
L = |0⟩⟨1| models spontaneous decay to the ground state.

The 2×2 density matrix ρ is vectorized to a 4-component complex array to
interface with solve_ivp. The script plots the state populations (diagonal
elements) and coherences (off-diagonal elements) as functions of time.

Generated with the assistance of Google Gemini.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import your custom solver here
from zvode import ZVODE

# ---------------------------------------------------------
# 1. Define Physics Parameters and Operators
# ---------------------------------------------------------
omega = 2.0  # Rabi frequency (oscillation speed)
gamma = 0.5  # Spontaneous emission rate (decay speed)

# Pauli matrices and lowering operator
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sm = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1|

# Hamiltonian
H = 0.5 * omega * sigma_x


# ---------------------------------------------------------
# 2. Define the Lindblad Master Equation
# ---------------------------------------------------------
def lindblad_deriv(t, y_flat):
    # Reshape the 1D state vector back into a 2x2 density matrix
    rho = y_flat.reshape((2, 2))

    # Unitary evolution: -i * [H, rho]
    unitary = -1j * (H @ rho - rho @ H)

    # Dissipative evolution: gamma * (L * rho * L_dag - 0.5 * {L_dag * L, rho})
    L = sm
    L_dag = L.conj().T
    dissipator = gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))

    # Return flattened 1D array for the ODE solver
    return (unitary + dissipator).flatten()


# ---------------------------------------------------------
# 3. Setup Initial Conditions and Solve
# ---------------------------------------------------------
# Initial state: Pure excited state |1><1|
rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

sol = solve_ivp(
    fun=lindblad_deriv,
    t_span=(0.0, 15.0),
    y0=rho0.flatten(),
    method=ZVODE,
)

print(sol.message)

# ---------------------------------------------------------
# 4. Extract and Plot Results
# ---------------------------------------------------------
# sol.y shape is (4, N) where the rows correspond to the flattened 2x2 matrix
rho_00 = sol.y[0]  # Ground state population
rho_01 = sol.y[1]  # Coherence
rho_10 = sol.y[2]  # Coherence (conjugate)
rho_11 = sol.y[3]  # Excited state population

plt.figure(figsize=(10, 6))

# Plot Populations (Diagonal elements are real, so we take np.real)
plt.plot(
    sol.t,
    np.real(rho_11),
    label="Excited State Population ($\\rho_{11}$)",
    color="blue",
    linewidth=2,
)
plt.plot(
    sol.t,
    np.real(rho_00),
    label="Ground State Population ($\\rho_{00}$)",
    color="orange",
    linewidth=2,
)

# Plot Coherences (Off-diagonal elements are complex, so we plot Real and Imaginary parts separately)
plt.plot(
    sol.t,
    np.real(rho_01),
    "--",
    label="Coherence Real ($\\Re[\\rho_{01}]$)",
    color="green",
)
plt.plot(
    sol.t,
    np.imag(rho_01),
    ":",
    label="Coherence Imaginary ($\\Im[\\rho_{01}]$)",
    color="red",
)

plt.title(
    "Quantum Master Equation Dynamics\nRabi Oscillations with Spontaneous Emission"
)
plt.xlabel("Time")
plt.ylabel("Density Matrix Elements")
plt.ylim([-0.6, 1.1])
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
