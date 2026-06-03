"""Benchmark problem definitions shared across benchmark modules."""

import numpy as np


def make_quantum_chain(n: int, t_end: float = 20.0):
    """
    Complex tight-binding chain: dy/dt = -i H y, H symmetric tridiagonal.

    H_jj = omega_j (site energies, uniform in [1, 2])
    H_{j,j±1} = kappa (nearest-neighbour hopping)

    Linear, non-stiff, purely oscillatory — the natural use case for ZVODE.
    The Jacobian is constant and tridiagonal (lband=uband=1), making it an
    ideal showcase for banded-Jacobian speedup as n grows.

    Returns
    -------
    fun        : RHS f(t, y), y complex (n,)
    jac_dense  : Dense Jacobian (n, n) complex
    jac_banded : Banded Jacobian (3, n) complex, lband=uband=1
    y0         : Initial state — unit excitation on site 0
    t_span     : (0, t_end)
    """
    omega = np.linspace(1.0, 2.0, n)
    kappa = 0.1

    def fun(t, y):
        dydt = -1j * omega * y
        dydt[:-1] -= 1j * kappa * y[1:]
        dydt[1:] -= 1j * kappa * y[:-1]
        return dydt

    def jac_dense(t, y):
        J = np.diag(-1j * omega).copy()
        J[np.arange(n - 1), np.arange(1, n)] = -1j * kappa
        J[np.arange(1, n), np.arange(n - 1)] = -1j * kappa
        return J

    def jac_banded(t, y):
        # ZVODE band storage: pd[i - j + uband, j] = J[i, j]
        # lband=uband=1 → shape (3, n)
        pd = np.zeros((3, n), dtype=np.complex128)
        pd[1, :] = -1j * omega       # diagonal J[j, j]
        pd[0, 1:] = -1j * kappa     # superdiagonal J[j-1, j]
        pd[2, :-1] = -1j * kappa    # subdiagonal   J[j+1, j]
        return pd

    y0 = np.zeros(n, dtype=np.complex128)
    y0[0] = 1.0
    return fun, jac_dense, jac_banded, y0, (0.0, t_end)


def make_decaying_oscillators(n: int, t_end: float = 5.0):
    """
    Independent decaying oscillators with a 100× spread in decay rates.

    dy_j/dt = (-gamma_j - i) * y_j,  gamma_j ∈ [1, 100] (log-spaced)

    Stiffness ratio ≈ 100.  Diagonal Jacobian → good for miter=3.
    Adams fails here because it needs tiny steps to track the fast modes;
    BDF handles the stiffness with large steps.
    """
    gamma = np.logspace(0, 2, n)   # decay rates: 1 … 100

    def fun(t, y):
        return (-gamma - 1j) * y

    def jac_dense(t, y):
        return np.diag(-gamma - 1j)

    y0 = np.ones(n, dtype=np.complex128)
    return fun, jac_dense, y0, (0.0, t_end)


def rober_problem(complex_y0: bool = False):
    """
    Robertson's chemical kinetics — the standard stiff ODE benchmark.

    Three species A→B, B+B→C, B+C→A+C with rate constants 0.04, 3e7, 1e4.
    Stiffness ratio ~1e9.  Real-valued; pass complex_y0=True for ZVODE
    (imaginary parts remain zero throughout).

    Returns
    -------
    fun   : RHS f(t, y)
    jac   : Dense Jacobian (3, 3)
    y0    : Initial condition [1, 0, 0], real or complex depending on flag
    t_span: (0, 1e4)
    """
    def fun(t, y):
        return np.array([
            -0.04 * y[0] + 1.0e4 * y[1] * y[2],
             0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1]**2,
             3.0e7 * y[1]**2,
        ])

    def jac(t, y):
        return np.array([
            [-0.04,           1.0e4 * y[2],                1.0e4 * y[1]],
            [ 0.04, -1.0e4 * y[2] - 6.0e7 * y[1],        -1.0e4 * y[1]],
            [ 0.0,              6.0e7 * y[1],                       0.0],
        ])

    dtype = complex if complex_y0 else float
    y0 = np.array([1.0, 0.0, 0.0], dtype=dtype)
    return fun, jac, y0, (0.0, 1e4)
