"""pybdf -- a small variable-order BDF integrator for stiff ODE systems.

A Fortran port of scipy.integrate.BDF (dense/banded Jacobians, user-supplied
or finite-difference, LAPACK linear algebra, Jacobian reuse) wrapped in a
friendly NumPy interface.
"""

from .solver import BDF, BdfResult, solve_bdf

__all__ = ["BDF", "BdfResult", "solve_bdf"]
__version__ = "0.1.0"
