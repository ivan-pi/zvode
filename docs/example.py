"""
C The program below uses ZVODE to solve the following system of 2 ODEs:
C dw/dt = -i*w*w*z, dz/dt = i*z; w(0) = 1/2.1, z(0) = 1; t = 0 to 2*pi.
C Solution: w = 1/(z + 1.1), z = exp(it).  As z traces the unit circle,
C w traces a circle of radius 10/2.1 with center at 11/2.1.
C For convenience, Main passes RPAR = (imaginary unit i) to FEX and JEX.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from zvode import ZVODE

def fun(t,y,rpar):
    ydot = np.empty_like(y)
    ydot[0] = -rpar*y[0]**2*y[1]
    ydot[1] = rpar*y[1]
    return ydot

# Only set the non-zero values
def jac(t,y,rpar):
    J = np.zeros((2,2),dtype=np.complex128)
    J[0,0] = -2.0 * rpar * y[0] * y[1]
    J[0,1] = -rpar * y[0]**2
    J[1,1] = rpar
    return J

t0 = 0.0
y0 = np.array([1.0/2.1,1.0],dtype=np.complex128)

dtout = 0.1570796326794896
t_eval = dtout*np.arange(1,41)

rtol = 1.0e-9
atol = 1.0e-8

rpar = complex(0.0,1.0)

sol = solve_ivp(fun,(t0,t_eval[-1]),y0,
    method=ZVODE,
    args=(rpar,),
    rtol=1.0e-9,
    atol=1.0e-8,
    jac=jac)

print(sol)
print(f'No. f-s = {sol.nfev}, No. J-s = {sol.njev}, No. LU-s = {sol.nlu}')


# Extract w and z for clarity
w = sol.y[0]
z = sol.y[1]

# Quick loop to match Fortran console output format
print("\n   t           w                          z")
for i in range(len(sol.t)):
    print(f"{sol.t[i]:8.5f}     {w[i].real:9.7f}  {w[i].imag:9.7f}      {z[i].real:9.7f}  {z[i].imag:9.7f}")

# ---------------------------------------------------------
# 2. Plotting in the Complex Plane
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: Trajectory of w
plt.subplot(1, 2, 1)
plt.plot(np.real(w), np.imag(w), 'bo-', label='$w$ trajectory')
plt.plot(11/2.1, 0, 'rx', label='Expected Center') # Mark the expected center
plt.axis('equal')       # <--- CRITICAL: Forces 1:1 aspect ratio
plt.title("Complex Plane Trajectory of $w$")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.grid(True, alpha=0.5)

# Subplot 2: Trajectory of z
plt.subplot(1, 2, 2)
plt.plot(np.real(z), np.imag(z), 'gv-', label='$z$ trajectory')
plt.plot(0, 0, 'rx', label='Expected Center')
plt.axis('equal')       # <--- CRITICAL: Forces 1:1 aspect ratio
plt.title("Complex Plane Trajectory of $z$")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

