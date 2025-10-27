import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Avec = -B_0y xhat
# Bvec = B_0 zhat

CONSTANT = 1

# -----------------------------------------------
# All in terms on non-dimensionalized quantities
# -----------------------------------------------
def sys(t, coords: tuple[NDArray]) -> tuple[NDArray, NDArray, NDArray]:
    """takes in [x,y,z,xdot,ydot,zdot]"""
    """returns [xdot,ydot,zdot,xddot,yddot,zddot]"""
    x, y, z, xdot, ydot, zdot = coords
    return [xdot, ydot, zdot, ydot, -xdot, 0]

sol = solve_ivp(sys, (0,50), y0=(0,0,0,1,1,2), dense_output = True, atol=1e-12, rtol = 1e-10)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')

ax.plot(sol.y[0], sol.y[1], zs=sol.y[2])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")


plt.show()



