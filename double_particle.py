import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


c = 1# 3e8
q = 1# 1.6e-19
B_0 = 100

FACTOR =  5.4134 / (B_0 ** 0.5) # e-18

-
def sys(t, coords: tuple[NDArray]):
    x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = coords
    d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2

    x1ddot = y1dot / c - FACTOR  * (x2 - x1) / (d**1.5)
    y1ddot = -x1dot / c - FACTOR  * (y2 - y1) / (d**1.5)
    z1ddot = -FACTOR * (z2 - z1) / (d**1.5)

    x2ddot = y2dot / c - FACTOR  * (x1 - x2) / (d**1.5)
    y2ddot = -x2dot / c - FACTOR * (y1 - y2) / (d**1.5)
    z2ddot = -FACTOR * (z1 - z2) / (d**1.5)

    return (
        x1dot, y1dot, z1dot,
        x1ddot, y1ddot, z1ddot,
        x2dot, y2dot, z2dot,
        x2ddot, y2ddot, z2ddot,
    )   

t_max = 1000

t_eval = np.arange(0,t_max, 4 * np.pi)

initial_condition = (
    1,1,0,
    1,1,1,
    0,0,0,
    1,1,1
)

sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=1e-12, rtol = 1e-10)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect(aspect=(1,1,5))

print(sol.y.shape)

ax.plot(sol.y[0], sol.y[1], zs=sol.y[2], c='blue',alpha=0.5)
ax.plot(sol.y[6], sol.y[7], zs=sol.y[8], c='red',alpha=0.5)


ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")


plt.show()