import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from manim import *

"""bools"""
boolPhasePlots = 0
boolManim = 0

q1 = 1
q2 = -1
m1 = 1
m2 = 1

FACTOR1 = 1 / (4*np.pi)
FACTOR2 = 1 / (4*np.pi) * q2/q1 * m1/m2

def sys(t, coords: tuple[NDArray]):
    x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = coords
    d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    dpow = d**(3/2)

    x1ddot = y1dot + FACTOR1 * (x1 - x2)/dpow
    y1ddot = -x1dot + FACTOR1 * (y1 - y2)/dpow
    z1ddot = FACTOR1 * (z1 - z2)/dpow

    x2ddot = FACTOR2 * 4*np.pi * y2dot + FACTOR2 * (x1 - x2)/dpow
    y2ddot = -FACTOR2 * 4*np.pi * x2dot + FACTOR2 * (y1 - y2)/dpow
    z2ddot = FACTOR2 * (z1 - z2)/dpow

    return (
        x1dot, y1dot, z1dot,
        x1ddot, y1ddot, z1ddot,
        x2dot, y2dot, z2dot,
        x2ddot, y2ddot, z2ddot,
    )   

t_max = 10

"""
x1, y1, z1
x1dot, y1dot, z1dot
x2, y2, z2
x2dot, y2dot, z2dot
"""
initial_condition = (
    0.5,-0.5,0,
    -1,1,0.1,
    -0.5,0.5,0,
    1,-1,-0.1
)

if not boolPhasePlots:

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

    # plt.show()

else:
    
    t_eval = np.arange(0,t_max, 4 * np.pi)

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, t_eval=t_eval, atol=1e-12, rtol = 1e-10)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)

    ax[0,0].plot(sol.y[0],sol.y[3])
    ax[0,0].set_xlabel("$x_1$")
    ax[0,0].set_ylabel("$\dot{x_1}$")

    ax[0,1].plot(sol.y[1],sol.y[4])
    ax[0,1].set_xlabel("$y_1$")
    ax[0,1].set_ylabel("$\dot{y_1}$")

    ax[0,2].plot(sol.y[2],sol.y[5])
    ax[0,2].set_xlabel("$z_1$")
    ax[0,2].set_ylabel("$\dot{z_1}$")

    ax[1,0].plot(sol.y[6],sol.y[9])
    ax[1,0].set_xlabel("$x_2$")
    ax[1,0].set_ylabel("$\dot{x_2}$")

    ax[1,1].plot(sol.y[7],sol.y[10])
    ax[1,1].set_xlabel("$y_2$")
    ax[1,1].set_ylabel("$\dot{y_2}$")

    ax[1,2].plot(sol.y[8],sol.y[11])
    ax[1,2].set_xlabel("$z_2$")
    ax[1,2].set_ylabel("$\dot{z_2}$")

    # plt.show()

class DoubleParticle(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(phi=60*DEGREES, theta=30*DEGREES)

        videoScaleFactor = 1
        timeScaleFactor = 1/10
        resolution = 1

        sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=1e-12, rtol = 1e-10)

        t = sol.t
        x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = sol.y

        ax = ThreeDAxes()

        print(len(t))
    
        particle1 = Sphere(color=PURE_GREEN, radius=DEFAULT_DOT_RADIUS, resolution=(2,2)).move_to(ax.c2p(x1[0], y1[0], z1[0]))
        particle2 = Sphere(color=PURPLE ,radius=DEFAULT_DOT_RADIUS, resolution=(2,2)).move_to(ax.c2p(x2[0], y2[0], z2[0]))
        self.add(particle1, particle2, ax)
        self.wait(1)
        for i in range(len(t)-1):
            self.play(AnimationGroup(
                particle1.animate.move_to(ax.c2p(x1[i+1], y1[i+1], z1[i+1])),
                particle2.animate.move_to(ax.c2p(x2[i+1], y2[i+1], z2[i+1])),
            ), run_time = (t[i+1]-t[i])*timeScaleFactor, rate_func=linear)
        self.wait(1)

