import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from manim import *

"""bools"""
boolPhasePlots = 1
boolManim = 0

q1 = 1
q2 = -1
m1 = 1
m2 = 1

A_TOL = 1e-13
R_TOL = 1e-11

FACTOR1 = q2/q1 / (4*np.pi)
FACTOR2 = q2/q1 * m1/m2 / (4*np.pi)

def sys(t, coords: tuple[NDArray]):
    x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = coords
    d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    dpow = d**(3/2)

    x1ddot = y1dot + FACTOR1 * (x1 - x2)/dpow
    y1ddot = -x1dot + FACTOR1 * (y1 - y2)/dpow
    z1ddot = FACTOR1 * (z1 - z2)/dpow

    x2ddot = FACTOR2 * 4*np.pi * y2dot + FACTOR2 * (x2 - x1)/dpow
    y2ddot = -FACTOR2 * 4*np.pi * x2dot + FACTOR2 * (y2 - y1)/dpow
    z2ddot = FACTOR2 * (z2 - z1)/dpow

    return (
        x1dot, y1dot, z1dot,
        x1ddot, y1ddot, z1ddot,
        x2dot, y2dot, z2dot,
        x2ddot, y2ddot, z2ddot,
    )   

t_max = 100 * 1.5

"""
x1, y1, z1
x1dot, y1dot, z1dot
x2, y2, z2
x2dot, y2dot, z2dot
"""
initial_condition = (
    0,0,1,
    1,0,0,
    0,0,0,
    0,0,0
)

if not boolPhasePlots:

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

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

else:
    t_eval = np.arange(0,t_max, 4 * np.pi)

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, t_eval=t_eval, atol=A_TOL, rtol = R_TOL)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)


    ax[0,0].plot(sol.y[0],sol.y[3])
    ax[0,0].set_xlabel(r"$x_1$")
    ax[0,0].set_ylabel(r"$\dot{x_1}$")

    ax[0,1].plot(sol.y[1],sol.y[4])
    ax[0,1].set_xlabel(r"$y_1$")
    ax[0,1].set_ylabel(r"$\dot{y_1}$")

    ax[0,2].plot(sol.y[2],sol.y[5])
    ax[0,2].set_xlabel(r"$z_1$")
    ax[0,2].set_ylabel(r"$\dot{z_1}$")

    ax[1,0].plot(sol.y[6],sol.y[9])
    ax[1,0].set_xlabel(r"$x_2$")
    ax[1,0].set_ylabel(r"$\dot{x_2}$")

    ax[1,1].plot(sol.y[7],sol.y[10])
    ax[1,1].set_xlabel(r"$y_2$")
    ax[1,1].set_ylabel(r"$\dot{y_2}$")

    ax[1,2].plot(sol.y[8],sol.y[11])
    ax[1,2].set_xlabel(r"$z_2$")
    ax[1,2].set_ylabel(r"$\dot{z_2}$")

    plt.show()

class DoubleParticle(ThreeDScene):

    def construct(self):

        # phi, theta, focal_distance, gamma, zoom = self.camera.get_value_trackers()
        self.set_camera_orientation(phi=60*DEGREES, theta=30*DEGREES)

        videoScaleFactor = 1
        timeScaleFactor = 1/10
        resolution = 25

        sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

        t = sol.t
        x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = sol.y

        ax = ThreeDAxes()

        print(len(t)//resolution)
    
        particle1 = Sphere(radius=DEFAULT_DOT_RADIUS, resolution=(2,2)).move_to(ax.c2p(x1[0], y1[0], z1[0])).set_color(PURE_GREEN)
        particle1trail = TracedPath(particle1.get_center, dissipating_time=None, stroke_color=GREEN_C, stroke_opacity=[1, 1])
        particle2 = Sphere(radius=DEFAULT_DOT_RADIUS, resolution=(2,2)).move_to(ax.c2p(x2[0], y2[0], z2[0])).set_color(PURPLE_C)
        particle2trail = TracedPath(particle2.get_center, dissipating_time=None, stroke_color=PURPLE_C, stroke_opacity=[1, 1])
        self.add(particle1, particle1trail, particle2, particle2trail, ax)
        self.wait(1)
        for i in range((len(t)-1)//resolution):
            self.play(AnimationGroup(
                particle1.animate.move_to(ax.c2p(x1[(i+1)*resolution], y1[(i+1)*resolution], z1[(i+1)*resolution])),
                particle2.animate.move_to(ax.c2p(x2[(i+1)*resolution], y2[(i+1)*resolution], z2[(i+1)*resolution])),
                self.camera._frame_center.animate.move_to(VGroup(particle1, particle2))
            ), run_time = (t[i+1]-t[i])*timeScaleFactor*resolution, rate_func=linear)
        self.wait(1)