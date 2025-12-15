import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from manim import *
from manim.utils.file_ops import open_file as open_media_file

# === Bools for what plots are generated === 
boolMotionPlot = False
boolPhasePlots = False
boolManim = True

# === Constants === 
q1 = 1
q2 = -1
m1 = 1
m2 = 1

A_TOL = 1e-13
R_TOL = 1e-11

FACTOR1 = q2/q1 / (4*np.pi)
FACTOR2 = q2/q1 * m1/m2 / (4*np.pi)

# === System === 
def sys(t, coords: tuple[NDArray]):
    x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = coords
    d = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    d_cubed = d**3

    x1ddot =  y1dot + FACTOR1 * (x1 - x2)/d_cubed
    y1ddot = -x1dot + FACTOR1 * (y1 - y2)/d_cubed
    z1ddot =          FACTOR1 * (z1 - z2)/d_cubed

    x2ddot =  FACTOR2 * 4*np.pi * y2dot + FACTOR2 * (x2 - x1)/d_cubed
    y2ddot = -FACTOR2 * 4*np.pi * x2dot + FACTOR2 * (y2 - y1)/d_cubed
    z2ddot =                              FACTOR2 * (z2 - z1)/d_cubed

    return (
        x1dot, y1dot, z1dot,
        x1ddot, y1ddot, z1ddot,
        x2dot, y2dot, z2dot,
        x2ddot, y2ddot, z2ddot,
    )   

# === Initial Conditions === 
"""
Initial Condition Format:
x1, y1, z1
x1dot, y1dot, z1dot
x2, y2, z2
x2dot, y2dot, z2dot
"""
initial_condition = (
    0.1,0,-0.5,
    0,0,0,
    -0.1,0,0.5,
    0,0,0
)


# === Simulations === 

t_max = 100

if boolMotionPlot:
    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect=(1,1,5))

    print("sol.y shape =", str(sol.y.shape))

    ax.plot(sol.y[0], sol.y[1], zs=sol.y[2], c='blue',alpha=0.5)
    ax.plot(sol.y[6], sol.y[7], zs=sol.y[8], c='red',alpha=0.5)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    plt.show()

if boolPhasePlots:
    t_eval = np.arange(0,t_max, 4 * np.pi)

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, t_eval=t_eval, atol=A_TOL, rtol = R_TOL)
    
    print("sol.y shape =", str(sol.y.shape))

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
        cameraZoomFactor = 1        ## zooms the camera in and out
        timeScaleFactor = 1/5       ## the smaller the number, the quicker the video time plays for
        resolution = 25             ## the larger the number, the less resolute it is. don't go below 1, I don't know what happens when you do, but it prob breaks

        self.set_camera_orientation(phi=60*DEGREES, theta=30*DEGREES)
        self.camera.set_zoom(cameraZoomFactor)

        sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

        t = sol.t
        x1, y1, z1, _, _, _, x2, y2, z2, _, _, _ = sol.y

        ax = ThreeDAxes()

        print("sol.y shape =", str(sol.y.shape))
        print("Number of Frames: " + str(len(t)//resolution))
    
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

if boolManim:
    if __name__ == '__main__':
        scene = DoubleParticle()
        scene.render()

        open_media_file(scene.renderer.file_writer.movie_file_path)