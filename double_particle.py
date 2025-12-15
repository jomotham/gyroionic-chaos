import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from manim import *
from manim.utils.file_ops import open_file as open_media_file
from random import uniform
import matplotlib as mpl

"""bools"""
boolPathPlot = 0
boolPositionPlots = 0
boolPhasePlots = 0
boolPoincarePlots = 0
boolManim = 1
# boolHamiltonian = 1        Hamiltonian isn't useful becuase Coulombic potential changes with particle displacement d.

q1 = 1
q2 = -1
m1 = 1
m2 = 1

A_TOL = 1e-12
R_TOL = 1e-10

FACTOR1 = q2/q1 / (4*np.pi)
FACTOR2 = q2/q1 * m1/m2 / (4*np.pi)

def sys(t, coords: tuple[NDArray]):
    x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = coords
    d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    dpow = d**(3/2) + A_TOL

    x1ddot = y1dot - FACTOR1 * (x2 - x1)/dpow
    y1ddot = -x1dot - FACTOR1 * (y2 - y1)/dpow
    z1ddot = -FACTOR1 * (z2 - z1)/dpow

    x2ddot = FACTOR2 * 4*np.pi * y2dot + FACTOR2 * (x2 - x1)/dpow
    y2ddot = -FACTOR2 * 4*np.pi * x2dot + FACTOR2 * (y2 - y1)/dpow
    z2ddot = FACTOR2 * (z2 - z1)/dpow

    return (
        x1dot, y1dot, z1dot,
        x1ddot, y1ddot, z1ddot,
        x2dot, y2dot, z2dot,
        x2ddot, y2ddot, z2ddot,
    )   

t_max = 100

"""
x1, y1, z1
x1dot, y1dot, z1dot
x2, y2, z2
x2dot, y2dot, z2dot
"""
initial_condition = (
    1,0,0.001,
    0,0,0,
    0,0,0,
    0,0,0
)

if boolPathPlot:

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect=(1,1,1))

    print("sol.y shape =", str(sol.y.shape))

    ax.plot(sol.y[0], sol.y[1], zs=sol.y[2], c='blue',alpha=0.5)
    ax.plot(sol.y[6], sol.y[7], zs=sol.y[8], c='red',alpha=0.5)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")

    plt.show()

if boolPositionPlots:

    # t_eval = np.arange(0, t_max, 1)                                                                         # For with t_eval

    # sol = solve_ivp(sys, (0,t_max), y0=initial_condition, t_eval=t_eval, atol=A_TOL, rtol = R_TOL)          # For with t_eval

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)    # For without t_eval
    
    print("sol.y shape =", str(sol.y.shape))

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)

    ax[0,0].plot(sol.t,sol.y[0])
    ax[0,0].set_xlabel(r"$t$")
    ax[0,0].set_ylabel(r"$x_1$")

    ax[0,1].plot(sol.t,sol.y[1])
    ax[0,1].set_xlabel(r"$t$")
    ax[0,1].set_ylabel(r"$y_1$")

    ax[0,2].plot(sol.t,sol.y[2])
    ax[0,2].set_xlabel(r"$t$")
    ax[0,2].set_ylabel(r"$z_1$")

    ax[1,0].plot(sol.t,sol.y[6])
    ax[1,0].set_xlabel(r"$t$")
    ax[1,0].set_ylabel(r"$x_2$")

    ax[1,1].plot(sol.t,sol.y[7])
    ax[1,1].set_xlabel(r"$t$")
    ax[1,1].set_ylabel(r"$y_2$")

    ax[1,2].plot(sol.t,sol.y[8])
    ax[1,2].set_xlabel(r"$t$")
    ax[1,2].set_ylabel(r"$z_2$")

    plt.show()

# if boolHamiltonian:
    
#     sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)
    
#     print("sol.y shape =", str(sol.y.shape))

#     """
#     x1, y1, z1
#     x1dot, y1dot, z1dot
#     x2, y2, z2
#     x2dot, y2dot, z2dot
#     """
#     hamiltonian_proportional = 1/2*(sol.y[3]**2 + sol.y[4]**2 + sol.y[5]**2 + sol.y[9]**2 + sol.y[10]**2 + sol.y[11]**2) + 1/(2*np.pi)*1/np.sqrt( (sol.y[0]-sol.y[6])**2 + (sol.y[1]-sol.y[7])**2 + (sol.y[2]-sol.y[8])**2) - sol.y[1]*sol.y[3] - sol.y[7]*sol.y[9] + sol.y[0]*sol.y[4] + sol.y[6]*sol.y[10]

#     fig, ax = plt.subplots()

#     ax.plot(sol.t, hamiltonian_proportional)

#     plt.show()

if boolPhasePlots:

    # t_eval = np.arange(0, t_max, 1)                                                                         # For with t_eval

    # sol = solve_ivp(sys, (0,t_max), y0=initial_condition, t_eval=t_eval, atol=A_TOL, rtol = R_TOL)          # For with t_eval

    sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)    # For without t_eval
    
    print("sol.y shape =", str(sol.y.shape))

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)

    ax[0,0].plot(sol.y[0],sol.y[3], ".")
    ax[0,0].set_xlabel(r"$x_1$")
    ax[0,0].set_ylabel(r"$\dot{x_1}$")

    ax[0,1].plot(sol.y[1],sol.y[4], ".")
    ax[0,1].set_xlabel(r"$y_1$")
    ax[0,1].set_ylabel(r"$\dot{y_1}$")

    ax[0,2].plot(sol.y[2],sol.y[5], ".")
    ax[0,2].set_xlabel(r"$z_1$")
    ax[0,2].set_ylabel(r"$\dot{z_1}$")

    ax[1,0].plot(sol.y[6],sol.y[9], ".")
    ax[1,0].set_xlabel(r"$x_2$")
    ax[1,0].set_ylabel(r"$\dot{x_2}$")

    ax[1,1].plot(sol.y[7],sol.y[10], ".")
    ax[1,1].set_xlabel(r"$y_2$")
    ax[1,1].set_ylabel(r"$\dot{y_2}$")

    ax[1,2].plot(sol.y[8],sol.y[11], ".")
    ax[1,2].set_xlabel(r"$z_2$")
    ax[1,2].set_ylabel(r"$\dot{z_2}$")

    plt.show()

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

if boolPoincarePlots:

    """
    x1, y1, z1
    x1dot, y1dot, z1dot
    x2, y2, z2
    x2dot, y2dot, z2dot
    """
    central_initial_condition =  (
        1.08385,0,0,
        0,0,0,
        0,0,0,
        0,0,0
    )
    """
    x1, y1, z1
    x1dot, y1dot, z1dot
    """
    variation_amplitudes = (
        0.0001,0,0,
        0,0,0,
    )            # if you want to leave either the positions or velocities unchanged, change the respective variation_amplitude to 0.
                 # Note that this is only for particle 1 because relative motion is really what's important
    resolution = 12   # NOTE: resolution**(nonzeroes in amplitudes) IS HOW MANY CONDITIONS YOU PLAN ON PLAYING. CONSIDER THIS BEFORE RUNNING THE CODE.
    color1 = "#D4CC47"
    color2 = "#7C4D8B"

    x1_vals = np.linspace(central_initial_condition[0] - variation_amplitudes[0], central_initial_condition[0] + variation_amplitudes[0], num = resolution)
    y1_vals = np.linspace(central_initial_condition[1] - variation_amplitudes[1], central_initial_condition[1] + variation_amplitudes[1], num = resolution)
    z1_vals = np.linspace(central_initial_condition[2] - variation_amplitudes[2], central_initial_condition[2] + variation_amplitudes[2], num = resolution)
    x2_vals = np.linspace(central_initial_condition[6], central_initial_condition[6], num = resolution)
    y2_vals = np.linspace(central_initial_condition[7], central_initial_condition[7], num = resolution)
    z2_vals = np.linspace(central_initial_condition[8], central_initial_condition[8], num = resolution)
    x1dot_vals = np.linspace(central_initial_condition[3] - variation_amplitudes[3], central_initial_condition[3] + variation_amplitudes[3], num = resolution)
    y1dot_vals = np.linspace(central_initial_condition[4] - variation_amplitudes[4], central_initial_condition[4] + variation_amplitudes[4], num = resolution)
    z1dot_vals = np.linspace(central_initial_condition[5] - variation_amplitudes[5], central_initial_condition[5] + variation_amplitudes[5], num = resolution)
    x2dot_vals = np.linspace(central_initial_condition[9], central_initial_condition[9], num = resolution)
    y2dot_vals = np.linspace(central_initial_condition[10], central_initial_condition[10], num = resolution)
    z2dot_vals = np.linspace(central_initial_condition[11], central_initial_condition[11], num = resolution)
    
    ICs = []
    counter = 0
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                for l in range(resolution):
                    for m in range(resolution):
                        for n in range(resolution):
                            if x1_vals[i] == x2_vals[i] and y1_vals[j] == y2_vals[j] and z1_vals[k] == z2_vals[k]:
                                counter += 1
                                print("IC creation counter == ", str(counter), ", ", str(resolution**6 - counter), " remaining.")
                                continue
                            IC = (
                                x1_vals[i], y1_vals[j], z1_vals[k],
                                x1dot_vals[l], y1dot_vals[m], z1dot_vals[n],
                                x2_vals[i], y2_vals[j], z2_vals[k],
                                x2dot_vals[l], y2dot_vals[m], z2dot_vals[n]
                            )
                            if IC in ICs:
                                counter += 1
                                print("IC creation counter == ", str(counter), ", ", str(resolution**6 - counter), " remaining.")
                                continue
                            ICs.append(IC)
                            counter += 1
                            print("IC creation counter == ", str(counter), ", ", str(resolution**6 - counter), " remaining.")

    print("created all initial conditions")

    sols = []
    t_eval = np.arange(0, t_max, 1)                                                                        
    counter = 0
    for IC in ICs:
        sol = solve_ivp(sys, (0,t_max), y0=IC, t_eval=t_eval, atol=A_TOL, rtol = R_TOL)
        sols.append(sol)
        counter += 1
        print("solutions counter == ", str(counter), ", ", str(len(ICs)-counter), " remaining.")

    print("graphing...")

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
    colors = get_color_gradient(color1, color2, len(ICs))
    counter = 0
    for i in range(len(sols)):
        ax[0,0].plot(sols[i].y[0],sols[i].y[3], ".", color=colors[i], linewidth = 1)
        ax[0,0].set_xlabel(r"$x_1$")
        ax[0,0].set_ylabel(r"$\dot{x_1}$")

        ax[0,1].plot(sols[i].y[1],sols[i].y[4], ".", color=colors[i], linewidth = 1)
        ax[0,1].set_xlabel(r"$y_1$")
        ax[0,1].set_ylabel(r"$\dot{y_1}$")

        ax[0,2].plot(sols[i].y[2],sols[i].y[5], ".", color=colors[i], linewidth = 1)
        ax[0,2].set_xlabel(r"$z_1$")
        ax[0,2].set_ylabel(r"$\dot{z_1}$")

        ax[1,0].plot(sols[i].y[6],sols[i].y[9], ".", color=colors[i], linewidth = 1)
        ax[1,0].set_xlabel(r"$x_2$")
        ax[1,0].set_ylabel(r"$\dot{x_2}$")

        ax[1,1].plot(sols[i].y[7],sols[i].y[10], ".", color=colors[i], linewidth = 1)
        ax[1,1].set_xlabel(r"$y_2$")
        ax[1,1].set_ylabel(r"$\dot{y_2}$")

        ax[1,2].plot(sols[i].y[8],sols[i].y[11], ".", color=colors[i], linewidth = 1)
        ax[1,2].set_xlabel(r"$z_2$")
        ax[1,2].set_ylabel(r"$\dot{z_2}$")
        counter += 1
        print("graphing counter == ", str(counter), ", ", str(len(ICs)-counter), " remaining.")

    print("showing plot...")
    plt.show()




    


class DoubleParticle(ThreeDScene):

    def construct(self):

        cameraZoomFactor = 1        ## zooms the camera in and out
        timeScaleFactor = 1/10       ## the smaller the number, the quicker the video time plays for. this will only affect the particle motion, not any self.wait() commands or other things
        resolution = 70             ## the larger the number, the less resolute it is. don't go below 1, I don't know what happens when you do, but it prob breaks

        # phi, theta, focal_distance, gamma, zoom = self.camera.get_value_trackers()
        self.set_camera_orientation(phi=60*DEGREES, theta=30*DEGREES)
        self.camera.set_zoom(cameraZoomFactor)

        sol = solve_ivp(sys, (0,t_max), y0=initial_condition, dense_output=True, atol=A_TOL, rtol = R_TOL)

        t = sol.t
        x1, y1, z1, x1dot, y1dot, z1dot, x2, y2, z2, x2dot, y2dot, z2dot = sol.y

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
            print("Number of Frames: " + str(len(t)//resolution))
        self.wait(1)

if boolManim:
    if __name__ == '__main__':
        scene = DoubleParticle()
        scene.render()

        open_media_file(scene.renderer.file_writer.movie_file_path)