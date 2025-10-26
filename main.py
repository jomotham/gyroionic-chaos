from __future__ import annotations
from typing import Union, Literal

import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
import numpy as np

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Computer Modern Roman"
# })

# SPARC PARAMETERS
# - R_0: 1.85 m
# - r_0: 0.57 m
# - B strength: 12.2-20 T
# If N=100 coils, => |I| = 1.5e6, which gives 12 T < |B| < 20 T between 1.5 and 2.5 m

N_COILS = 100
I_STRENGTH = 1.5e5

CONSTANT_B_TERM = (4 * np.pi * 1e-7 * N_COILS * I_STRENGTH) / (2 * np.pi)

NUM_DIVISIONS = 100



def to_cartesian(theta, zeta, r, R0, s_theta=-1, s_zeta=+1):
    """Vectorized toroidal → Cartesian (simplified for plotting)."""
    R = R0 + r * np.cos(theta)
    x = R * np.cos(zeta)
    y = s_zeta * R * np.sin(zeta)
    z = s_theta * r * np.sin(theta)
    return x, y, z


def plot_torodial_field(R0: float, r0: float) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Mesh in theta-zeta space (fixed minor radius)
    theta = np.linspace(0, 2 * np.pi, NUM_DIVISIONS)
    zeta = np.linspace(0, 2 * np.pi, NUM_DIVISIONS)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta)

    x, y, z = to_cartesian(theta_grid, zeta_grid, r0, R0)

    dist_to_z = np.sqrt(x**2 + y**2)
    surface_B_field = CONSTANT_B_TERM / dist_to_z

    norm = colors.Normalize(vmin=np.min(surface_B_field), vmax=np.max(surface_B_field))
    facecolors = cm.viridis(norm(surface_B_field))

    ax.plot_surface(
        x, y, z, 
        cmap="viridis", 
        facecolors=facecolors, 
        edgecolor=(0,0,0,0.05), # black 5% opacity
        alpha=0.5
    )
    
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(surface_B_field)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="|$B$| (T)")


    # test_coords = np.array([[-2,-2,-0.5],[-1.5,-1.5,-0.375],[-1,-1,-0.25],[-0.5,-0.5,-0.125],[0,0,0],[1,2,0.5]])
    # test_x = [p[0] for p in test_coords]
    # test_y = [p[1] for p in test_coords]
    # test_z = [p[2] for p in test_coords]

    # ax.plot(test_x,test_y,test_z)


    ax.set_aspect("equal")
    ax.zaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_zlabel("$z$ (m)")
    ax.set_title(f"Toroidal Magnetic Field ($R_0={R0}$ m, $r_0={r0}$ m)")

    plt.show()


def is_in_torus(
    coord_1: float,
    coord_2: float,
    coord_3: float,
    R0: float,
    r0: float,
    coordinate_system: Union[Literal["cartesian"], Literal["toroidal"]] = "toroidal",
) -> bool:
    """
    Return True iff the point lies inside (or on) a torus with
    major radius R0 and tube radius r0, centered on the z-axis.

    Parameters
    ----------
    - If `coordinate_system == "toroidal"`: (coord_1, coord_2, coord_3) = (theta, zeta, r)
    - If `coordinate_system == "cartesian"`: (coord_1, coord_2, coord_3) = (x, y, z)

    Uses the right-handed convention s_theta * s_zeta = -1 (default s_theta=-1, s_zeta=+1).
    """
    if R0 <= 0:
        raise ValueError("R0 must be positive")
    if r0 < 0:
        raise ValueError("r0 must be nonnegative")

    system = coordinate_system.lower()
    if system == "toroidal":
        _, _, r = coord_1, coord_2, coord_3
        return r <= r0

    elif system == "cartesian":
        x, y, z = coord_1, coord_2, coord_3
        return (math.hypot(x, y) - R0) ** 2 + z**2 <= r0**2

    else:
        raise ValueError("coordinate_system must be 'cartesian' or 'toroidal'")


if __name__ == "__main__":
    plot_torodial_field(R0=1.85, r0=0.57)
