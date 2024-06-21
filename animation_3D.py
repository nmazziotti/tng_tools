# Created base code in ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

import argparse
from plot_distance import download_hdf5

font = {'family' : 'serif',
        'size'   : 14}

plt.rc('font', **font)

parser = argparse.ArgumentParser()

# Command line arguments 
# Ex: python animate.py -s TNG300-1
parser.add_argument("-s", "--sim", required=True, type=str, help="Select simulation from IllustrisTNG")


def animation(sim_name, sub1, sub2, snap_start):
    f1, f2 = download_hdf5(sim_name, sub1, sub2, snap_start)

    r1 = f1['SubhaloHalfmassRad'][:]
    r2 = f2['SubhaloHalfmassRad'][:]

    position1 = f1['SubhaloPos'][:]
    x1 = []
    y1 = []
    z1 = []

    position2 = f2['SubhaloPos'][:]
    x2 = []
    y2 = []
    z2 = []

    if len(position2) <= len(position1):
        steps = len(position2)
        t_var = f2['SnapNum'][:]
    else:
        steps = len(position1)
        t_var = f1['SnapNum'][:]

    time = list(t_var)
    time.reverse()

    for index in range(steps):
        x2.append(position2[index][0]/1000)
        y2.append(position2[index][1]/1000)
        z2.append(position2[index][2]/1000)
    x2.reverse()
    y2.reverse()
    z2.reverse()

    for index in range(steps):
        x1.append(position1[index][0]/1000)
        y1.append(position1[index][1]/1000)
        z1.append(position1[index][2]/1000)
    # Had to rever because first in list was at snapshat 99 not 1 - this is better because it will correlate with frame in update function
    x1.reverse()
    y1.reverse()
    z1.reverse()

    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(-140, 60) # Can change orientation with this first num is elevation of z axis (height viewing from) and second is azimuthal angle (xy-plane)

    # Initial scatter plot (empty at start)
    # Initial scatter plot (empty at start)
    scatter1 = ax.scatter([], [], [], c='b', label=f'Subhalo {sub1}', s=r1[0], alpha=0.8)
    scatter2 = ax.scatter([], [], [], c='r', label=f'Subhalo {sub2}', s=r2[0], alpha=0.8)

    # Set axes limits
    if min(x1) <= min(x2):
        min_x = min(x1)
    else:
        min_x = min(x2)
    if min(y1) <= min(y2):
        min_y = min(y1)
    else:
        min_y = min(y2)
    if min(z1) <= min(z2):
        min_z = min(z1)
    else:
        min_z = min(z2)

    if max(x1) >= max(x2):
        max_x = max(x1)
    else:
        max_x = max(x2)
    if max(y1) >= max(y2):
        max_y = max(y1)
    else:
        max_y = max(y2)
    if max(z1) >= max(z2):
        max_z = max(z1)
    else:
        max_z = max(z2)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    # Add legend
    ax.legend()
    ax.set_xlabel('x (cMpc/h)')
    ax.set_ylabel('y (cMpc/h)')
    ax.set_zlabel('z (cMpc/h)')

    def update(frame):
        # Update the scatter plots with the current points from the lists
        scatter1._offsets3d = ([x1[frame]], [y1[frame]], [z1[frame]])
        scatter1.set_sizes([r1[frame]])
        scatter2._offsets3d = ([x2[frame]], [y2[frame]], [z2[frame]])
        scatter2.set_sizes([r2[frame]])
        fig.suptitle(f"Snapshot = {time[frame]}")
        return scatter1, scatter2

    # Create animation with blit=False
    ani = FuncAnimation(fig, update, frames=len(x1), interval=60, blit=False)

    gif_fn = f"3d_anim_{sub1}&{sub2}.gif"

    anim_dir = f"./{sim_name}/animations/"
    if not os.path.exists(anim_dir):
        os.makedirs(anim_dir)
    fn = anim_dir + gif_fn

    # Show plot
    plt.show()

    # Save as a GIF
    ani.save(fn, writer='Pillow', dpi=300)
    print(f"Animation saved at {fn}")

    # Keep a reference to the animation object
    animation = ani


if __name__ == '__main__':
    args = parser.parse_args()

    sim_name = args.sim

    # Ex: If subhalos 0 and 1 are desired, type the following after the prompt: 0 1
    subhalo_ids = list(input("Which pair of subhalos would you like to plot? Separate by a space: ").split(" "))
    sub1 = subhalo_ids[0]
    sub2 = subhalo_ids[1]

    print("------------------------------------------")
    animation(sim_name, sub1, sub2, snap_start=99)