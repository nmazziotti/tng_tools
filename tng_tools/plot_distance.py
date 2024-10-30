import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py

from matplotlib import rc 

from .utils import redshifts, download_hdf5, lookback_time, align_snapshots

font = {'family' : 'serif',
        'size'   : 14}

rc('font', **font)

parser = argparse.ArgumentParser()

# Command line arguments 
# Ex: python plot_distance.py -s TNG300-1
parser.add_argument("-s", "--sim", required=True, type=str, help="Select simulation from IllustrisTNG")

# Ex: python plot_distance.py -s TNG300-1 -i 130
# This argument is not required and will default to 99 (initial snapshot of TNG300-1 simulations)
parser.add_argument("-i", default=99, type=int, help="Initial snapshot") 

# Ex: python plot_distance.py -s TNG300-1 -f 2
# If this argument is not used in the command line, a prompt will show up to specify snapshot to stop at  
parser.add_argument("-f", type=int, help="Final snapshot")

parser.add_argument("-x", type=str, default='time', help="Plot either redshift, snapshot, or time on x-axis")


def calc_dist(x1, y1, z1, x2, y2, z2):
    """Returns a list of distances between the pair of subhalo coordinates over time"""
    dist = []
    for i in range(len(x1)):
        r = np.sqrt( (x2[i]-x1[i])**2 + (y2[i]-y1[i])**2 + (z2[i]-z1[i])**2 )
        dist.append(r)
    return dist

def plot_distance(sim_name, sub1, sub2, snap_start, snap_end, x_axis):
    """Plots the distance between the two subhalos over the range of given snapshots"""
    files = download_hdf5(sim_name, snap_start, subhalo_list=[sub1, sub2])

    f1, f2 = files[0], files[1]

    try: 
        df = pd.read_csv(f"{sim_name}/redshifts+scale_factors.csv")
    except FileNotFoundError:
        df = redshifts(sim_name)
        df.to_csv(f"{sim_name}/redshifts+scale_factors.csv")

    z_vals = np.flip(np.array(df["z"]))
    a_vals = np.flip(np.array(df["a"]))

    new_data = align_snapshots(f1, f2)
    
    snapnum1 = new_data[0][0]
    pos1 = new_data[0][1] # x,y,z coordinates of subhalo 1 over all snapshots

    snapnum2 = new_data[1][0]
    pos2 = new_data[1][1] # x,y,z coordinates of subhalo 2 over all snapshots


    new_a_vals = []
    new_z_vals = []
    for snap in snapnum1:
        new_a_vals.append(df["a"][snap])
        new_z_vals.append(df["z"][snap])

    new_a_vals = np.array(new_a_vals)
    new_z_vals = np.array(new_z_vals)

    look_at = (new_z_vals < 1)
    

    # if len(snapnum1) != len(snapnum2):
    #     print(f"NOTE: Subhalo {sub1} has {len(snapnum1)} snapshots available, while Subhalo {sub2} has {len(snapnum2)} snapshots available.")
    #     if len(snapnum1) > len(snapnum2):
    #         print(f"--> Subhalo {sub2} only goes to snapshot {snapnum2[-1]}")

    #         # Need to stop at the last snapshot of Subhalo 2 if Subhalo 2 does not go past snap_end 
    #         if snapnum2[-1] > snap_end:
    #             snap_end = snapnum2[-1]

    #         # All of the snapshot numbers should be the same for each subhalo even if they aren't the same length, but this will catch 
    #         # any of that do not for some reason.
    #         for j in range(len(snapnum2)):
    #             if snapnum1[j] != snapnum2[j]:
    #                 print("WARNING: Some of the snapshots in this pairing of subhalos do not match up.")
    #                 break
    #     else:
    #         print(f"--> Subhalo {sub1} only goes to snapshot {snapnum1[-1]}")

    #         # Need to stop at the last snapshot of Subhalo 1 if Subhalo 1 does not go past snap_end 
    #         if snapnum1[-1] > snap_end:
    #             snap_end = snapnum1[-1]

    #         for j in range(len(snapnum1)):
    #             if snapnum1[j] != snapnum2[j]:
    #                 print("WARNING: Some of the snapshots in this pairing do not match up.")
    #                 break

    #     # Converts snap_end to an index i that the data arrays can stop at (includes snap_end)
    #     i = snap_start - snap_end + 1
    # else:
    #     i = snap_start - snap_end + 1

    # Subhalo coordinates are by default in ckpc/h, so this converts to cMpc/h
    codist = np.array(calc_dist(pos1[look_at, 0]/1000, pos1[look_at, 1]/1000, pos1[look_at, 2]/1000, pos2[look_at, 0]/1000, pos2[look_at, 1]/1000, pos2[look_at, 2]/1000))
    co_y_label = "Distance [cMpc/h]"

    # Factor to convert each comoving coordinate to the physical coordinate
    convert = (1/0.6774)*0.001*new_a_vals[look_at]
    dist = np.array(calc_dist(pos1[look_at, 0]*convert, pos1[look_at, 1]*convert, pos1[look_at, 2]*convert, pos2[look_at, 0]*convert, pos2[look_at, 1]*convert, pos2[look_at, 2]*convert))
    y_label = "Distance [Mpc]"

    # Plot both comoving and physical coordinates over time in across redshifts (nonlinear time) and snapshot number (linear time)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))


    if x_axis.lower() == 'redshift':
        ax[0].plot(new_z_vals[look_at], codist)
        ax[1].plot(new_z_vals[look_at], dist)
        ax[0].set_xlabel("Redshift")
        ax[1].set_xlabel("Redshift")
    elif x_axis.lower() == 'snapshot':
        ax[0].plot(snapnum1[look_at], codist)
        ax[1].plot(snapnum1[look_at], dist)
        ax[0].set_xlabel("Snapshot")
        ax[1].set_xlabel("Snapshot")
        ax[0].invert_xaxis()
        ax[1].invert_xaxis()
    else:
        times = [lookback_time(float(z)) for z in new_z_vals[look_at]]
        ax[0].plot(times, codist)
        ax[1].plot(times, dist)
        ax[0].set_xlabel("Lookback Time [Gyr]")
        ax[1].set_xlabel("Lookback Time [Gyr]")

    ax[0].set_ylabel(co_y_label)
    ax[1].set_ylabel(y_label)

    fig.suptitle("Distance between subhalos " + str(sub1) + " & " + str(sub2))

    # Save figure to a directory called plots
    fig_fn = "./%s/plots/dist_%s&%s_snap=%s-%s.png"%(sim_name, sub1, sub2, snap_start, snap_end)
    plot_path = f"./{sim_name}/plots"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fig.legend([sim_name])
    fig.savefig(fig_fn)
    print(f"Figure saved at {fig_fn}")

if __name__ == '__main__':
    args = parser.parse_args()

    sim_name = args.sim

    # Ex: If subhalos 0 and 1 are desired, type the following after the prompt: 0 1
    subhalo_ids = list(input("Which pair of subhalos would you like to plot? Separate by a space: ").split(" "))

    snap_start = args.i

    x_axis = args.x

    if args.f:
        snap_end = args.f 
    else:
        snap_end = int(input("What snapshot would you like to end at? "))

    print("------------------------------------------")
    
    plot_distance(sim_name, subhalo_ids[0], subhalo_ids[1], snap_start, snap_end, x_axis=x_axis) 
