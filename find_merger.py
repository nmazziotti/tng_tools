from .utils import calc_dist, redshifts, download_hdf5, align_snapshots, lookback_time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os


def locate_min(dist, candidates, concav, max_snap, comoving=False):
    """Function that flags any minima that occur below 5 Mpc and are concave up"""
    x = []
    y = []

    mins = dist[candidates]
    for i in range(len(mins)):
        if mins[i] < 5:
            try:
                avg_concav = (concav[candidates[0][i] - 1] + concav[candidates[0][i]] + concav[candidates[0][i] + 1])/3 
                if avg_concav > 0:
                    x.append(max_snap - candidates[0][i])
                    y.append(mins[i])
                    # if comoving:
                    #     print(f"Potential merger event in comoving coordinates at Snapshot {max_snap - candidates[0][i]}")
                    # else:
                    #     print(f"Potential merger event in physical at Snapshot {max_snap - candidates[0][i]}")
            except IndexError:
                pass

    return x,y


def create_runlist(data, sim_name, min_sep):
    runlist = {}

    try: 
        df = pd.read_csv(f"{sim_name}/redshifts+scale_factors.csv")
    except FileNotFoundError:
        df = redshifts(sim_name)
        df.to_csv(f"{sim_name}/redshifts+scale_factors.csv")

    a = df["a"][99]

    convert = (1/0.6774)*0.001*a
        
    x = np.array(data["x"][:])*convert
    y = np.array(data["y"][:])*convert
    z = np.array(data["z"][:])*convert
    
    for i in tqdm(range(len(df))):
        close = []
        for j in range(len(df)):
            if i != j:
                dist = np.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2 )
                if dist < min_sep:
                    close.append(data['subhalo_id'][j])
        if close:
            runlist[f"{data['subhalo_id'][i]}"] = close

    return runlist



def flag_merger(sim_name, sub1, sub2, snap_start, snap_end):
    """Plots the distance between the two subhalos over the range of given snapshots"""
    f1, f2 = download_hdf5(sim_name, snap_start, subhalo_list=[sub1, sub2])

    try: 
        df = pd.read_csv(f"{sim_name}/redshifts+scale_factors.csv")
    except FileNotFoundError:
        df = redshifts(sim_name)
        df.to_csv(f"{sim_name}/redshifts+scale_factors.csv")

    #z_vals = np.flip(np.array(df["z"]))
    #a_vals = np.flip(np.array(df["a"]))

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

    # Subhalo coordinates are by default in ckpc/h, so this converts to cMpc/h
    codist = np.array(calc_dist(pos1[:, 0]/1000, pos1[:, 1]/1000, pos1[:, 2]/1000, pos2[:, 0]/1000, pos2[:, 1]/1000, pos2[:, 2]/1000))
    co_y_label = "Distance [cMpc/h]"

    # Factor to convert each comoving coordinate to the physical coordinate
    convert = (1/0.6774)*0.001*new_a_vals
    dist = np.array(calc_dist(pos1[:, 0]*convert, pos1[:, 1]*convert, pos1[:, 2]*convert, pos2[:, 0]*convert, pos2[:, 1]*convert, pos2[:, 2]*convert))
    y_label = "Distance [Mpc]"

    co_min = argrelextrema(codist, np.less)
    co_concav = np.diff(codist, n=2)

    phys_min = argrelextrema(dist, np.less)
    phys_concav = np.diff(dist, n=2)

    co_x, co_y = locate_min(codist, co_min, co_concav, max_snap=snap_start, comoving=True)
    phys_x, phys_y = locate_min(dist, phys_min, phys_concav, max_snap=snap_start)

    
    if co_x or phys_x:
        return [True, new_z_vals, codist, dist, co_x, co_y, phys_x, phys_y]
    else:
        return [False]
    
def plot_merger(sim_name, sub1, sub2, snap_start, snap_end):
    try: 
        df = pd.read_csv(f"{sim_name}/redshifts+scale_factors.csv")
    except FileNotFoundError:
        df = redshifts(sim_name)
        df.to_csv(f"{sim_name}/redshifts+scale_factors.csv")

    # Plot both comoving and physical coordinates over time in across redshifts (nonlinear time) and snapshot number (linear time)

    return_list = flag_merger(sim_name, sub1, sub2, snap_start, snap_end)
    if return_list[0] == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

        z_vals = return_list[1]
        codist = return_list[2]
        dist = return_list[3]
        co_x = return_list[4]
        co_y = return_list[5]
        phys_x = return_list[6]
        phys_y = return_list[7]


        ax[0].plot(lookback_time(z_vals[:]), codist)
        ax[1].plot(lookback_time(z_vals[:]), dist)
        ax[0].set_xlabel("Lookback Time [Gyr]")
        ax[1].set_xlabel("Lookback Time [Gyr]")

        # ax[1,0].plot(snapnum1[:i], codist)
        # ax[1,1].plot(snapnum1[:i], dist)

        co_y_label = "Comoving Distance [cMpc/h]"
        y_label = "Distance [Mpc]"


        ax[0].set_ylabel(co_y_label)
        ax[1].set_ylabel(y_label)
        

        ax[0].plot([lookback_time(float(df["z"][snap])) for snap in co_x], co_y, 'o', color='red', markersize=4, label="Merger?")
        ax[1].plot([lookback_time(float(df["z"][snap])) for snap in phys_x], phys_y, 'o', color='red', markersize=4, label="Merger?")


        ax[0].legend()
        ax[1].legend()

        fig.suptitle("Distance between subhalos " + str(sub1) + " & " + str(sub2))

        # Save figure to a directory called plots
        fig_fn = "./%s/plots/dist_%s&%s_snap=%s-%s.png"%(sim_name, sub1, sub2, snap_start, snap_end)

        revese_fn = "./%s/plots/dist_%s&%s_snap=%s-%s.png"%(sim_name, sub2, sub1, snap_start, snap_end)
        plot_path = f"./{sim_name}/plots"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        if not os.path.exists(revese_fn):
            fig.savefig(fig_fn)

        fig.legend([sim_name])
        #print(f"Figure saved at {fig_fn}")
        plt.show()