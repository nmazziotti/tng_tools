from .utils import *


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("../")
import illustris_python.sublink as sl 
import illustris_python.groupcat as gc
import illustris_python.lhalotree as lht


def locate_min(dist, candidates, concav, snap_list):
    """Function that flags any minima that occur below 1 Mpc and are concave up"""
    x = []
    y = []

    mins = dist[candidates]
    for i in range(len(mins)):
        if mins[i] < 1.0:
            try:
                avg_concav = (concav[candidates[0][i] - 1] + concav[candidates[0][i]] + concav[candidates[0][i] + 1])/3 
                if avg_concav > 0:
                    x.append(snap_list[candidates[0][i]])
                    y.append(mins[i])
                    # if comoving:
                    #     print(f"Potential merger event in comoving coordinates at Snapshot {max_snap - candidates[0][i]}")
                    # else:
                    #     print(f"Potential merger event in physical at Snapshot {max_snap - candidates[0][i]}")
            except IndexError:
                pass

    return x,y


def create_runlist(df, IDs, coords, min_sep, snapNum):
    runlist = {}
    a = df["a"][snapNum]

    convert = (1/0.6774)*0.001*a
 
    x = np.array(coords[:,0])*convert
    y = np.array(coords[:,1])*convert
    z = np.array(coords[:,2])*convert
    
    for i in tqdm(range(len(coords))):
        close = []
        for j in range(len(coords)):
            if i != j:
                dist = np.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2 )
                if dist < min_sep:
                    close.append(IDs[j])
        if close:
            runlist[f"{IDs[i]}"] = close

    return runlist

def find_pericenter(times, dists):
    p = min(dists)
    p_index = np.where(dists == p)[0][0]
    t_p = times[p_index]

    return t_p, p

def find_apocenter(times, dists):
    a = max(dists)
    a_index = np.where(dists == a)[0][0]
    t_a = times[a_index]
    return t_a, a
    

# Half-mass radius method
def pass_through(df, dir_name, basePath, sub1, sub2, snap_start, logger=None):
    
    f1 = sl.loadTree(basePath, snapNum=snap_start, id=sub1, fields=['SubhaloPos', 'SubhaloMass', 'SubhaloHalfmassRad', 'SnapNum'], onlyMDB=True, treeName="SubLink", cache=True)
    f2 = sl.loadTree(basePath, snapNum=snap_start, id=sub2, fields=['SubhaloPos', 'SubhaloMass', 'SubhaloHalfmassRad', 'SnapNum'], onlyMDB=True, treeName="SubLink", cache=True)
 
    try:
        if len(f1['SubhaloPos']) > 101 - snap_start or len(f2['SubhaloPos']) > 101 - snap_start:
            return [False]
        else:
            if max(f1['SubhaloMass'][:]*1e10/0.704) > 1e13 and max(f2['SubhaloMass'][:]*1e10/0.704) > 1e13:
                new_data = align_snapshots(f1, f2)

                snapnum1 = new_data[0][0]
                pos1 = new_data[0][1] # x,y,z coordinates of subhalo 1 over all snapshots

                snapnum2 = new_data[1][0]
                pos2 = new_data[1][1] # x,y,z coordinates of subhalo 2 over all snapshots

                snap_list = snapnum1
                final_snap = snap_list[-1]


                new_a_vals = []
                new_z_vals = []
                for snap in snapnum1:
                    new_a_vals.append(df["a"][snap])
                    new_z_vals.append(df["z"][snap])

                new_a_vals = np.array(new_a_vals)
                new_z_vals = np.array(new_z_vals)

                look_at = (new_z_vals < 100)

                # Subhalo coordinates are by default in ckpc/h, so this converts to cMpc/h
                codist = np.array(calc_dist(pos1, pos2, convert=[]))/1000
                co_y_label = "Distance [cMpc/h]"

                # Factor to convert each comoving coordinate to the physical coordinate
                convert = (1/0.6774)*0.001*new_a_vals[look_at]
                dist = np.array(calc_dist(pos1, pos2, convert=convert))
                y_label = "Distance [Mpc]"

                rad = align_snap_info(f1, f2, f1['SubhaloHalfmassRad'][:], f2['SubhaloHalfmassRad'][:])
                r1 = (rad[0][1])*convert
                r2 = (rad[1][1])*convert

                #snap = get_merger_snap("TNG-Cluster", basePath, sub1=sub1, sub2=sub2, snap_start=snap_start)

                #plt.plot(lookback_time(new_z_vals[look_at]), r1)
                #plt.plot(lookback_time(new_z_vals[look_at]), r2)

                if abs(max(dist) - min(dist)) < 50:
                    r = r1 + r2 
                    has_merged = dist < r

                    if True in has_merged:
                        if logger:
                            logger.info(f"POSSIBLE MERGER BETWEEN: Subhalos {sub1} and {sub2}")
                        else:
                            pass
                            #print(f"POSSIBLE MERGER BETWEEN: Subhalos {sub1} and {sub2}")
                        return [True, new_z_vals, dist, r, y_label, snapnum1]
                    else:
                        return [False]
                else:
                    if logger:
                        logger.info(f"Unphysical distance jump between Subhalos {sub1} and {sub2}")
                    else:
                        pass
                        #print(f"Unphysical distance jump between Subhalos {sub1} and {sub2}")

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,5))

                    ax[0].plot(lookback_time(new_z_vals[look_at]), codist)
                    ax[1].plot(lookback_time(new_z_vals[look_at]), dist)
                    ax[0].set_xlabel("Lookback Time [Gyr]")
                    ax[1].set_xlabel("Lookback Time [Gyr]")


                    fig.suptitle("Distance between subhalos " + str(sub1) + " & " + str(sub2))

                    # Save figure to a directory called plots
                    fig_fn = "./%s/plots/unphysical/dist_%s&%s_snap=%s-%s.png"%(dir_name, sub1, sub2, snapnum1[0], snapnum1[-1])

                    reverse_fn = "./%s/plots/unphysical/dist_%s&%s_snap=%s-%s.png"%(dir_name, sub2, sub1, snapnum1[0], snapnum1[-1])
                    plot_path = f"./{dir_name}/plots/unphysical"
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)

                    if not os.path.exists(reverse_fn):
                        fig.savefig(fig_fn)

                    plt.close(fig=fig)

                    return [False]
            else:
                return [False]
    except TypeError:
        print(f"Error: {sub1} & {sub2}")
        return [False]
        
    # Add in criteria later that requires True to hold for multiple snapshots 
    #print(list(has_merged).count(True))
    
    
    
def plot_merger_hmr(df, dir_name, basePath, sub1, sub2, snap_start, logger=None):
    return_list = pass_through(df, dir_name, basePath, sub1, sub2, snap_start, logger)
    if return_list[0] == True:

        z_vals = return_list[1]
        dist = return_list[2]
        r = return_list[3]
        y_label = return_list[4]
        snap_list = return_list[5]
        
        final_snap = snap_list[-1]
        

        t_p, p = find_pericenter([lookback_time(float(df["z"][snap])) for snap in snap_list], dist)
        z_p, p = find_pericenter([float(df["z"][snap]) for snap in snap_list], dist)
        snap_p, p = find_pericenter(snap_list, dist)

        t_a, a = find_apocenter([lookback_time(float(df["z"][snap])) for snap in snap_list], dist)
        z_a, a = find_apocenter([float(df["z"][snap]) for snap in snap_list], dist)
        snap_a, a = find_apocenter(snap_list, dist)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
        ax.plot(lookback_time(z_vals), dist, label='Distance')
        ax.plot(lookback_time(z_vals), r, label='Combined HMR')
        ax.set_xlabel("Lookback Time [Gyr]")

        ax.axvline(x = t_p, color='gray', linestyle="--")

        ax.text(t_p - 0.1, max(dist)/2, "Merger?", rotation=90, color='gray', backgroundcolor='white', fontsize=12)


        ax.set_ylabel(y_label)


        ax.plot(lookback_time(float(df["z"][snap_p])), p, 'o', color='red', markersize=4, label=f"({np.round(t_p,3)} Gyr, {np.round(p, 3)} Mpc)")

        ax.legend(loc='upper right')

        fig.suptitle("Distance between subhalos " + str(sub1) + " & " + str(sub2))

        # Save figure to a directory called plots
        fig_fn = "./%s/plots/mergers/dist_%s&%s_snap=%s-%s.png"%(dir_name, sub1, sub2, snap_list[0], snap_list[-1])

        reverse_fn = "./%s/plots/mergers/dist_%s&%s_snap=%s-%s.png"%(dir_name, sub2, sub1, snap_list[0], snap_list[-1])
        plot_path = f"./{dir_name}/plots/mergers"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        if not os.path.exists(reverse_fn):
            fig.savefig(fig_fn)

        plt.close(fig=fig)

        row = {"sub1" : sub1,
              "sub2": sub2, 
              "pericenter": p,
               "p_info": [z_p, t_p, snap_p],
               "apocenter": a,
               "a_info": [z_a, t_a, snap_a]
              }
        return row
