import pandas as pd
import requests
import os
import math
import numpy as np
import h5py

from scipy.integrate import quad

def get(path, sim_name=None, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"525de37a430c4c1fae3b595204b6c2d2"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
         filename = r.headers['content-disposition'].split("filename=")[1]

        # Creates new directories for the simulation and hdf5_files
         folder_path = f"./{sim_name}/hdf5_files"
         if not os.path.exists(folder_path):
             os.makedirs(folder_path)

        # hdf5 files will then be stored to this directory
         file_path = os.path.join(folder_path, filename)
         with open(file_path, 'wb') as f:
             f.write(r.content)
         return file_path # return the filename string

    return r


def get_subhalo(id, snap, sim_name):
    """Retrieve subhalo information of a given id, snapshot, and simulation"""
    subhalo_url = f"http://www.tng-project.org/api/{sim_name}/snapshots/{snap}/subhalos/" + str(id)
    subhalo = get(subhalo_url, sim_name=sim_name)
    return subhalo

def calc_dist(x1, y1, z1, x2, y2, z2):
    """Returns a list of distances between the pair of subhalo coordinates over time"""
    dist = []
    for i in range(len(x1)):
        r = np.sqrt( (x2[i]-x1[i])**2 + (y2[i]-y1[i])**2 + (z2[i]-z1[i])**2 )
        dist.append(r)
    return dist

def download_hdf5(sim_name, snap_start, subhalo_list):
    """Downloads the hdf5 files of the subhalos or reads in the hdf5 files if they already exist"""
    files = []

    for sub in subhalo_list:
        mpb = f"./{sim_name}/hdf5_files/lhalotree_mpb_{sub}.hdf5"
    # Subhalo 1
        if os.path.isfile(mpb):
            f = h5py.File(mpb, 'r')
            files.append(f)
        else:
            print(f"Retrieving Subhalo {sub} information...")
            subhalo = get_subhalo(sub, snap_start, sim_name=sim_name)
            mpb = get( subhalo['trees']['lhalotree_mpb'], sim_name=sim_name)
            f = h5py.File(mpb, 'r')
            files.append(f)

    return files

def redshifts(sim_name):
    url = f"http://www.tng-project.org/api/{sim_name}/snapshots"
    snaps = get(url, sim_name=sim_name)
    info = {}
    max_snap = snaps[-1]["number"]

    # Go through all of the snapshots in this simulation and retrieve the redshift and calculate a corresponding scale factor
    for index in range(max_snap + 1):
        info[snaps[index]['number']] = [snaps[index]['redshift']]
        info[snaps[index]['number']].append(1/(1 + snaps[index]['redshift'] ))

    # Saves list of redshifts and scale factors to a pandas table for future reference
    df = pd.DataFrame.from_dict(info, orient="index", columns=["z", "a"])
    return df

def integrand(zz):
    om = 0.3089
    ol = 0.6911
    E = math.sqrt(om*(1+zz)**3 + ol)
    return 1/((1.0 + zz) * E)

def lookback_time(x):

    if type(x) != int and type(x) != float and type(x):
        t_list = np.empty(len(x))

        for i in range(len(x)):
            th = (3.09*(10**17))*(1/0.6774)
            I = quad(integrand, 0, float(x[i]))
            result = I[0]*th/(365.25*24*60*60)/(10**9)
            t_list[i] = result 
        
        return t_list 

    else:
        th = (3.09*(10**17))*(1/0.6774)
        #print(th)
        I = quad(integrand, 0, float(x))
        #print(I)
        result = I[0]*th/(365.25*24*60*60)/(10**9)
        #print("Lookback Time: ~" + str(result) + " Gyr")
        #print("Age of Universe (at that redshift): ~" + str(13.803 - result) + " Gyr")
        return result
    

def align_snapshots(f1, f2):
    delete_from_f1 = []
    for item in f1['SnapNum'][:]:
        if item not in f2['SnapNum'][:]:
            delete_from_f1.append(np.where(f1['SnapNum'][:]==item)[0][0])

    new_f1_snap = np.delete(f1['SnapNum'][:], delete_from_f1)
    new_f1_pos = np.delete(f1['SubhaloPos'][:], delete_from_f1, axis=0)

    delete_from_f2 = []
    for item in f2['SnapNum'][:]:
        if item not in new_f1_snap:
            delete_from_f2.append(np.where(f2['SnapNum'][:]==item)[0][0])

    new_f2_snap = np.delete(f2['SnapNum'][:], delete_from_f2)
    new_f2_pos = np.delete(f2['SubhaloPos'][:], delete_from_f2, axis=0)

    return [[new_f1_snap, new_f1_pos], [new_f2_snap, new_f2_pos]]
