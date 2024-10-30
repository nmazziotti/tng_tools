import pandas as pd
import requests
import os
import math
import numpy as np
import h5py
import sys

from tqdm.auto import tqdm

from scipy.integrate import quad

sys.path.append("../")
import illustris_python.sublink as sl 
import illustris_python.groupcat as gc
import illustris_python.lhalotree as lht

basePathC = '/virgotng/mpia/TNG-Cluster/L680n8192TNG/output'
basePath1 = '/virgotng/universe/IllustrisTNG/TNG300-1/output'
basePath2 = '/virgotng/universe/IllustrisTNG/TNG300-2/output'
basePath3 = '/virgotng/universe/IllustrisTNG/TNG300-3/output'

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

def calc_dist(pos1, pos2, convert):
    """Returns a list of distances between the pair of subhalo coordinates over time"""
    dist = []
    for i in range(len(pos1[:,0])):
        if list(convert):
            r = np.sqrt( ((pos2[i,0]-pos1[i,0])*convert[i])**2 + ((pos2[i,1]-pos1[i,1])*convert[i])**2 + ((pos2[i,2]-pos1[i,2])*convert[i])**2 )
            dist.append(r)
        else:
            r = np.sqrt( (pos2[i,0]-pos1[i,0])**2 + (pos2[i,1]-pos1[i,1])**2 + (pos2[i,2]-pos1[i,2])**2 )
            dist.append(r)
    return dist

def download_hdf5(basePath, snap_start, subhalo_list):
    """Downloads the hdf5 files of the subhalos or reads in the hdf5 files if they already exist"""
    files = []
   

    for subhalo in subhalo_list:
        f = lht.loadTree(basePath, snapNum=snap_start, id=subhalo, fields=None, onlyMPB=True)
        files.append(f)

    return files

def redshifts(basePath):
    info = {}

    for i in range(100):
        hdr = gc.loadHeader(basePath, snapNum=i)
        info[i] = [hdr['Redshift']]
        info[i].append(hdr['Time'])

    df = pd.DataFrame.from_dict(info, orient="index", columns=["z", "a"])
    df.to_csv('./redshifts+scale_factors.csv', index=False)
    return df

try:
    df = pd.read_csv('./redshifts+scale_factors.csv')
except FileNotFoundError:
    df = redshifts(basePathC)

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
    
def get_index(basePath, snap_start, snap_end, id_start, MDB=True, MPB=False):
    """
    Get index of subhalo at any snapshot
    
    Parameters
    ----------
    - basePath: Path to TNG output directory
    
    - snap_start: Snapshot where subhalo index is known 
    
    - snap_end: Snapshot to find index of subhalo at
    
    - id_start: Known index of subhalo at snap_start
    
    """
    if MDB:
        slt = sl.loadTree(basePath, snapNum=snap_start, id=id_start, fields=['SnapNum', 'SubhaloID'], onlyMDB=True, treeName="SubLink", cache=True)
    elif MPB:
        slt = sl.loadTree(basePath, snapNum=snap_start, id=id_start, fields=['SnapNum', 'SubhaloID'], onlyMPB=True, treeName="SubLink", cache=True)
    
    snap_index = np.where(slt['SnapNum'][:] == snap_end)[0][0]
    if snap_index != None:
        subhaloID = slt['SubhaloID'][snap_index]
        f = h5py.File(gc.offsetPath(basePath, snap_end), 'r')
        id_index = np.where(f['Subhalo']['SubLink']['SubhaloID'][:] == subhaloID)[0][0]
        return id_index
    else:
        print(f"This subhalo only goes to snapshot {slt['SnapNum'][0]}")
    

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


def align_snap_info(f1, f2, data1, data2):
    delete_from_f1 = []
    for item in f1['SnapNum'][:]:
        if item not in f2['SnapNum'][:]:
            delete_from_f1.append(np.where(f1['SnapNum'][:]==item)[0][0])

    new_f1_snap = np.delete(f1['SnapNum'][:], delete_from_f1)
    new_data1 = np.delete(data1, delete_from_f1)

    delete_from_f2 = []
    for item in f2['SnapNum'][:]:
        if item not in new_f1_snap:
            delete_from_f2.append(np.where(f2['SnapNum'][:]==item)[0][0])

    new_f2_snap = np.delete(f2['SnapNum'][:], delete_from_f2)
    new_data2 = np.delete(data2, delete_from_f2)

    return [[new_f1_snap, new_data1], [new_f2_snap, new_data2]]