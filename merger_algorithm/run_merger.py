import h5py
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
from matplotlib import ticker, rc

sys.path.append("../")


import illustris_python.sublink as sl 
import illustris_python.groupcat as gc
import illustris_python.lhalotree as lht
import illustris_python.snapshot as sn 

from tng_tools.utils import *
from tng_tools.animation_3D import *
from tng_tools.find_merger import *

font = {'family' : 'serif',
        'size'   : 14}

rc('font', **font)

sim_name = "TNG300-2"
basePath = basePath2

try:
    df = pd.read_csv('./redshifts+scale_factors.csv')
except FileNotFoundError:
    df = redshifts(basePath)

snapNum = 49    
    
print(f"Selected redshift: z = {df['z'][snapNum]}")
print("Loading in subhalos...")
subhalos = gc.loadSubhalos(basePath, snapNum=snapNum)
print(f"Number of subhalos in snap {snapNum}: {len(subhalos['SubhaloMass'][:])}")

indices = np.arange(0, len(subhalos['SubhaloMass'][:]), 1)

min_mass = 10**13.5
mass_cut = (subhalos['SubhaloMass'][:]*1e10/0.704 > min_mass)
masses = subhalos['SubhaloMass'][mass_cut]*1e10/0.704
log_masses = np.log10(masses)
c = subhalos['SubhaloPos'][mass_cut]
hm_rad = subhalos['SubhaloHalfmassRad'][mass_cut]
ids = indices[mass_cut]
print(f"Number of subhalos above mass {min_mass}: {len(ids)}")

dir_name = f"{sim_name}/z={np.round(df['z'][snapNum], 4)}_10^{np.log10(min_mass)}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print("Constructing runlist...")
try:
    df_runlist = pd.read_csv(f"./{dir_name}/runlist.csv")
    runlist = {}
    for column in df_runlist.columns:
        runlist[column] = json.loads(df_runlist[column][0])
except FileNotFoundError:
    runlist = create_runlist(df, ids, c, 10, snapNum)
    df_runlist = pd.DataFrame([runlist])
    df_runlist.to_csv(f"./{dir_name}/runlist.csv", index=False)
    
    
print("Beginning merger-identifier algorithm...")
logging.basicConfig(filename=f"./{dir_name}/runlist.log", 
                format='%(asctime)s %(message)s', 
                filemode='w')

logger=logging.getLogger() 

logger.setLevel(logging.INFO)

logger.info("----------------------NEW RUN----------------------")

already_ran = []

df_mergers = pd.DataFrame(columns=["sub1", "sub2", "pericenter", "p_info", "apocenter", "a_info"])


for key in tqdm(list(runlist.keys())[:]):
    for id in runlist[key]:
            if [int(key), int(id)] in already_ran or [int(id), int(key)] in already_ran:
                 pass
            else:
                row = plot_merger_hmr(df, dir_name=dir_name, basePath=basePath, snap_start=snapNum, sub1=int(key), sub2=int(id), logger=logger)
                if row:
                    df_mergers = pd.concat([df_mergers, pd.DataFrame([row])])
                already_ran.append([int(key), int(id)])
                
df_mergers.to_csv(f"{dir_name}/mergers_found_since_snap{snapNum}.csv", index=False)
print("Run complete")