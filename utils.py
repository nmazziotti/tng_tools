import pandas as pd
import requests
import os

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