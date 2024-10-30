from .utils import get, get_subhalo
from tqdm.auto import tqdm
import os
import numpy as np

from astropy.table import Table


def save_table(ids, masses, coords, phalos, filename):
    t = Table([ids, masses, coords[:,0], coords[:,1], coords[:,2], phalos], names=("subhalo_id", "mass_log_msun", "x", "y", "z", "phalo_id"))
    t.write(filename, format='ascii', overwrite=True)

def find_subhalos(snap, sim_name, mass_lim, existing_sample_fn=None):
    """Retrieve mass, position, and parent halo id of subhaloes bigger than 10^14 solar masses"""

    mass_min = mass_lim*10**14.0 / 1e10 * 0.704

    search_query = "?mass__gt=" + str(mass_min)

    # form the url and make the request
    url = f"http://www.tng-project.org/api/{sim_name}/snapshots/{snap}/subhalos/" + search_query
    subhalos = get(url, params={'limit':300000})
    print(f"Number of subhalos found: {subhalos['count']}")
    if subhalos['count'] != 0:
        ids = [ subhalos['results'][i]['id'] for i in range(len(subhalos['results'])) ]
        masses = [ subhalos['results'][i]['mass_log_msun'] for i in range(len(subhalos['results'])) ]

        coords = []
        phalos = []

        t = Table.read("./TNG300-1/mass>0.5_TNG300-1.dat", format="ascii")

        try:
            for i in tqdm(range(len(ids))):
                id = ids[i]
                if existing_sample_fn:
                    t = Table.read(existing_sample_fn, format="ascii")
                    
                    if id in t['subhalo_id']:
                        index = np.where(np.array(t['subhalo_id']) == id)[0][0]
                        coords.append([t['x'][index], t['y'][index], t['z'][index]])
                        phalos.append(t['phalo_id'][index])
                    else:
                        subhalo = get_subhalo(id=id, snap=snap, sim_name=sim_name)
                        coords.append([subhalo['pos_x'], subhalo['pos_y'], subhalo['pos_z']])
                        phalos.append(subhalo['grnr'])
                else:
                    subhalo = get_subhalo(id=id, snap=snap, sim_name=sim_name)
                    coords.append([subhalo['pos_x'], subhalo['pos_y'], subhalo['pos_z']])
                    phalos.append(subhalo['grnr']) 

            folder_path = f"./{sim_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


            fn = folder_path + f"mass>{mass_lim}_snap{snap}.dat"
            save_table(ids, masses, np.array(coords), phalos, fn)
            print(f"Table saved at {fn}")
        except ConnectionError:
            folder_path = f"./{sim_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            fn = folder_path + f"PARTIAL_mass>{mass_lim}_snap{snap}.dat"
            save_table(ids[:i], masses[:i], np.array(coords), phalos, fn)
            print(f"Table saved at {fn}")