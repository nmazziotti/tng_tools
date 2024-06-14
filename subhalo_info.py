from utils import get

def find_subhalos(snap, sim_name):
    """Retrieve mass, position, and parent halo id of subhaloes bigger than 10^14 solar masses"""

    mass_min = 10**14.0 / 1e10 * 0.704

    search_query = "?mass__gt=" + str(mass_min)

    # form the url and make the request
    url = f"http://www.tng-project.org/api/{sim_name}/snapshots/{snap}/subhalos/" + search_query
    subhalos = get(url, {'limit':300000})
    print(subhalos['count'])
    ids = [ subhalos['results'][i]['id'] for i in range(len(subhalos['results'])) ]
    masses = [ subhalos['results'][i]['mass_log_msun'] for i in range(len(subhalos['results'])) ]
    
    return ids, masses


def get_subhalo(id, snap, sim_name):
    """Retrieve subhalo information of a given id, snapshot, and simulation"""
    subhalo_url = f"http://www.tng-project.org/api/{sim_name}/snapshots/{snap}/subhalos/" + str(id)
    subhalo = get(subhalo_url, sim_name=sim_name)
    return subhalo