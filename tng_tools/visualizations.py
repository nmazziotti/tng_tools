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
import matplotlib as mpl
from astropy import constants as const
from astropy import units as u
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from astropy import wcs                                                          
from astropy.io import fits 

sys.path.append("../")


import illustris_python.sublink as sl 
import illustris_python.groupcat as gc
import illustris_python.lhalotree as lht
import illustris_python.snapshot as sn 

from .utils import *
from .animation_3D import *
from .find_merger import *

font = {'family' : 'serif', 
       'size': 12}

rc('font', **font)
    

def create_proj(s, snapNum, proj, pos, depth, max_dist, npix):
    """
    Projects 3D subhalo onto 2D projection along the x, y, or z axes
    
    Parameters
    ----------
    - s: Subset of full snapshot hdf5 file (at snapNum) containing the partType information 
        for the cells belonging to this subhalo
        
    - snapNum: Snapshot at which s is extracted
    
    - proj: Coordinate plane on which subhalo information is projected onto. 0 is x-y plane, 1 is x-z plane,
        and 2 is y-z plane. 
        
    - pos: Spatial position of the particle with the minium gravitational potential energy in given subhalo 
        converted to physical coordinates [Mpc]. 
        
    - depth: Sets maximum depth of projected image [Mpc]. For a complete projection, depth should be set to a value
        larger than the distance between the frontmost and backmost cells of the subhalo along the projection axis.
        
    - max_dist: Maximum separation distance along each axis from the corresponding coordinate of pos [Mpc]. Only cells
        within max_dist will be considered in the projection. 
        
    - npix: Pixel length of image. 
    """
    
    convert = (1/0.6774)*df['a'][snapNum] # Conversion factor to move from comoving to physical coordinates
    max_dist = max_dist * 1000
    
    if proj == 0:
        z = s['Coordinates'][:,2]*convert
        d = abs((z - min(z))/1000)
        if depth > np.max(d):
            print(f"Maximum depth of image: {np.round(np.max(d), 4)} Mpc")
        xmin, xmax = pos[0] - max_dist, pos[0] + max_dist
        ymin, ymax = pos[1] - max_dist, pos[1] + max_dist
        x = s['Coordinates'][:,0]*convert
        y = s['Coordinates'][:,1]*convert
        condition = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (d < depth)
        x = x[condition] - pos[0]
        y = y[condition] - pos[1]
        proj_label = 'x-y'
    elif proj == 1:
        z = s['Coordinates'][:,1]*convert
        d = (z - min(z))/1000
        if depth > np.max(d):
            print(f"Maximum depth of image: {np.round(np.max(d), 4)} Mpc")
        xmin, xmax = pos[0] - max_dist, pos[0] + max_dist
        ymin, ymax = pos[2] - max_dist, pos[2] + max_dist 
        x = s['Coordinates'][:,0]*convert
        y = s['Coordinates'][:,2]*convert
        condition = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (d < depth)
        x = x[condition] - pos[0]
        y = y[condition] - pos[2]
        proj_label = 'x-z'
    elif proj == 2:
        z = s['Coordinates'][:,0]*convert
        d = (z - min(z))/1000
        if depth > np.max(d):
            print(f"Maximum depth of image: {np.round(np.max(d), 4)} Mpc")
        xmin, xmax = pos[1] - max_dist, pos[1] + max_dist
        ymin, ymax = pos[2] - max_dist, pos[2] + max_dist
        x = s['Coordinates'][:,1]*convert
        y = s['Coordinates'][:,2]*convert
        condition = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (d < depth)
        x = x[condition] - pos[1]
        y = y[condition] - pos[2]
        proj_label = 'y-z'
        
    extent = xmin, xmax, ymin, ymax
    return x, y, condition, proj_label, extent

def stack_info(x, y, weights, npix):
    """
    Takes 3D information of subhalo and constructs a 2D histogram to generate projected image 
    
    Parameters
    ----------
    - x: Horizontal coordinates of projection [Mpc].
    
    - y: Vertical coordinates of projection [Mpc].
    
    - weights: partType information that is being stacked in the projection. 
    
    - npix: Pixel length of image.
    """
    
    plt.figure(figsize=(12,12))
    h = plt.hist2d(x,y, weights=weights, bins=npix)
    plt.close()
    return h

def log(h):
    """
    Base 10 logs each pixel of 2D histogram if > 0. Returns log10 image.
    """
    
    log_scale = np.zeros(h.shape)

    for i in range(len(h[:,0])):
        for j in range(len(h[:,0])):
            if h[i,j] > 0: 
                log_scale[i,j] = np.log10(h[i,j])
    return log_scale

def create_wcs(npix, crpix_x, crpix_y, xps, yps, proj, pos):
    if proj == 0:
        xlabel = 'Offset y'
        ylabel = 'Offset x'
    elif proj == 1:
        xlabel = 'Offset z'
        ylabel = 'Offset x'
    elif proj == 2:
        xlabel = 'Offset z'
        ylabel = 'Offset y'
        
    wcs_dict = {
    "CTYPE1": xlabel,
    "CUNIT1": "kpc",
    "CDELT1": yps,
    "CRPIX1": crpix_x,
    "CRVAL1": pos[1],
    "NAXIS1": npix,
    "CTYPE2": ylabel,
    "CUNIT2": "kpc",
    "CDELT2": xps,
    "CRPIX2": crpix_y,
    "CRVAL2": pos[0],
    "NAXIS2": npix,
    }
    input_wcs = wcs.WCS(wcs_dict)
    
    return input_wcs

def concatenate_dicts(dict1, dict2):
    combined_dict = {}
    
    for key in dict1.keys():
        if key in dict2:
            # Concatenate arrays from both dictionaries
            if type(dict1[key]) == np.ndarray:
                combined_dict[key] = np.concatenate((dict1[key], dict2[key]))
        else:
            # If the key is not in dict2, just keep dict1's array
            combined_dict[key] = dict1[key]
    
    return combined_dict


def combine_subhalos(basePath, snapNum, ids, partType):
    s1 = sn.loadSubhalo(basePath, snapNum, ids[0], partType)
    s2 = sn.loadSubhalo(basePath, snapNum, ids[1], partType)
    
    new_s = concatenate_dicts(s1, s2)
    
    slt1 = sl.loadTree(basePath, snapNum, ids[0], fields=['SnapNum', 'SubhaloPos', 'SubhaloMass'], onlyMPB=True, treeName="SubLink", cache=True)
    snap_index1 = np.where(slt1['SnapNum'] == snapNum)[0][0]
    
    slt2 = sl.loadTree(basePath, snapNum, ids[1], fields=['SnapNum', 'SubhaloPos', 'SubhaloMass'], onlyMPB=True, treeName="SubLink", cache=True)
    snap_index2 = np.where(slt2['SnapNum'] == snapNum)[0][0]
    
    
    if slt1['SubhaloMass'][snap_index1] > slt2['SubhaloMass'][snap_index2]:
        pos = slt1['SubhaloPos'][snap_index1] * (1/0.6774) * df['a'][snapNum]
    else:
        pos = slt2['SubhaloPos'][snap_index2] * (1/0.6774) * df['a'][snapNum]
    
    return new_s, pos

def plot_gas_image(basePath, id, snapNum, infoName, 
                   proj, max_dist=5, depth = 20, npix=500):
    """
    Plots projected image of subhalo as seen in a given gas property. 
    
    Parameters
    ----------
    - basePath: Path to TNG output directory
    
    - id: Index of subhalo at snapNum
    
    - snapNum: Full snapshot to look at 
    
    - infoName: partType parameter to look at. See https://www.tng-project.org/data/docs/specifications/ for more info.
    
    - proj: Coordinate plane on which subhalo information is projected onto. 0 is x-y plane, 1 is x-z plane,
        and 2 is y-z plane. 
    
    - max_dist: Maximum separation distance along each axis from the corresponding coordinate of pos [Mpc]. Only cells
        within max_dist will be considered in the projection. 
    
    - depth: Sets maximum depth of projected image [Mpc]. For a complete projection, depth should be set to a value
        larger than the distance between the frontmost and backmost cells of the subhalo along the projection axis.
    
    - npix: Pixel length of image.
    """

        
    if type(id) == list:
        s, pos = combine_subhalos(basePath, snapNum, id, 'gas')
        fig_title = f"Subhalos {id[0]} and {id[1]} at Snap {snapNum}"
    else:
        s = sn.loadSubhalo(basePath, snapNum, id, 'gas')
        slt = sl.loadTree(basePath, snapNum, id, fields=['SnapNum', 'SubhaloPos'], onlyMPB=True, treeName="SubLink", cache=True)
        snap_index = np.where(slt['SnapNum'] == snapNum)[0][0]
        pos = slt['SubhaloPos'][snap_index] * (1/0.6774) * df['a'][snapNum]
        fig_title = f"Subhalo {id} at Snap {snapNum}"
            

    x, y, condition, proj_label, extent = create_proj(s, snapNum, proj, pos, depth, max_dist, npix)
    
    mass = s['Masses'][condition]*1e10/0.704
    h = stack_info(x,y, mass, npix)
    h_xps = (h[1][-1] - h[1][0])/h[0].shape[0]
    h_yps = (h[2][-1] - h[2][0])/h[0].shape[1]
    h_posx = h[1][0]
    h_posy = h[2][0]
    area = h_xps * h_yps

    if infoName == 'Density':
        data = log(h[0]/area)
        units = r'log$\Sigma_{gas}$ [M$_{\odot}$ kpc$^{-2}$]'
        vmin = 5.5
        vmax = 8.5
        cmap = 'magma'
    elif infoName == 'Temperature':
        mass = s['Masses'][condition]
        mass = mass.astype('float64')
        mass = (mass*1e10*u.M_sun)/0.704
        dens = (s['Density'][condition]*1e10*u.M_sun/0.704)/(u.kpc/0.704*(df['a'][snapNum]))**3
        x_e = s['ElectronAbundance'][condition]
        ie = s['InternalEnergy'][condition]
        ie = ie.astype('float64')
        ie = ie*(u.km/u.s)**2
        vol = mass/dens

        M_e = (x_e*0.76*dens.cgs/const.m_p.cgs)*const.m_e.cgs*vol.cgs
        M_e = (M_e/const.M_sun.cgs)*u.M_sun

        M_h = (0.76*dens.cgs/const.m_p.cgs)*vol.cgs*const.m_p.cgs
        M_h = (M_h/const.M_sun.cgs)*u.M_sun

        he = stack_info(x,y, M_e, npix)
        hh = stack_info(x,y,M_h, npix)
        hx_e = np.divide(he[0], hh[0], where=hh[0] != 0)
        nx_e = hx_e*(const.m_p.cgs/const.m_e.cgs)
        h = he

        IE = (ie.cgs*mass.cgs).value * u.erg
        hE = stack_info(x,y,IE,npix)
        hM = stack_info(x,y,mass,npix)
        nie = np.divide(hE[0], hM[0], where=hM[0]!=0)

        k_B = (const.k_B.cgs.value)*(u.cm**2 * u.g)/(u.K * u.s**2)
        mu = 4/(1 + 3*0.76 + 4*0.76*(nx_e))*const.m_p.cgs
        T = (5/3 - 1) * ((nie.cgs)/k_B) * mu
        T = (T * 8.61733326 * 1e-5)/1000
        data = T.value
        
        units = r'T [keV]'
        vmin = 1
        vmax = 7
        cmap = 'afmhot'
    elif infoName == 'EnergyDissipation':
        ediss = s['EnergyDissipation'][condition]
        ediss = ediss.astype('float64')

        ediss_units = (1/df['a'][snapNum])*(1e10/0.704*u.M_sun/u.kpc)*(u.km/u.s)**3
        conversion = ediss_units.cgs.value 
        ediss = ediss*conversion

        h = stack_info(x,y,ediss, npix)
        #h = np.histogram2d(x, y, bins=npix, weights=ediss)

        data = log(h[0]/area)
        units = r'log$E_{diss}$ [erg s$^{-1}$ kpc$^{-2}$]'
        vmin = 37
        vmax = np.max(data)
        cmap='plasma'
    elif infoName == 'Machnumber':
        mach = s['Machnumber'][condition]

        h = stack_info(x, y, mach, npix)
        data = log(h[0])

        vmin = 0
        vmax = 2
        units = r'log$M$'
        cmap = 'RdYlBu_r'
    else:
        raise Exception(f"'{infoName}' is not a valid keyword for this partType.")
        
    h_wcs = create_wcs(data.shape[0], 0, 0, h_xps, h_yps, proj, np.array([h_posx, h_posy]))

    print('Generating image...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12), subplot_kw={'projection': h_wcs})
    image = ax.imshow(data, interpolation='gaussian', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=units)
    ax.set_title(fig_title, fontsize=16)
    plt.show()
    return data, ax

def plot_dm_image(basePath, id, snapNum, infoName, proj, max_dist=5, depth=20, npix=500):
    """
    Plots projected image of subhalo as seen in a given dark matter property. 
    
    Parameters
    ----------
    See plot_gas_image
    """
    
    if type(id) == list:
        s, pos = combine_subhalos(basePath, snapNum, id, 'dm')
        fig_title = f"Subhalos {id[0]} and {id[1]} at Snap {snapNum}"
    else:
        s = sn.loadSubhalo(basePath, snapNum, id, 'dm')
        slt = sl.loadTree(basePath, snapNum, id, fields=['SnapNum', 'SubhaloPos'], onlyMPB=True, treeName="SubLink", cache=True)
        snap_index = np.where(slt['SnapNum'] == snapNum)[0][0]
        pos = slt['SubhaloPos'][snap_index] * (1/0.6774) * df['a'][snapNum]
        fig_title = f"Subhalo {id} at Snap {snapNum}"
    
    x, y, condition, proj_label, extent = create_proj(s, snapNum, proj, pos, depth, max_dist, npix)
    
        
    with h5py.File(sn.snapPath(basePath, snapNum, chunkNum=0),'r') as f:
        header = dict( f['Header'].attrs.items() )
    mass_dm_particle = header['MassTable'][1]*1e10/0.704
    mass = np.ones(len(x))*mass_dm_particle
    
    h = stack_info(x,y, mass, npix)
    h_xps = (h[1][-1] - h[1][0])/h[0].shape[0]
    h_yps = (h[2][-1] - h[2][0])/h[0].shape[1]
    h_posx = h[1][0]
    h_posy = h[2][0]
    area = h_xps * h_yps
    
    if infoName == 'Mass':
        h = stack_info(x,y, mass, npix)
        data = log(h[0])
        
        units = r'log[M$_{\odot}$]'
        vmin = 8.2
        vmax = 10.6
        cmap = 'gist_earth'
    elif infoName == 'Density':
        h = stack_info(x,y, mass, npix)
        data = log(h[0]/area)
        
        units = r'log$\Sigma_{DM}$ [M$_{\odot}$ kpc$^{-2}$]'
        vmin = 6.5
        vmax = np.max(data)
        cmap = 'cividis'
    else:
        raise Exception(f"'{infoName}' is not a valid keyword for this partType.")
        
    h_wcs = create_wcs(data.shape[0], 0, 0, h_xps, h_yps, proj, np.array([h_posx, h_posy]))

    print('Generating image...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12), subplot_kw={'projection': h_wcs})
    image = ax.imshow(data, interpolation='gaussian', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=units)
    ax.set_title(fig_title, fontsize=16)
    plt.show()
    return data, extent, ax
    
    
def plot_stars_image(basePath, id, snapNum, infoName, proj, max_dist=5, depth=20, npix=500):
    """
    Plots projected image of subhalo as seen in a given stellar property. 
    
    Parameters
    ----------
    See plot_gas_image
    """
    
    if type(id) == list:
        s, pos = combine_subhalos(basePath, snapNum, id, 'stars')
        fig_title = f"Subhalos {id[0]} and {id[1]} at Snap {snapNum}"
    else:
        s = sn.loadSubhalo(basePath, snapNum, id, 'stars')
        slt = sl.loadTree(basePath, snapNum, id, fields=['SnapNum', 'SubhaloPos'], onlyMPB=True, treeName="SubLink", cache=True)
        snap_index = np.where(slt['SnapNum'] == snapNum)[0][0]
        pos = slt['SubhaloPos'][snap_index] * (1/0.6774) * df['a'][snapNum]
        fig_title = f"Subhalo {id} at Snap {snapNum}"
    
    x, y, condition, proj_label, extent = create_proj(s, snapNum, proj, pos, depth, max_dist, npix)
    
    mass = s['Masses'][condition]*1e10/0.704
    h = stack_info(x,y, mass, npix)
    h_xps = (h[1][-1] - h[1][0])/h[0].shape[0]
    h_yps = (h[2][-1] - h[2][0])/h[0].shape[1]
    h_posx = h[1][0]
    h_posy = h[2][0]
    area = h_xps * h_yps
    
    if infoName == 'Density':
        data = log(h[0]/area)
        units = r'log$\Sigma_{stars}$ [M$_{\odot}$ kpc$^{-2}$]'
        vmin = 4
        vmax = 8
        cmap = 'copper'
    else:
        raise Exception(f"'{infoName}' is not a valid keyword for this partType.")
    
    h_wcs = create_wcs(data.shape[0], 0, 0, h_xps, h_yps, proj, np.array([h_posx, h_posy]))
    print('Generating image...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12), subplot_kw={'projection': h_wcs})
    image = ax.imshow(data, interpolation='gaussian', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=units)
    ax.set_title(fig_title, fontsize=16)
    plt.show()
    return data, ax

def plot_BH_image(basePath, id, snapNum, infoName, proj, max_dist=5, depth=20, npix=500):
    """
    Plots projected image of subhalo as seen in a given Black Hole property. 
    
    Parameters
    ----------
    See plot_gas_image
    """
    
    if type(id) == list:
        s, pos = combine_subhalos(basePath, snapNum, id, 'BH')
        fig_title = f"Subhalos {id[0]} and {id[1]} at Snap {snapNum}"
    else:
        s = sn.loadSubhalo(basePath, snapNum, id, 'BH')
        slt = sl.loadTree(basePath, snapNum, id, fields=['SnapNum', 'SubhaloPos'], onlyMPB=True, treeName="SubLink", cache=True)
        snap_index = np.where(slt['SnapNum'] == snapNum)[0][0]
        pos = slt['SubhaloPos'][snap_index] * (1/0.6774) * df['a'][snapNum]
        fig_title = f"Subhalos {id} at Snap {snapNum}"
    
    x, y, condition, proj_label, extent = create_proj(s, snapNum, proj, pos, depth, max_dist, npix)
    
    mass = s['BH_Mass'][condition]*1e10/0.704
    h = stack_info(x,y, mass, npix)
    h_xps = (h[1][-1] - h[1][0])/h[0].shape[0]
    h_yps = (h[2][-1] - h[2][0])/h[0].shape[1]
    h_posx = h[1][0]
    h_posy = h[2][0]
    area = h_xps * h_yps
    
#     xmin, xmax = np.min(x), np.max(x)
#     ymin, ymax = np.min(y), np.max(y)
    
    if infoName == 'Density':
        data = log(h[0]/area)
        
        units = r'log$\Sigma_{BH}$ [M$_{\odot}$ kpc$^{-2}$]'
        vmin = 0
        vmax = np.max(data)
        cmap = 'viridis'
    else:
        raise Exception(f"'{infoName}' is not a valid keyword for this partType.")
        
    h_wcs = create_wcs(data.shape[0], 0, 0, h_xps, h_yps, proj, np.array([h_posx, h_posy]))
    print('Generating image...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12), subplot_kw={'projection': h_wcs})
    image = ax.imshow(data, interpolation='gaussian', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=units)
    ax.set_title(fig_title, fontsize=16)
    plt.show()
    return data, ax
        
        
def plot_tot_mass(basePath, id, snapNum, infoName, 
                   proj, max_dist=5, depth = 20, npix=500, dontPlot=0):
    
    if type(id) == list:
        gas, pos = combine_subhalos(basePath, snapNum, id, 'gas')
        dm, pos = combine_subhalos(basePath, snapNum, id, 'dm')
        stars, pos = combine_subhalos(basePath, snapNum, id, 'stars')
        BH, pos = combine_subhalos(basePath, snapNum, id, 'BH')
        
        fig_title = f"Subhalos {id[0]} and {id[1]} at Snap {snapNum}"
    else:
        gas = sn.loadSubhalo(basePath, snapNum, id, 'gas')
        dm = sn.loadSubhalo(basePath, snapNum, id, 'dm')
        stars = sn.loadSubhalo(basePath, snapNum, id, 'stars')
        BH = sn.loadSubhalo(basePath, snapNum, id, 'BH')
        
        slt = sl.loadTree(basePath, snapNum, id, fields=['SnapNum', 'SubhaloPos'], onlyMPB=True, treeName="SubLink", cache=True)
        snap_index = np.where(slt['SnapNum'] == snapNum)[0][0]
        pos = slt['SubhaloPos'][snap_index] * (1/0.6774) * df['a'][snapNum]
        
        fig_title = f"Subhalo {id} at Snap {snapNum}"

    
    x_gas, y_gas, condition_gas, proj_label, extent_gas = create_proj(gas, snapNum, proj, pos, depth, max_dist, npix)
    x_dm, y_dm, condition_dm, proj_label, extent_dm = create_proj(dm, snapNum, proj, pos, depth, max_dist, npix)
    x_stars, y_stars, condition_stars, proj_label, extent_stars = create_proj(stars, snapNum, proj, pos, depth, max_dist, npix)
    x_BH, y_BH, condition_BH, proj_label, extent_BH = create_proj(BH, snapNum, proj, pos, depth, max_dist, npix)
    

    
    gas_mass = gas['Masses'][condition_gas]*1e10/0.704
    stars_mass = stars['Masses'][condition_stars]*1e10/0.704
    BH_mass = BH['BH_Mass'][condition_BH]*1e10/0.704
    
    with h5py.File(sn.snapPath(basePath, snapNum, chunkNum=0),'r') as f:
        header = dict( f['Header'].attrs.items() )
    mass_dm_particle = header['MassTable'][1]*1e10/0.704
    dm_mass = np.ones(len(x_dm))*mass_dm_particle
    
    if dontPlot:
        if dontPlot == 'gas':     
            x_tot = np.concatenate((x_dm, x_stars, x_BH))
            y_tot = np.concatenate((y_dm, y_stars, y_BH))
            mass_tot = np.concatenate((dm_mass, stars_mass, BH_mass))
        elif dontPlot == 'dm':     
            x_tot = np.concatenate((x_gas, x_stars, x_BH))
            y_tot = np.concatenate((y_gas, y_stars, y_BH))
            mass_tot = np.concatenate((gas_mass, stars_mass, BH_mass))
        elif dontPlot == 'stars':     
            x_tot = np.concatenate((x_gas, x_dm, x_BH))
            y_tot = np.concatenate((y_gas, y_dm, y_BH))
            mass_tot = np.concatenate((gas_mass, dm_mass, BH_mass))
        elif dontPlot == 'BH':   
            x_tot = np.concatenate((x_gas, x_dm, x_stars))
            y_tot = np.concatenate((y_gas, y_dm, y_stars))
            mass_tot = np.concatenate((gas_mass, dm_mass, stars_mass))
    else:
        x_tot = np.concatenate((x_gas, x_dm, x_stars, x_BH))
        y_tot = np.concatenate((y_gas, y_dm, y_stars, y_BH))
        mass_tot = np.concatenate((gas_mass, dm_mass, stars_mass, BH_mass))
        
    h = stack_info(x_tot,y_tot, mass_tot, npix)
    h_xps = (h[1][-1] - h[1][0])/h[0].shape[0]
    h_yps = (h[2][-1] - h[2][0])/h[0].shape[1]
    h_posx = h[1][0]
    h_posy = h[2][0]
    area = h_xps * h_yps
    
    if infoName == 'Density':
        data = log(h[0]/area)
        
        units = r'log$\Sigma_{tot}$ [M$_{\odot}$ kpc$^{-2}$]'
        vmin = 6
        vmax = 9
        cmap = 'cubehelix'
    else:
        raise Exception(f"'{infoName}' is not a valid keyword for this partType.")
        
    h_wcs = create_wcs(data.shape[0], 0, 0, h_xps, h_yps, proj, np.array([h_posx, h_posy]))
    print('Generating image...')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12), subplot_kw={'projection': h_wcs})
    image = ax.imshow(data, interpolation='gaussian', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=units)
    ax.set_title(fig_title, fontsize=16)
    plt.show()
    return data, ax