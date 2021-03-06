#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:23:20 2017

@author: kremin
"""

from astropy.table import Table
import astropy.io as io
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as astconsts
from astropy.cosmology import Planck15
import astropy.constants as astroconsts
import pdb

import numpy as np
import matplotlib.pyplot as plt
import os

import causticpy

from causticpy import Caustic
# CausticSurface = causticsurface.CausticSurface
# MassInfo = datastructs.MassInfo
# CausticFitResults = causticpy.datastructs.CausticFitResults
from causticpy import ClusterData

### Define any needed constants
c = 3e5  # km/s
lowz_cut = 0.25
highz_cut = 0.55
lowmem_cut = 25
highmem_cut = 400

gapper = False

clusters = Table.read('./merged_cluster_table.fits', format='fits')
galaxies = Table.read('./galaxies_valadded_table.fits', format='fits')

clusters = clusters[clusters['NotMasked']]
galaxies = galaxies[galaxies['NotMasked']]

redshift_cut = ((clusters['z'] < lowz_cut) | (clusters['z'] > highz_cut))
number_cut = ((clusters['Nmem'] < lowmem_cut) | (clusters['Nmem'] > highmem_cut))

red_num_cut = (redshift_cut | number_cut)

bad_cluster_names = clusters['CName'][red_num_cut]
clusters['NotMasked'][red_num_cut] = False
for name in bad_cluster_names:
    bad_inds = np.where(name == galaxies['CName'])[0]
    galaxies['NotMasked'][bad_inds] = False

clusters = clusters[clusters['NotMasked']]
galaxies = galaxies[galaxies['NotMasked']]
caustic_fitter = Caustic(h=1., Om0=0.3, rlimit=4.0, vlimit=3500, kernal_stretch=10.0,
                         rgridmax=6.0, vgridmax=5000.0, cut_sample=True, edge_int_remove=False,
                         gapper=gapper, mirror=True, inflection=False, edge_perc=0.1, fbr=0.65)

fbeta_masses = []
vdisps = []
for cluster in clusters:
    clustername = cluster['CName']
    clustermembers = galaxies[galaxies['CName'] == clustername]
    if len(clustermembers) < 30:
        continue
    ras = clustermembers['RAdeg']
    decs = clustermembers['DEdeg']
    mags = clustermembers['imag']
    specs = clustermembers['zspec']
    velocities = clustermembers['zg-zc'] * astconsts.c.to(u.km / u.s).value
    radii = clustermembers['CSep_Mpc']

    cluster_obj = ClusterData(ras=ras, decs=decs, specs=specs, gal_mags=mags, gal_memberflags=None,
                              clus_ra=cluster['RA'], clus_dec=cluster['DEC'], clus_z=cluster['z'],
                              gal_r=radii, gal_v=velocities, r200=None, clus_vdisp=None,
                              clus_name=clustername, abs_flag=False)

    print("\n\n\n-----------------------------------------------------------")
    print("Name\tRA\tDEC\tz")
    print(clustername, cluster['RA'], cluster['DEC'], cluster['z'])

    # try:
    results = caustic_fitter.run_caustic(cluster_obj)
    fbeta_masses.append(float(results.M200_est_fbeta))
    vdisps.append(float(results.vdisp_gal))
    # except:
    #     print("That one didn't work. Continuing")
    # pdb.set_trace()
    # try:
    plt.figure()
    plt.title("{}, z={}".format(clustername, cluster['z']), fontsize='xx-large')
    plt.xlabel('Radial Dist [Mpc]', fontsize='x-large')
    plt.ylabel('Velocity [km/s]', fontsize='x-large')
    xs = caustic_fitter.r_range
    yprofp = results.caustic_profile
    yprofm = -1 * results.caustic_profile
    yfitp = results.caustic_fit
    yfitm = -1 * results.caustic_fit
    vmems = velocities[(results.memflag == 1)]
    vnonmems = velocities[(results.memflag != 1)]
    rmems = radii[(results.memflag == 1)]
    rnonmems = radii[(results.memflag != 1)]
    plt.plot(rmems, vmems, 'b.', alpha=0.6, label='Mem')
    plt.plot(rnonmems, vnonmems, 'k.', alpha=0.4, label='NonMem')
    plt.plot(xs, yprofp, 'b-', label='C Profile')
    plt.plot(xs, yfitp, 'r-', label='C Fit')
    plt.plot(xs, yprofm, 'b-')
    plt.plot(xs, yfitm, 'r-')
    plt.ylim([-4000, 4000])
    plt.xlim([0, 1.5])
    plt.legend(loc='best')
    plt.savefig('{}__z-{}__massprofile.png'.format(clustername, cluster['z']), dpi=600)
    # except:
    #     pass
    print("\n-----------------------------------------------------------")
plt.figure()
plt.subplot(211)
plt.title('Masses [M_sol]', fontsize='x-large')
plt.hist(fbeta_masses)
plt.subplot(212)
plt.title("Velocity Dispersions [km/s]", fontsize='x-large')
plt.hist(vdisps)
plt.savefig('masses_and_vdisps_spt.png', dpi=600)
