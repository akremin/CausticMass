
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

import matplotlib.patches as patches



def define_footprint(polydef):
    ra = []
    dec = []

    with open(polydef) as f:
        for line in f:
            if line[0] != '#':
                s = line.split(' ')
                ra_i = -999
                for i in range(len(s)):
                    if s[i] != '' and ra_i == -999:
                        ra_i = float(s[i])
                        ra.append(ra_i * np.pi / 180)
                    elif s[i] != '' and ra_i != -999:
                        dec.append(float(s[i]) * np.pi / 180)
        return zip(ra, dec)

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





poly_13 = define_footprint('round13-poly.txt')
foot = patches.Polygon(poly_13, facecolor='red', edgecolor='none', alpha=0.3)
plt.figure(1, facecolor='w', edgecolor='k', figsize=(16, 8))
ax = plt.subplot(111, projection='mollweide')
ax.add_patch(foot)

# Example of adding a point at a given ra, dec. Note that angles are in radians.
# Longitude in the Mollweide projection runs from -pi to pi, so you have to wrap RA if >pi.
#ra, dec = ephem.hours('22:44:33.01'), ephem.degrees('-0:47:51.3')
print(len(clusters))
for ra,dec in zip(clusters['RA'],clusters['DEC']):
    #print(ra,dec)
    ra,dec = np.deg2rad(ra),np.deg2rad(dec)
    if ra > np.pi: ra -= 2 * np.pi
    ax.scatter(ra,dec, color='b', marker='*', s=40, alpha=0.8)

plt.grid()
print(clusters.colnames)
masses = [mass for mass in clusters['M500'] if not np.isnan(mass)]

print(masses)
print(len(masses))
plt.figure()
plt.hist(masses,bins=6)
plt.title('SZ Mass Distribution',size='xx-large')
plt.xlabel(r'$M500_{SZ}$',size='x-large')
plt.figure()
plt.hist(clusters['Nspec'],bins=20)
plt.title('Member Number Per Cluster Distribution',size='xx-large')
plt.xlabel(r'$M_{mem}$',size='x-large')
plt.figure()
plt.hist(clusters['z'],bins=6)
plt.title('Redshift Distribution',size='xx-large')
plt.xlabel('z',size='x-large')
plt.show()