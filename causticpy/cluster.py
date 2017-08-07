#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:15:21 2017

@author: kremin
"""
import numpy as np
import astropy.constants as astconsts


class ClusterData:
    def __init__(self,data,gal_mags=None,gal_memberflag=None,clus_ra=None,clus_dec=None,clus_z=None,\
                    gal_r=None,gal_v=None,r200=None,clus_vdisp=None):
        if clus_ra == None:
            #calculate average ra from galaxies
            self.clus_ra = np.median(data[:,0])
        if clus_dec == None:
            #calculate average dec from galaxies
            self.clus_dec = np.median(data[:,1])       
        if gal_r == None:
            #Reduce data set to only valid redshifts
            self.data_spec = data[np.where((np.isfinite(data[:,2])) & (data[:,2] > 0.0) & (data[:,2] < 5.0))]
        else:
            self.data_spec = data[np.where(np.isfinite(gal_v))]
        if clus_z == None:
            #calculate average z from galaxies
            self.clus_z = np.median(self.data_spec[:,2])
        if gal_v == None:
            self.v = (astconsts.c.value/1000.)*(self.data_spec[:,2] - self.clus_z)/(1+self.clus_z)
        else:
            self.v = gal_v
        # Now define the rest where we can't estimate defaults
        self.r = gal_r
        self.gal_mags = gal_mags
        self.gal_memberflag = gal_memberflag
        self.r200 = r200
        self.clus_vdisp = clus_vdisp
        