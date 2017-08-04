#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:49:50 2017

@author: kremin
"""

import astropy.cosmology as cosmology
import astropy.constants as aconsts
import astropy.units as u

class Constants:
    def __init__(self,h=1.,Om0=0.3):
        self.h = h
        self.Om0 = Om0
        self.fidcosmo = cosmology.FlatLambdaCDM(H0=100*self.h, Om0=self.Om0)
        self.c = aconsts.c.value/1000.#3e5
        self.G = aconsts.G.value#6.67E-11
        self.solmass = aconsts.M_sun.value#1.98892e30
        self.kpc2km = u.kpc.to(u.km)#3.09e16
        self.Mpc2km = 1000.*self.kpc2km#3.08568025e19
        #self.units = u
        #self.astroconsts = aconsts
