#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes: Currently uses an NFW fit as the caustic surface*

CausticSurface:
    functions: findvdisp(), findvesc(), findphi(), findAofr(), restrict_gradient2(), identifyslot(), NFWfit()
    attributes: self.levels, self.r200, self.halo_scale_radius, self.halo_scale_radius_e, self.gal_vdisp,
                self.vvar, self.vesc, self.skr, self.level_elem, self.level_final, self.Ar_finalD,
                self.halo_scale_density, self.halo_scale_density_e, self.vesc_fit
"""

import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import biweight_midvariance
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from skimage import measure

import warnings

# import astropy.constants as astconsts
import astropy.units as astunits
import astropy.cosmology as cosmology

warnings.filterwarnings('ignore')


class CausticSurface:
    """
    - For now if r200 is not supplied I am using a default value of 2Mpc 
    
    - If a scale radius is not given for the cluster, then I am using a default value of r200/5.0 with uncertainty 0.01Mpc

    CausticSurface(self,r,v,ri,vi,Zi,memberflags=None,r200=2.0,maxv=5000,halo_scale_radius=None,halo_scale_radius_e=0.01,halo_vdisp=None,bin=None):

        r/v - rvalues/vvalues of galaxies

        ri/vi - x_range/y_range of grid

        Zi - density map

        memberflags = None - indices of known member galaxies to calculate a velocity dispersion

        r200 = 2.0 - critical radius of the cluster

        maxv = 5000km/s -  maximum velocity allowed

        halo_scale_radius - scale radius (default is r200/5.0)

        halo_scale_radius_e = 0.01 - uncertainty in scale radius

        halo_vdisp = None - velocity dispersion

        bin = None - if doing multiple halos, can assign an ID number
    """

    def __init__(self, cosmo=cosmology.FlatLambdaCDM(H0=100., Om0=0.3)):
        self.cosmo = cosmo

    def findsurface(self, data, ri, vi, Zi, memberflags=None, r200=2.0, maxv=5000.0, halo_scale_radius=None,
                    halo_scale_radius_e=0.01, halo_vdisp=None, bin=None, plotphase=False, beta=None,
                    mirror=True, q=10.0, Hz=100.0, edge_perc=0.1, edge_int_remove=False):

        kappaguess = np.max(Zi)  # first guess at the level
        # self.levels = np.linspace(0.00001,kappaguess,100)[::-1] #create levels (kappas) to try out
        self.levels = np.logspace(np.log10(np.min(Zi[Zi > 0] / 5.0)), np.log10(kappaguess), 200)[::-1]
        fitting_radii = np.where(
            (ri >= r200 / 3.0) & (ri <= r200))  # when fitting an NFW (later), this defines the r range to fit within

        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200 / 5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e

        if beta is None:
            self.beta = 0.2 + np.zeros(ri.size)
        else:
            self.beta = beta
        self.gb = (3 - 2.0 * self.beta) / (1 - self.beta)

        # Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = data['vel'][np.where(memberflags == 1)]
            try:
                self.gal_vdisp = biweight_midvariance(vvarcal[np.where(np.isfinite(vvarcal))], 9.0)
                print('O ya! membership calculation!')
            except:
                self.gal_vdisp = np.std(vvarcal, ddof=1)
            self.vvar = self.gal_vdisp * self.gal_vdisp
        elif halo_vdisp is not None:
            self.gal_vdisp = halo_vdisp
            self.vvar = self.gal_vdisp * self.gal_vdisp
        else:
            # Variable self.gal_vdisp
            try:
                self.findvdisp(data['rad'], data['vel'], r200, maxv)
            except:
                self.gal_vdisp = np.std(data['vel'][np.where((data['rad'] < r200) & (np.abs(data['vel']) < maxv))],
                                        ddof=1)
            self.vvar = self.gal_vdisp * self.gal_vdisp

        ##initilize arrays
        # self.vesc = np.zeros(self.levels.size)
        # self.Ar_final_opt = np.zeros((self.levels.size,ri[np.where((ri<r200) & (ri>=0))].size))
        #
        ##find the escape velocity for all level (kappa) guesses
        # for i in range(self.vesc.size):
        #    self.vesc[i],self.Ar_final_opt[i] = self.findvesc(self.levels[i],ri,vi,Zi,r200)
        #
        ##optimization equation to search for minimum value
        # self.skr = (self.vesc-4.0*self.vvar)**2

        # try:
        #    self.level_elem = np.where(self.skr == np.min(self.skr[np.isfinite(self.skr)]))[0][0]
        #    self.level_final = self.levels[self.level_elem]
        #    self.Ar_finalD = np.zeros(ri.size)
        #    for k in range(self.Ar_finalD.size):
        #        self.Ar_finalD[k] = self.findAofr(self.level_final,Zi[k],vi)
        #        if k != 0:
        #            self.Ar_finalD[k] = self.restrict_gradient2(np.abs(self.Ar_finalD[k-1]),np.abs(self.Ar_finalD[k]),ri[k-1],ri[k])
        #
        ##This exception occurs if self.skr is entirely NAN. A flag should be raised for this in the output table
        # except ValueError:
        #    self.Ar_finalD = np.zeros(ri.size)
        #

        # find contours (new)
        self.Ar_finalD = self.findcontours(Zi, self.levels, ri, vi, r200, self.vvar, Hz, int(q))

        data_e = data
        # remove outliers from edge calculation
        if edge_int_remove:
            try:
                data_e = self.edge_outlier_clip(data_e, ri, vi, Zi)
                print('completed edge_outlier_clip')
            except:
                data_e = data

        # Identify sharp phase-space edge
        numbins = 6
        perc_top = edge_perc  # what percent of top velocity galaxies per/bin used to identify surface
        numrval = (data_e['rad'][data_e['rad'] < r200]).size  # number of galaxies less than r200
        size_bin = int(np.ceil(numrval * 1.0 / numbins))  # how many galaxies are in each bin
        rsort = data_e['rad'][np.argsort(data_e['rad'])]  # sort r positions
        if mirror == True:
            vsort = np.abs(data_e['vel'][np.argsort(data_e['rad'])])  # sort absolute value of velocities by r position
        else:
            vsort = data_e['vel'][np.argsort(data_e['rad'])]  # same as above but not abs
        self.data_e = data_e
        mid_rbin = np.array([])
        avgmax = np.array([])
        avgmin = np.array([])
        mincomp = np.array([])
        for nn in range(numbins):
            vbin = vsort[nn * size_bin:(nn + 1) * size_bin]  # pick velocities in bin # nn
            if vbin.size == 0:
                if nn >= 4: break
            rbin = rsort[nn * size_bin:(nn + 1) * size_bin]  # pick radii in bin # nn
            vemax = (vbin[np.argsort(vbin)][::-1])[:int(np.ceil(vbin[
                                                                    vbin > 0.0].size * perc_top))]  # sort by velocity -> flip array from max-min -> take first edge_perc values where v>0
            vemin = (vbin[np.argsort(vbin)])[:int(
                np.ceil(vbin[vbin < 0.0].size * perc_top))]  # sort by velocity -> take first edge_perc values where v<0
            avgmax = np.append(avgmax, np.average(vemax))  # add average of top edge_perc velocities to max array
            avgmin = np.append(avgmin, np.average(vemin))  # same as above but min array
            # take the minimum of either the above || below zero caustic
            if np.isnan(avgmax)[-1] == True: break
            if np.min(vbin) >= 0:
                mincomp = np.append(mincomp, avgmax[nn])  # if no negative velocities (aka, mirrored)
            else:
                mincomp = np.append(mincomp, np.min([np.abs(avgmin[nn]), avgmax[nn]]))  # else take the minimum extreme
            mid_rbin = np.append(mid_rbin, np.median(rbin))  # take median rvalue of bin
        chi = np.array([])
        # loop through contours and find squared difference with edge extreme
        for nn in range(len(self.contours)):
            fint = interp1d(ri[ri < r200], self.contours[nn][ri < r200])  # interpolate contour
            Ar_comp = fint(mid_rbin[mid_rbin < np.max(ri[ri < r200])])  # interpolated contour
            chi = np.append(chi, np.median(
                np.abs(Ar_comp - mincomp[mid_rbin < np.max(ri[ri < r200])])))  # measure squared distance
        try:
            self.Ar_finalE = \
                np.array(self.contours)[np.isfinite(chi)][
                    np.where(chi[np.isfinite(chi)] == np.min(chi[np.isfinite(chi)]))][
                    0]  # find level with min chi value
            # self.level_finalE = ((self.levels[np.isfinite(chi)])[np.where(chi[np.isfinite(chi)] == np.min(chi[np.isfinite(chi)]))])[0] #find level with min chi value
            # self.Ar_finalE = np.zeros(ri.size)
            # for k in range(self.Ar_finalE.size):
            #    self.Ar_finalE[k] = self.findAofr(self.level_finalE,Zi[k],vi)
            #    if k != 0:
            #        self.Ar_finalE[k] = self.restrict_gradient2(np.abs(self.Ar_finalE[k-1]),np.abs(self.Ar_finalE[k]),ri[k-1],ri[k])
        except ValueError:
            self.Ar_finalE = np.zeros(ri.size)

        # fit an NFW to the resulting caustic profile.
        self.vesc_fit = self.NFWfit(ri[fitting_radii], self.Ar_finalD[fitting_radii] * np.sqrt(self.gb[fitting_radii]),
                                    self.halo_scale_radius, ri, self.gb)
        self.vesc_fit_e = self.NFWfit(ri[fitting_radii],
                                      self.Ar_finalE[fitting_radii] * np.sqrt(self.gb[fitting_radii]),
                                      self.halo_scale_radius, ri, self.gb)
        # set first element (which is NaN) equal to the second value
        self.vesc_fit[0] = self.vesc_fit[1]
        self.vesc_fit_e[0] = self.vesc_fit_e[1]

        if plotphase is True:
            s, ax = plt.subplots(1, figsize=(10, 7))
            # ax.pcolormesh(ri,vi,Zi.T)
            ax.plot(data['rad'], data['vel'], 'k.', markersize=0.5, alpha=0.8)
            for t, con in enumerate(self.contours):
                ax.plot(ri, con, c='0.4', alpha=0.5)
                ax.plot(ri, -con, c='0.4', alpha=0.5)
            ax.plot(ri, self.Ar_finalD, c='red')
            ax.plot(ri, -self.Ar_finalD, c='red')
            ax.plot(ri, self.Ar_finalE, c='blue')
            # ax.plot(mid_rbin,avgmax,c='r')
            ax.set_ylim(0, 5000)
            ax.set_xlim(0, 4)
            s.savefig('plotphase.png')
            plt.close()
            # show()

        ##Output galaxy membership
        kpc2km = astunits.kpc.to(astunits.km)
        try:
            fitfunc = lambda x, a, b: np.sqrt(
                2 * 4 * np.pi * 6.67e-20 * a * (b * kpc2km) ** 2 * np.log(1 + x / b) / (x / b))
            self.popt, self.pcov = curve_fit(fitfunc, ri, self.Ar_finalD, p0=[5e14, 1])
            self.Arfit = fitfunc(ri, self.popt[0], self.popt[1])
        except:
            fitfunc = lambda x, a: np.sqrt(
                2 * 4 * np.pi * 6.67e-20 * a * (30.0 * kpc2km) ** 2 * np.log(1 + x / 30.0) / (x / 30.0))
            self.popt, pcov = curve_fit(fitfunc, ri, self.Ar_finalD)
            self.Arfit = fitfunc(ri, self.popt[0])
        self.memflag = np.zeros(data.shape[0])
        # fcomp = interp1d(ri,self.Ar_finalD)
        # print ri.size, self.vesc_fit.size
        fcomp = interp1d(ri, self.vesc_fit)
        for k in range(self.memflag.size):
            vcompare = fcomp(data['rad'][k])
            if np.abs(vcompare) >= np.abs(data_e['vel'][k]):
                self.memflag[k] = 1

    @staticmethod
    def edge_outlier_clip(data_e, ri, vi, Zi):
        r_inside = []
        v_inside = []
        i = 0
        while ri[i] <= np.max(data_e['rad']):
            inner_el = i
            outer_el = i + 5
            inner_r = ri[inner_el]
            outer_r = ri[outer_el]
            '''
            dens = np.average(Zi[inner_el:outer_el],axis=0)
            roots = np.sort(np.abs(vi[dens>0.05]))
            databinned = data_e[np.where((data_e['rad']>=inner_r)&(data_e['rad']<outer_r))]
            if len(roots) == 0:
                root = 2 * biweight_midvariance(databinned['vel'].copy(),9.0)
            elif np.abs(roots[-1]) < 500.0:
                root = 2 * biweight_midvariance(databinned['vel'].copy(),9.0)
            elif np.abs(roots[-1]) > 3500.0:
                root = 3500.0
            else:
                root = np.abs(roots[-1])
            r_inside.extend(databinned['rad'][np.where(np.abs(databinned['vel'])<root)])
            v_inside.extend(databinned['vel'][np.where(np.abs(databinned['vel'])<root)])
            i += 5
        data_e = np.vstack((np.array(r_inside),np.array(v_inside))).T
        return data_e
            '''
            # deriv = (np.average(Zi[inner_el:outer_el],axis=0)[1:]-np.average(Zi[inner_el:outer_el],axis=0)[:-1]) \
            #           /(vi[1:]-vi[:-1])
            roots = np.sort(np.abs(vi[((np.average(Zi[inner_el:outer_el], axis=0)[1:] -
                                        np.average(Zi[inner_el:outer_el], axis=0)[:-1]) / (vi[1:] -
                                                                                           vi[:-1]))[1:] * ((np.average(
                Zi[inner_el:outer_el], axis=0)[1:] -
                                                                                                             np.average(
                                                                                                                 Zi[
                                                                                                                 inner_el:outer_el],
                                                                                                                 axis=0)[
                                                                                                             :-1]) / (
                                                                                                                vi[
                                                                                                                1:] - vi[
                                                                                                                      :-1]))[
                                                                                                           :-1] < 0]))
            databinned = data_e[np.where((data_e['rad'] >= inner_r) & (data_e['rad'] < outer_r))]
            if len(roots) > 1:
                if roots[1] < 1000.0:
                    if len(roots) > 2:
                        if roots[2] < 1000.0:
                            root = 3 * biweight_midvariance(databinned['vel'].copy(), 9.0)
                        else:
                            root = roots[2]
                    else:
                        root = 3 * biweight_midvariance(databinned['vel'].copy(), 9.0)
                else:
                    root = roots[1]
            else:
                root = 3500.0
            r_inside.extend(databinned['rad'][np.where(np.abs(databinned['vel']) < root)])
            v_inside.extend(databinned['vel'][np.where(np.abs(databinned['vel']) < root)])
            i += 5
        data_e = np.vstack((np.array(r_inside), np.array(v_inside))).T
        return data_e

    def findsurface_inf(self, data, ri, vi, Zi, Zi_inf, memberflags=None, r200=2.0, maxv=5000.0, halo_scale_radius=None,
                        halo_scale_radius_e=0.01, halo_vdisp=None, beta=None):
        """
        Identifies the caustic surface using the iso-density contours in phase space, 
        as well as the second derivative of the density (aptly named the inflection technique).
        This technique attempts to rid the caustic technique of the dreaded velocity dispersion
        calibration that is used to pick a surface.
        

        Parameters
        ----------
        data : first and second columns must be radius and velocity

        ri : x-grid values

        vi : y-grid values

        Zi : density image

        Zi_inf : second derivative of the density image

        memberflags = None : array of 1's if member 0's if not

        r200 = 2.0 : r200 value

        maxv = 5000.0 : maximum y-value

        halo_scale_radius = None : The default is actually a concentration of 5.0 
                                   which is applied later if None is given.

        halo_scale_radius_e=0.01 : error in halo_scale_radius

        halo_vdisp = None : supply cluster velocity dispersion

        beta = None : The default is actually 0.2 which is applied later in the code
                      although as of now beta is not used in this function

        Variables
        ---------
        
        """
        kappaguess = np.max(Zi)  # first thing is to guess at the level
        self.levels = np.linspace(0.00001, kappaguess, 100)[::-1]  # create levels (kappas) to try out
        fitting_radii = np.where((ri >= r200 / 3.0) & (ri <= r200))

        self.r200 = r200

        if halo_scale_radius is None:
            self.halo_scale_radius = self.r200 / 5.0
        else:
            self.halo_scale_radius = halo_scale_radius
            self.halo_scale_radius_e = halo_scale_radius_e

        # c_guess = np.array([halo_srad])#np.linspace(1.0,12.0,100)
        # density_guess = np.linspace(1e13,5e16,1000)

        if beta is None:
            self.beta = 0.2 + np.zeros(ri.size)
        else:
            self.beta = beta
        self.gb = (3 - 2.0 * self.beta) / (1 - self.beta)

        # Calculate velocity dispersion with either members, fed value, or estimate using 3.5sigma clipping
        if memberflags is not None:
            vvarcal = data['vel'][np.where(memberflags == 1)]
            try:
                self.gal_vdisp = biweight_midvariance(vvarcal[np.where(np.isfinite(vvarcal))], 9.0)
                print('O ya! membership calculation!')
            except:
                self.gal_vdisp = np.std(vvarcal, ddof=1)
            self.vvar = self.gal_vdisp ** 2
        elif halo_vdisp is not None:
            self.gal_vdisp = halo_vdisp
            self.vvar = self.gal_vdisp ** 2
        else:
            # Variable self.gal_vdisp
            try:
                self.findvdisp(data['rad'], data['vel'], r200, maxv)
            except:
                self.gal_vdisp = np.std(data['vel'][np.where((data['rad'] < r200) & (np.abs(data['vel']) < maxv))],
                                        ddof=1)
            self.vvar = self.gal_vdisp ** 2

        self.Ar_final_opt = np.zeros((self.levels.size, ri[
            np.where((ri < r200) & (ri >= 0))].size))  # 2D array: density levels x velocity profile
        self.inf_vals = np.zeros((self.levels.size, ri[
            np.where((ri < r200) & (ri >= 0))].size))  # 2D array: density levels x inflection profile
        # s = figure()
        # ax = s.add_subplot(111)
        for i in range(self.levels.size):  # find the escape velocity for all level (kappa) guesses
            self.Ar_final_opt[i], self.inf_vals[i] = self.findvesc2(self.levels[i], ri, vi, Zi, Zi_inf, r200)
            # ax.plot(ri[np.where((ri<r200) & (ri>=0))],np.abs(self.Ar_final_opt[i]),c='black',alpha=0.4) #plot each density contour
        self.inf_avg = np.average(self.inf_vals.T[fitting_radii],
                                  axis=0)  # average inflection along each contour surface
        self.Ar_avg = np.average(self.Ar_final_opt, axis=1)  # average velocity along each contour surface inside r200

        # Need to identify maximum average inflection, so smooth the measurement. Might want to do this a non-parametric way
        # tryfit = np.polyfit(self.levels,self.inf_avg,7)
        # self.infyvals = tryfit[0]*self.levels**7+tryfit[1]*self.levels**6+tryfit[2]*self.levels**5+tryfit[3]*self.levels**4+tryfit[4]*self.levels**3+tryfit[5]*self.levels**2+tryfit[6]*self.levels+tryfit[7]
        tryfit = np.polyfit(self.Ar_avg, self.inf_avg, 7)
        self.infyvals = tryfit[0] * self.Ar_avg ** 7 + tryfit[1] * self.Ar_avg ** 6 + tryfit[2] * self.Ar_avg ** 5 + \
                        tryfit[3] * self.Ar_avg ** 4 + tryfit[4] * self.Ar_avg ** 3 + tryfit[5] * self.Ar_avg ** 2 + \
                        tryfit[6] * self.Ar_avg + tryfit[7]

        self.inf_std = np.std(self.inf_vals.T[fitting_radii], axis=0)  # std of inflection along each caustic surface
        # self.level_elem = (self.levels[Ar_avg>np.sqrt(vvar)])[np.where(self.inf_avg[Ar_avg>np.sqrt(vvar)] == np.max(self.inf_avg[Ar_avg>np.sqrt(vvar)]))]
        self.level_elem = self.levels[np.where(self.inf_avg == np.max(self.inf_avg))][0]
        # low_zone = np.where((np.average(np.abs(self.Ar_final_opt),axis=1)>np.max(v)/2.0) & (np.average(np.abs(self.Ar_final_opt),axis=1)<np.max(v)))
        # high_zone = np.where((np.average(np.abs(self.Ar_final_opt),axis=1)>np.max(data['vel'])/2.0))
        # level_elem_low = self.levels[low_zone][np.where(self.inf_avg[low_zone] == np.min(self.inf_avg[low_zone]))][-1]
        # level_elem_high = self.levels[high_zone][np.where(self.inf_avg[high_zone] == np.max(self.inf_avg[high_zone]))][-1]
        try:
            self.level_elem_high = (self.levels[1:-1][np.where(
                (self.infyvals[1:-1] > self.infyvals[2:]) & (self.infyvals[1:-1] > self.infyvals[:-2]))])[-1]
        except IndexError:
            self.level_elem_high = self.levels[0]
        self.Ar_final_high = np.zeros(ri.size)
        # self.Ar_final_low = np.zeros(ri.size)
        for i in range(ri.size):
            self.Ar_final_high[i] = self.findAofr(self.level_elem_high, Zi[i], vi)
            # self.Ar_final_low[i] = self.findAofr(level_elem_low,Zi[i],vi)
            if i > 0:
                self.Ar_final_high[i] = self.restrict_gradient2(np.abs(self.Ar_final_high[i - 1]),
                                                                np.abs(self.Ar_final_high[i]), ri[i - 1], ri[i])
                # self.Ar_final_low[i] = self.restrict_gradient2(np.abs(self.Ar_final_low[i-1]),np.abs(self.Ar_final_low[i]),ri[i-1],ri[i])
        # Ar_final = self.Ar_final_opt[np.where(self.inf_avg == np.max(self.inf_avg))][0]
        # self.Ar_final = (self.Ar_final_high+self.Ar_final_low)/2.0
        self.Ar_finalD = self.Ar_final_high

        ##Output galaxy membership
        kpc2km = 3.09e16
        try:
            fitfunc = lambda x, a, b: np.sqrt(
                2 * 4 * np.pi * 6.67e-20 * a * (b * kpc2km) ** 2 * np.log(1 + x / b) / (x / b))
            self.popt, self.pcov = curve_fit(fitfunc, ri, self.Ar_final)
            self.vesc_fit = fitfunc(ri, self.popt[0], self.popt[1])
        except:
            fitfunc = lambda x, a: np.sqrt(
                2 * 4 * np.pi * 6.67e-20 * a * (30.0 * kpc2km) ** 2 * np.log(1 + x / 30.0) / (x / 30.0))
            self.popt, self.pcov = curve_fit(fitfunc, ri, self.Ar_finalD)
            self.vesc_fit = fitfunc(ri, self.popt[0])

        self.memflag = np.zeros(data.shape[0])
        # fcomp = interp1d(ri,self.Ar_finalD)
        # print ri.size, self.vesc_fit.size
        fcomp = interp1d(ri, self.vesc_fit)
        for k in range(self.memflag.size):
            vcompare = fcomp(data['rad'][k])
            if np.abs(vcompare) >= np.abs(data_e['vel'][k]):
                self.memflag[k] = 1

                # ax.plot(ri,np.abs(self.Ar_final),c='red',lw=2)
                # ax.plot(ri,vesc_fit,c='green',lw=2)
                # ax.plot(r,v,'k.')
                # pcolormesh(ri,vi,Zi_inf.T)
                # ax.set_ylim(0,3500)
                # savefig('/nfs/christoq_ls/giffordw/flux_figs/surfacetests/nideal/'+str(bin-1)+'.png')
                # close()

    def causticmembership(self, data, ri, caustics):
        self.memflag = np.zeros(data.shape[0])
        for k in range(self.memflag.size):
            # diff = data['rad'][k]-ri
            xrange_up = ri[np.where(ri > data['rad'][k])][0]
            xrange_down = ri[np.where(ri <= data['rad'][k])][-1]
            c_up = np.abs(caustics[np.where(ri > data['rad'][k])])[0]
            c_down = np.abs(caustics[np.where(ri <= data['rad'][k])])[-1]
            slope = (c_up - c_down) / (xrange_up - xrange_down)
            intercept = c_up - slope * xrange_up
            vcompare = slope * data['rad'][k] + intercept
            if vcompare >= np.abs(data_e['vel'][k]):
                self.memflag[k] = 1

    def findvdisp(self, r, v, r200, maxv):
        """
        Use astropy.stats biweight sigma clipping variance for the velocity dispersion
        """
        v_cut = v[np.where((r < r200) & (np.abs(v) < maxv))]
        try:
            self.gal_vdisp = biweight_midvariance(v_cut[np.where(np.isfinite(v_cut))], 9.0)
        except:
            self.gal_vdisp = np.std(v_cut, ddof=1)

    def findvesc(self, level, ri, vi, Zi, r200):
        """
        Calculate vesc^2 by first calculating the integrals in Diaf 99 which are not labeled but in 
        between Eqn 18 and 19
        """
        useri = ri[np.where((ri < r200) & (ri >= 0))]  # look only inside r200
        Ar = np.zeros(useri.size)
        phir = np.zeros(useri.size)
        # loop through each dr and find the caustic amplitude for the given level (kappa) passed to this function
        for i in range(useri.size):
            Ar[i] = self.findAofr(level, Zi[np.where((ri < r200) & (ri >= 0))][i], vi)
            if i > -1:  # to fix the fact that the first row of Zi may be 'nan'
                # The Serra paper also restricts the gradient when the ln gradient is > 2. We use > 3
                Ar[i] = self.restrict_gradient2(np.abs(Ar[i - 1]), np.abs(Ar[i]), useri[i - 1], useri[i])
                philimit = np.abs(Ar[i])  # phi integral limits
                phir[i] = self.findphir(Zi[i][np.where((vi < philimit) & (vi > -philimit))],
                                        vi[np.where((vi < philimit) & (vi > -philimit))])
        return (np.trapz(Ar ** 2 * phir, useri) / np.trapz(phir, useri), Ar)

    def findvesc2(self, level, ri, vi, Zi, Zi_inf, r200):
        """
        Used by findsurface_inf to identify caustic surfaces

        Parameters
        ----------
        level = density value

        ri = x-grid values

        vi = y-grid values

        Zi = density image

        Zi_inf = second derivative of density image

        r200 = r200 of cluster

        Returns
        -------
        (Ar,inf_val)

        Ar = caustic surface

        inf_val = inflection values along caustic surface
        """
        useri = ri[np.where((ri < r200) & (ri >= 0))]  # look only inside r200
        Ar = np.zeros(useri.size)
        inf_val = np.zeros(useri.size)
        for i in range(useri.size):
            Ar[i] = self.findAofr(level, Zi[np.where((ri < r200) & (ri >= 0))][i], vi)
            if i > 0:
                Ar[i] = self.restrict_gradient2(np.abs(Ar[i - 1]), np.abs(Ar[i]), useri[i - 1], useri[i])
            inf_val[i] = Zi_inf[i][np.where(np.abs(vi - Ar[i]) == np.min(np.abs(vi - Ar[i])))][0]
        return Ar, inf_val

    @staticmethod
    def findphir(shortZi, shortvi):
        short2Zi = np.ma.masked_array(shortZi)
        vi = shortvi[np.ma.where(np.ma.getmaskarray(short2Zi) == False)]
        Zi = short2Zi[np.ma.where(np.ma.getmaskarray(short2Zi) == False)]

        vi = vi[np.isfinite(Zi)]
        Zi = Zi[np.isfinite(Zi)]
        x = np.trapz(Zi.compressed(), vi)
        return x

    def findAofr(self, level, Zi, vgridvals):
        """
        Finds the velocity where kappa is
        """
        # dens0 = Zi[np.where(vgridvals>=0)][0]
        dens0 = np.max(Zi)
        # if dens0:#dens0 >= level:
        if dens0 >= level:
            maxdens = 0.0  # v value we are centering on
            highvalues = Zi[np.where(vgridvals >= maxdens)]  # density values above the center v value maxdens
            lowvalues = Zi[np.where(vgridvals < maxdens)]  # density values below the center v value maxdens
            highv = vgridvals[np.where(vgridvals >= maxdens)]  # v values above the center v value maxdens
            lowv = vgridvals[np.where(vgridvals < maxdens)]  # v values below the center v value maxdens
            highslot = self.identifyslot(highvalues, level)  # identify the velocity
            flip_lowslot = self.identifyslot(lowvalues[::-1], level)
            lowslot = lowvalues.size - flip_lowslot
            if len(lowv) == 0 or len(highv) == 0:  # probably all zeros
                highamp = lowamp = 0
                return highamp
            if highslot == highv.size:
                highamp = highv[-1]
            if lowslot == 0:
                lowamp = lowv[0]
            if highslot == 0 or lowslot == lowv.size:
                highamp = lowamp = 0
            if highslot != 0 and highslot != highv.size:
                highamp = highv[highslot] - (highv[highslot] - highv[highslot - 1]) * (
                    1 - (highvalues[highslot - 1] - level) / (highvalues[highslot - 1] - highvalues[highslot]))
            if lowslot != 0 and lowslot != lowv.size:
                lowamp = lowv[lowslot - 1] - (lowv[lowslot - 1] - lowv[lowslot]) * (
                    1 - (lowvalues[lowslot] - level) / (lowvalues[lowslot] - lowvalues[lowslot - 1]))
            if not highamp and not lowamp:
                return 0
            elif not highamp:
                return lowamp
            elif not lowamp:
                return highamp
            if np.abs(highamp) >= np.abs(lowamp):
                return lowamp
            if np.abs(highamp) < np.abs(lowamp):
                return highamp
        else:
            return 0  # no maximum density exists

    @staticmethod
    def restrict_gradient2(pastA, newA, pastr, newr):
        """
        It is necessary to restrict the gradient the caustic can change at in order to be physical
        """
        gradu = 0.5
        gradd = 2.0
        if pastA <= newA:
            if (np.log(newA) - np.log(pastA)) / (np.log(newr) - np.log(pastr)) > gradu and pastA != 0:
                dr = np.log(newr) - np.log(pastr)
                return np.exp(np.log(pastA) + gradu * dr)
            else:
                return newA
        if pastA > newA:
            if (np.log(newA) - np.log(pastA)) / (np.log(newr) - np.log(pastr)) < -gradd and pastA != 0:
                dr = np.log(newr) - np.log(pastr)
                return np.exp(np.log(pastA) - gradd * dr)
            else:
                return newA

    @staticmethod
    def identifyslot(dvals, level):
        '''This function takes the density values for a given r grid value either above or below
        the v grid value that corresponds to the maximum density at the r slice and returns the indici
        where the level finally falls below the given level. Density values should be in order
        starting with the corresponding value to the v value closest to the maximum and working toward
        the edges (high to low density in general).'''

        slot = dvals.size - 1
        if len(dvals[dvals > level]) == 0:
            slot = 0
            return slot
        for i in range(dvals.size):
            if dvals[i] == 0.0:
                continue
            if i < np.where(dvals > level)[0][0]:
                continue
            if level >= dvals[i]:
                if i != 0:
                    slot = i - 1
                    break
                else:
                    slot = i
                    break
        return slot

    def NFWfit(self, ri, Ar, halo_srad, ri_full, g_b):
        min_func = lambda x, d0: np.sqrt(
            2 * 4 * np.pi * 4.5e-48 * d0 * (halo_srad) ** 2 * np.log(1 + x / halo_srad) / (x / halo_srad)) * 3.08e19
        v0 = np.array([1e15])
        out = curve_fit(min_func, ri, Ar, v0[:], maxfev=2000)
        self.halo_scale_density = out[0][0]
        try:
            self.halo_scale_density_e = np.sqrt(out[1][0][0])
        except:
            self.halo_scale_density_e = 1e14
        return np.sqrt(
            2 * 4 * np.pi * 4.5e-48 * self.halo_scale_density * (halo_srad) ** 2 * np.log(1 + ri_full / halo_srad) / (
                ri_full / halo_srad)) * 3.08e19 / np.sqrt(g_b)

    def findcontours(self, Zi, levels, ri, vi, r200, vvar, Hz=100.0, q=10):
        '''This function will use skimage find_contours() to locate escape surfaces'''
        self.contours = []  # initialize contour array
        rspace = ri[1] - ri[0]  # find r spacing
        for i, level in enumerate(levels):
            fcontours = measure.find_contours(Zi, level)

            for j, contour in enumerate(fcontours):  # sometimes 1 level has more than one contour
                # rescale x & y
                xcont = contour[:, 0] * rspace
                ycont = (contour[:, 1] - vi.size / 2.0 - 1) * Hz * q * rspace

                # only consider contours that are "full" and don't loop back only in positive or negative space
                if np.max(xcont) >= 0.4 and np.min(xcont) <= 0.05 and np.max(ycont) > 0 > np.min(ycont):
                    xcont_u, ycont_u = xcont[ycont > 0], ycont[ycont > 0]  # find positive/negative contours
                    xcont_d, ycont_d = xcont[ycont < 0], ycont[ycont < 0]
                    y_u = np.zeros(ri.size)  # initialize positive, negative, and final arrays
                    y_d = np.zeros(ri.size)
                    y_f = np.zeros(ri.size)

                    for k, xplace in enumerate(ri):  # loop over r grid values
                        # match contour grid to r grid (nearest neighbor interpolate)
                        try:
                            y_u[k] = ycont_u[np.where((xcont_u > xplace - 0.01) & (xcont_u < xplace + 0.01))].max()
                        except:
                            y_u[k] = 0.0

                        try:
                            y_d[k] = ycont_d[np.where((xcont_d > xplace - 0.01) & (xcont_d < xplace + 0.01))].max()
                        except:
                            y_d[k] = 0.0

                        # apply gradient restriction for positive and negative cases.
                        if k != 0:
                            y_u[k] = self.restrict_gradient2(np.abs(y_u[k - 1]), np.abs(y_u[k]), ri[k - 1], ri[k])
                            y_d[k] = self.restrict_gradient2(np.abs(y_d[k - 1]), np.abs(y_d[k]), ri[k - 1], ri[k])

                        y_f[k] = np.min([y_u[k], np.abs(y_d[k])])  # take minimum value of positive and negative arrays
                    self.contours.append(y_f)

        # now I need to do the average calculation in Diaferio 99
        # because an integral is involved, I don't want to do this for all contours.
        # instead I select the 25% around the preliminary closest average and do
        # the full calculation for them
        avg_contours = np.average(np.asarray(self.contours).T[ri <= r200] ** 2.0, axis=0)  # prelim avg
        avg_cont_diff = (avg_contours - 4.0 * vvar) ** 2.0  # prelim diff calc
        i_sort = np.argsort(avg_cont_diff)  # sort indices based on prelim diff
        i_sort_small = i_sort[:np.int(i_sort.size / 4.0)]
        tot_avg = np.zeros(i_sort_small.size)
        for i, isrt in enumerate(i_sort_small):
            Ar = self.contours[isrt]
            lessr200 = np.where(ri <= r200)
            useri = ri[lessr200]
            Ar = Ar[lessr200]
            phir = np.zeros(useri.size)
            for j in range(useri.size):
                philimit = np.abs(Ar[j])  # phi integral limits
                phir[j] = np.sum(Zi[j][np.where((vi < philimit) & (vi > -philimit))])
                # phir[j] = self.findphir(Zi[j][np.where((vi<philimit) & (vi>-philimit))],vi[np.where((vi<philimit) & (vi>-philimit))])
            # print np.trapz(phir,useri)
            # tot_avg[i] = np.trapz(Ar**2*phir,useri) / np.trapz(phir,useri)
            tot_avg[i] = np.sum(Ar ** 2 * phir) / np.sum(phir)
        final_contour = self.contours[i_sort_small[((tot_avg - 4.0 * vvar) ** 2.0).argmin()]]
        print('complete')
        return final_contour
