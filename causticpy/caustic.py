#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes: Currently uses an NFW fit as the caustic surface*

Caustic:
    functions: findangle(), set_sample(), shiftgapper(), _gaussian_kernel()
    attributes: self.clus_ra, self.clus_dec, self.clus_z, r200, self.r, self.v, self.data, self.data_set,
                self.ang_d, self.angle, self.x_scale, self.y_scale, self.x_range, self.y_range, self.ksize_x, 
                self.ksize_y, self.img, self.img_grad, self.img_inf
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from astropy.stats import biweight_midvariance
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
import pdb
import warnings
import astropy.constants as astconsts
import astropy.units as astunits
import astropy.cosmology as cosmology

from causticsurface import CausticSurface

warnings.filterwarnings('ignore')



class CausticFitResults:
    '''
    Initialized as a trivial class, but meant to be used to store all of the 
    fit values generated by the Caustic class in the run_caustic function
    '''
    pass


class MassInfo:
    '''
    A simple data object that holds information related to the caustic estimated
    mass values
    '''
    def __init__(self, g_b=None, conc=None, f_beta=None, massprofile=None, \
                 avg_density=None, r200_est=None, M200=None):
        self.g_b = g_b
        self.conc = conc
        self.f_beta = f_beta
        self.massprofile = massprofile
        self.avg_density = avg_density
        self.r200_est = r200_est
        self.M200 = M200
        

class Caustic:
    """
    Required input: Galaxy RA,DEC,Z which must be first 3 columns in data input
    
    Optional input: Galaxy mags,memberflag   Cluster RA,DEC,Z,rlimit,vlimit,H0
    
    - if the optional Cluster inputs are not given, average values are calculated. It is far better for the user
    to calculate their own values and feed them to the module than rely on these estimates. The defaults for 
    rlimit = 4 and vlimit = +/- 3500km/s
    
    - User can submit a 2D data array if there are additional galaxy attribute columns not offered by default
    that can be carried through in the opperations for later.

    data -- 2d array with columns starting with RA,DEC,Z
    """

    def __init__(self,h=1.,Om0=0.3,rlimit=4.0,vlimit=3500,kernal_stretch=10.0,
                 rgridmax=6.0,vgridmax=5000.0,cut_sample=True,edge_int_remove=False,\
                 gapper=True,mirror=True,absflag=False,inflection=False,edge_perc=0.1,fbr=0.65):
        
        self.cosmo = cosmology.FlatLambdaCDM(H0=100.*h, Om0=Om0)
        self.rlimit = rlimit
        self.vlimit = vlimit
        self.kernal_stretch = kernal_stretch
        self.rgridmax = rgridmax
        self.vgridmax = vgridmax
        self.cut_sample = cut_sample
        self.edge_int_remove = edge_int_remove
        self.gapper = gapper
        self.mirror = mirror
        self.absflag = absflag
        self.inflection = inflection
        self.edge_perc = edge_perc
        self.fbr= fbr

    #def change_param(self,h=None,Om0=None,rlimit=None,vlimit=None,kernal_stretch=None,
    #             rgridmax=None,vgridmax=None,cut_sample=None,edge_int_remove=None,\
    #             gapper=None,mirror=None,absflag=None,inflection=None,edge_perc=None,fbr=None):
    #    
    #    self.cosmo = cosmology.FlatLambdaCDM(H0=100.*h, Om0=Om0)
    #    results = CausticFitResults()
    #    self.rlimit = rlimit
    #    self.vlimit = vlimit
    #    self.kernal_stretch = kernal_stretch
    #    self.rgridmax = rgridmax
    #    self.vgridmax = vgridmax
    #    self.cut_sample = cut_sample
    #    self.edge_int_remove = edge_int_remove
    #    self.gapper = gapper
    #    self.mirror = mirror
    #    self.absflag = absflag
    #    self.inflection = inflection
    #    self.edge_perc = edge_perc
    #    self.fbr= fbr
    
    def run_caustic(self,cluster_data):
        S = CausticSurface(self.cosmo)
        results = CausticFitResults()
        
        clus_z, r200, v, gal_memberflag = cluster_data.clus_z, cluster_data.gal_r, \
                                            cluster_data.gal_v, cluster_data.gal_memberflag
        if cluster_data.gal_r == None:
            #calculate angular diameter distance. 
            #Variable self.ang_d
            ang_d = self.cosmo.fidcosmo.angular_diameter_distance(clus_z)
            #calculate the spherical angles of galaxies from cluster center.
            #Variable self.angle
            angle = self.findangle(cluster_data.data_spec[:,0],cluster_data.data_spec[:,1],\
                                   cluster_data.clus_ra,cluster_data.clus_dec)
            r = angle*ang_d
        else:
            r = cluster_data.gal_r

        #calculate H(z)
        Hz = self.cosmo.fidcosmo.H(clus_z).value  #self.cosmo.h*100*np.sqrt(0.25*(1+self.clus_z)**3 + 0.75)
        #hz = self.Hz/100   #self.Hz / 100.0  #little h(z)

        #package galaxy data, USE ASTROPY TABLE HERE!!!!!
        if gal_memberflag is None:
            data_table = np.vstack((r,v,cluster_data.data_spec.T)).T
        else:
            data_table = np.vstack((r,v,cluster_data.data_spec.T,gal_memberflag)).T
        
        #reduce sample within limits
        if self.cut_sample == True:
            data_set = self.set_sample(data_table,rlimit=self.rlimit,vlimit=self.vlimit)
        else:
            data_set = data_table

        if data_set.shape[0] < 2:
            print('Encountered Error: Data set has too few elements. Check the r and v objects. Could indicate wrong cluster/galaxy positions or redshifts')
            return 0
        
        #further select sample via shifting gapper
        if self.gapper == True:
            data_set = self.shiftgapper(data_set)
        print('DATA SET SIZE',data_set[:,0].size)
        
        ##tries to identify double groups that slip through the gapper process
        #upper_max = np.max(data_set[:,1][np.where((data_set[:,1]>0.0)&(data_set[:,0]<1.0))])
        #lower_max = np.min(data_set[:,1][np.where((data_set[:,1]<0.0)&(data_set[:,0]<1.0))])
        #if np.max(np.array([upper_max,-lower_max])) > 1000.0+np.min(np.array([upper_max,-lower_max])):
        #    data_set = data_set[np.where(np.abs(data_set[:,1])<1000.0+np.min(np.array([upper_max,-lower_max])))]
        
        
        #measure Ngal above mag limit
        try:
            if self.absflag:
                abs_mag = data_table[:,5]
            else:
                abs_mag = data_table[:,7] - self.cosmo.fidcosmo.distmod(clus_z)
            results.Ngal_1mpc = r[np.where((abs_mag < -19.55) & (r < 0.5) & (np.abs(v) < 3500))].size
        except IndexError:
            abs_mag = np.zeros(data_table[:,0].size)
            results.Ngal_1mpc = None
        
        if r200 == None:
            vdisp_prelim = biweight_midvariance(data_set[:,1][np.where(data_set[:,0]<3.0)],9.0)
            if np.sum(abs_mag) == 0:
                r200_mean_prelim = 0.002*vdisp_prelim + 0.40
                r200 = r200_mean_prelim/1.7
            else:
                r200 = results.Ngal_1mpc**0.51*np.exp(-1.86)
            
            
            if r200 > 3.0:
                r200 = 3.0
            #if 3.0*r200 < 6.0:
            #    rlimit = 3.0*r200
            #else:
            #    rlimit = 5.5

        else:
            r200 = r200
            if r200 > 3.0:
                r200 = 3.0
        print('Pre_r200=',r200)

        if self.mirror == True:
            print('Calculating Density w/Mirrored Data')
            self._gaussian_kernel(np.append(data_set[:,0],data_set[:,0]),\
                                 np.append(data_set[:,1],-data_set[:,1]),r200,\
                                 normalization=Hz,scale=self.kernal_stretch,\
                                 xmax=self.rgridmax,ymax=self.vgridmax)
        else:
            print('Calculating Density')
            self._gaussian_kernel(data_set[:,0],data_set[:,1],r200,\
                                 normalization=Hz, scale=self.kernal_stretch,\
                                 xmax=self.rgridmax,ymax=self.vgridmax)
        img_tot = self.img/np.max(np.abs(self.img))
        #img_grad_tot = self.img_grad/np.max(np.abs(self.img_grad))
        #img_inf_tot = self.img_inf/np.max(np.abs(self.img_inf))
        
        if cluster_data.clus_vdisp is None:
            #pre_vdisp = 9.15*Ngal_1mpc+350.32
            #print 'Pre_vdisp=',pre_vdisp
            #print 'Ngal<1Mpc=',Ngal_1mpc
            v_cut = data_set[:,1][np.where((data_set[:,0]<r200) & (np.abs(data_set[:,1])<self.vlimit))]
            try:
                pre_vdisp2 = biweight_midvariance(v_cut[np.where(np.isfinite(v_cut))],9.0)
            except:
                pre_vdisp2 = np.std(v_cut,ddof=1)
            print('Vdisp from galaxies=',pre_vdisp2)
            #if data_set[:,0].size < 15: 
            #    v_unc = 0.35
            #    c_unc_sys = 0.75
            #    c_unc_int = 0.35
            #elif data_set[:,0].size < 25 and data_set[:,0].size >= 15: 
            #    v_unc = 0.30
            #    c_unc_sys = 0.55
            #    c_unc_int = 0.22
            #elif data_set[:,0].size < 50 and data_set[:,0].size >= 25: 
            #    v_unc = 0.23
            #    c_unc_sys = 0.42
            #    c_unc_int = 0.16
            #elif data_set[:,0].size < 100 and data_set[:,0].size >= 50: 
            #    v_unc = 0.18
            #    c_unc_sys = 0.34
            #    c_unc_int = 0.105
            #else: 
            #    v_unc = 0.15
            #    c_unc_sys = 0.29
            #    c_unc_int = 0.09

            #if pre_vdisp2 > 1.75*pre_vdisp: pre_vdisp_comb = 9.15*Ngal_1mpc+450.32
            #else:
            pre_vdisp_comb = pre_vdisp2
            
            #if data_set[:,1][np.where(data_set[:,0]<r200)].size >= 10:
            #    pre_vdisp_comb = biweight_midvariance(data_set[:,1][np.where(data_set[:,0]<r200)],9.0)
            #else:
            #    pre_vdisp_comb = np.std(data_set[:,1][np.where(data_set[:,0]<r200)],ddof=1)
            #    #pre_vdisp_comb = (pre_vdisp*(pre_vdisp2*v_unc)**2+\
            #                             pre_vdisp2*118.14**2)/(118.14**2+(pre_vdisp2*v_unc)**2)
            
        else:
            pre_vdisp_comb = cluster_data.clus_vdisp
        print('Combined Vdisp=',pre_vdisp_comb)

        #beta = 0.5*self.x_range/(self.x_range + r200/4.0)
        #Identify initial caustic surface and members within the surface
        print('Calculating initial surface')
        if self.inflection == False:
            if gal_memberflag is None:
                S.findsurface(data_set,self.x_range,self.y_range,img_tot,\
                                   r200=r200,halo_vdisp=pre_vdisp_comb,beta=None,\
                                   mirror=self.mirror,edge_perc=self.edge_perc,Hz=Hz,\
                                   edge_int_remove=self.edge_int_remove,q=self.kernal_stretch,plotphase=False)
            else:
                S.findsurface(data_set,self.x_range,self.y_range,img_tot,\
                                   memberflags=data_set[:,-1],r200=r200,\
                                   mirror=self.mirror,edge_perc=self.edge_perc,Hz=Hz,q=self.kernal_stretch)
        else:
            if gal_memberflag is None:
                S.findsurface_inf(data_set,self.x_range,self.y_range,img_tot,\
                                       self.img_inf,r200=r200,halo_vdisp=pre_vdisp_comb,\
                                       beta=None,Hz=Hz,q=self.kernal_stretch)
            else:
                S.findsurface_inf(data_set,self.x_range,self.y_range,img_tot,\
                                       self.img_inf,memberflags=data_set[:,-1],r200=r200,\
                                       Hz=Hz,q=self.kernal_stretch)

        results.clus_ra = cluster_data.clus_ra
        results.clus_dec = cluster_data.clus_dec
        results.clus_z = cluster_data.clus_z
        
        results.caustic_profile = S.Ar_finalD
        results.caustic_fit = S.vesc_fit
        results.caustic_edge = np.abs(S.Ar_finalE)
        results.caustic_fit_edge = S.vesc_fit_e
        results.gal_vdisp = S.gal_vdisp
        results.memflag = S.memflag

        #Estimate the mass based off the caustic profile, beta profile (if given), and concentration (if given)
        if clus_z is not None:
            crit = self.cosmo.fidcosmo.critical_density(clus_z).to(astunits.solMass/astunits.Mpc**3)#2.7745946e11*(self.cosmo.h)**2.0*(0.25*(1+clus_z)**3.0 + 0.75)
            Mass = self.calculate_mass( ri=self.x_range, A=results.caustic_profile,\
                                       crit=crit, r200=r200, fbr=None)
            Mass2 = self.calculate_mass( ri=self.x_range, A=results.caustic_profile,\
                                        crit=crit, r200=r200, fbr=self.fbr)
            MassE = self.calculate_mass( ri=self.x_range, A=results.caustic_edge,\
                                        crit=crit, r200=r200, fbr=self.fbr)
            MassF = self.calculate_mass( ri=self.x_range, A=results.caustic_fit,\
                                        crit=crit, r200=r200, fbr=self.fbr)
            MassFE = self.calculate_mass( ri=self.x_range, A=results.caustic_fit_edge,\
                                         crit=crit, r200=r200, fbr=self.fbr)
            
            results.crit = crit
            results.mprof = Mass.massprofile
            results.mprof_fbeta = Mass2.massprofile
            results.mprof_edge = MassE.massprofile
            results.r200_est = Mass.r200_est
            results.r200_est_fbeta = Mass2.r200_est
            results.r200_est_edge = MassE.r200_est
            results.r500_est = Mass.r500_est
            results.r500_est_fbeta = Mass2.r500_est
            results.M200_est = Mass.M200_est
            results.M200_est_fbeta = Mass2.M200_est
            results.M200_fbeta = Mass2.M200
            results.M200_edge = MassE.M200
            results.M200_edge_est = MassE.M200_est
            results.M200_fit = MassF.M200
            results.M200_fit_est = MassF.M200_est
            results.M200_fit_edge = MassFE.M200
            results.M200_fit_edge_est = MassFE.M200_est
            results.M500_est = Mass.M500_est
            results.M500_est_fbeta = Mass2.M500_est

            print('r200 estimate: ',Mass2.r200_est)
            print('M200 estimate: ',Mass2.M200_est)
            
            results.Ngal = data_set[np.where((results.memflag==1)&(data_set[:,0]<=results.r200_est_fbeta))].shape[0]
        
        #calculate velocity dispersion
        try:
            results.vdisp_gal = biweight_midvariance(data_set[:,1][results.memflag==1],9.0)
        except:
            try:
                results.vdisp_gal = np.std(data_set[:,1][results.memflag==1],ddof=1)
            except:
                results.vdisp_gal = 0.0
        return results

    def _gaussian_kernel(self,xvalues,yvalues,r200,normalization=100.0,scale=10.0,xres=200,yres=220,xmax=6.0,ymax=5000.0):
        """
        Uses a 2D gaussian kernel to estimate the density of the phase space.
        As of now, the maximum radius extends to 6Mpc and the maximum velocity allowed is 5000km/s
        The "q" parameter is termed "scale" here which we have set to 10 as default, but can go as high as 50.
        "normalization" is simply H0
        "x/yres" can be any value, but are recommended to be above 150
        "adj" is a custom value and changes the size of uniform filters when used (not normally needed)

        Parameters
        ----------
        xvalues : x-coordinates of points in phase space

        yvalues : y-coordinates of points in phase space

        r200 : Required estimate of r200 to calculate a rough dispersion

        normalization = 100 : This is equivalent to H0. Default is H0=100

        scale = 10 : "q" parameter in Diaferio 99. Literature says this can be between 10-50

        xres = 200 : x-grid resolution

        yres = 220 : y-grid resolution

        xmax = 6.0 : Maximum x-grid value. If data points exceed this amount either increase
                     this value or cut sample to be within this value.

        ymax = 5000 : Maximum/minimum y-grid value. If data points exceed this amount either increase
                     this value or cut sample to be within this value.

        Returns
        -------
        self.x_range : array of x-grid values
        self.y_range : array of y-grid values
        self.img : smoothed density image
        self.img_grad : first derivative of img
        self.img_inf : second derivative of img
        """
        if np.max(xvalues) >= xmax:
            raise Exception('Bounding Error: Please either increase your xmax value or trim your sample to be x < '+str(xmax))
        if np.max(np.abs(yvalues)) >= ymax:
            raise Exception('Bounding Error: Please either increase your ymax value or trim your sample to be y < '+str(ymax))

        yvalues = yvalues/(normalization*scale)

        self.x_range = np.arange(0,xmax,0.05)
        x_range_bin = np.arange(0,xmax+0.05,0.05)
        xres = self.x_range.size
        self.y_range = np.arange(-ymax/(normalization*scale),ymax/(normalization*scale),0.05)*normalization*scale
        y_range_bin = np.arange(-ymax/(normalization*scale),ymax/(normalization*scale)+0.05,0.05)*normalization*scale
        #yres = self.y_range.size
        x_scale = (xvalues/xmax)*xres
        y_scale = ((yvalues*(normalization*scale)+ymax)/(ymax*2.0))*self.y_range.size
        #self.ksize_x = (4.0/(3.0*xvalues.size))**(1/5.0)*np.std(x_scale[xvalues<r200])
        ksize_x =  (4.0/(3.0*xvalues.size))**(1/5.0)*\
                          np.sqrt((                                                                     \
                                biweight_midvariance((x_scale[xvalues<r200]).copy(),9.0)**2 +      \
                                biweight_midvariance((y_scale[xvalues<r200]).copy(),9.0)**2)/2.0   \
                          )
        ksize_x *= 1.0
        ksize_y = ksize_x#(4.0/(3.0*xvalues.size))**(1/5.0)*np.std(y_scale[xvalues<r200])
        imgr,xedge,yedge = np.histogram2d(xvalues,yvalues,bins=[x_range_bin,y_range_bin/(normalization*scale)])
        self.img = ndi.gaussian_filter(imgr, (ksize_x,ksize_y),mode='reflect')
        self.img_grad = ndi.gaussian_gradient_magnitude(imgr, (ksize_x,ksize_y))
        self.img_inf = ndi.gaussian_gradient_magnitude(ndi.gaussian_gradient_magnitude(imgr, (ksize_x,ksize_y)), (ksize_x,ksize_y))

    def calculate_mass(A,ri=np.arange(0,6.,0.05),beta=0.25,r200=2.0,crit=2.2e11,conc1=None,fbr=None):
        """
        calculate_mass(A,ri,r200,conc1=None,crit,beta=None,fbr=None)


        input:
            ri : rgrid values
               
            A : caustic profile values
            
            r200 = 2.0 : critical radius of cluster. Default is 2.0, but advised to take the output r200 and rerun
                        the analysis with this better estimate.

            conc1 = None : concentration of cluster. If None given then calculated from relationship
            
            crit = 2.2e11 : Critical density of the Universe. crit ~ 2.7745946e11*(self.cosmo.h)**2.0*(0.25*(1+clus_z)**3.0 + 0.75)

            beta = 0.2 : Anisotrpy parameter. Default value is 0.2, but a profile can be given that has same xvalues as ri.

            fbr = None : An exact guess of Fbeta by whatever means. Usually not used.
        
        returns:
            mass_info: MassInfo data object
                variables: g_b, conc, f_beta, massprofile, avg_density, r200_est, M200

        """
        "Calculate the mass profile"
        #vdisp = self.gal_vdisp
        G = astconsts.G.value
        solmass = astconsts.M_sun.value
        mass_info = MassInfo()
        r2 = ri[ri>=0]
        A2 = A[ri>=0]
        Mpc2km = astunits.Mpc.to(astunits.km)
        sumtot = np.zeros(A2.size)
        #print 'Using beta = %.2f'%(beta)
        if conc1 == None:
            #conc = 4.0*(vdisp/700.0)**(-0.306)
            conc = 5.0 + np.random.normal(0,2.0)
            if conc <= 0: conc = 5.0
        else:
            conc = conc1
        beta = 0.5*(ri/(ri+r200/conc))
        mass_info.g_b = (3-2.0*beta)/(1-beta)
        if fbr is None:
            f_beta = 0.5*((r2/r200*conc)**2)/((1+((r2/r200*conc)))**2*np.log(1+((r2/r200*conc))))*mass_info.g_b
            f_beta[0] = 0
            for i in range(A2.size-1):
                i += 1    
                sumtot[i] = np.trapz(f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*Mpc2km*1000)
                #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*Mpc2km*1000)
            #sum = integrate.cumtrapz(f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*Mpc2km*1000,initial=0.0)
        else:
            if type(fbr) == float or type(fbr) == int or type(fbr) == np.float64:
                f_beta = np.zeros(A2.size)+fbr*1.0
            else:
                f_beta = fbr
            f_beta[0] = 0
            for i in range(A2.size-1):
                i += 1    
                sumtot[i] = np.trapz(f_beta[1:i+1]*(A2[1:i+1]*1000)**2,(r2[1:i+1])*Mpc2km*1000)
                #sum[i] = np.trapz((A2[:i+1]*1000)**2,(r2[:i+1])*Mpc2km*1000)
            #sum = integrate.cumtrapz(f_beta*(A2[:f_beta.size]*1000)**2,r2[:f_beta.size]*Mpc2km*1000,initial=0.0)
        mass_info.massprofile = sumtot/(G*solmass)
        f_beta_size = f_beta.size
        #return the caustic r200
        mass_info.avg_density = mass_info.massprofile/(4.0/3.0*np.pi*(ri[:f_beta_size])**3.0)
        try:
            #mass_info.r200_est = (ri[:f_beta_size])[np.where(mass_info.avg_density >= 200*crit)[0]+1][-1]
            finterp = interp1d(mass_info.avg_density[::-1],ri[:f_beta_size][::-1])
            mass_info.r200_est = finterp(200*crit)
            mass_info.r500_est = finterp(500*crit)
        except IndexError:
            mass_info.r200_est = 0.0
            mass_info.r500_est = 0.0
        #mass_info.M200_est = mass_info.massprofile[np.where(ri[:f_beta_size] <= mass_info.r200_est)[0][-1]]
        finterp = interp1d(ri[:f_beta_size],mass_info.massprofile)
        mass_info.M200_est = finterp(mass_info.r200_est)
        mass_info.M500_est = finterp(mass_info.r500_est)
        mass_info.M200 = mass_info.massprofile[np.where(ri[:f_beta_size] <= r200)[0][-1]]
        mass_info.f_beta = f_beta
        mass_info.conc = conc
        return mass_info
        
    def findangle(ra,dec,clus_RA,clus_DEC):
        """
        Calculates the angles between the galaxies and the estimated cluster center.
        The value is returned in radians.
        """
        zsep = np.sin(np.deg2rad(clus_DEC))*np.sin(np.deg2rad(np.array(dec)))
        xysep = np.cos(np.deg2rad(clus_DEC))*np.cos(np.deg2rad(np.array(dec)))*np.cos(np.deg2rad((clus_RA-np.array(ra))))
        angle = np.arccos(zsep+xysep)
        return angle

    def set_sample(data,rlimit=4.0,vlimit=3500):
        """
        Reduces the sample by selecting only galaxies inside r and v limits.
        The default is to use a vlimit = 3500km/s and rlimit = 4.0Mpc.
        Specify in parameter file.
        """
        data_set = data[np.where((data[:,0] < rlimit) & (np.abs(data[:,1]) < vlimit))]
        return data_set

    def shiftgapper(data):
        npbin = 25
        #gap_prev = 2000.0 #initialize gap size for initial comparison (must be larger to start).
        nbins = np.int(np.ceil(data[:,0].size/(npbin*1.0)))
        #origsize = data[:,0].shape[0]
        data = data[np.argsort(data[:,0])] #sort by r to ready for binning
        #print 'NBINS FOR GAPPER = ', nbins
        for i in range(nbins):
            #print 'BEGINNING BIN:',str(i)
            databin = data[npbin*i:npbin*(i+1)]
            datanew = None
            nsize = databin[:,0].size
            datasize = nsize-1
            datafinal = np.array([])
            if nsize > 5:
                while nsize - datasize > 0 and datasize >= 5:
                    #print '    ITERATING'
                    nsize = databin[:,0].size
                    databinsort = databin[np.argsort(databin[:,1])] #sort by v
                    f = (databinsort[:,1])[databinsort[:,1].size-np.int(np.ceil(databinsort[:,1].size/4.0))]-(databinsort[:,1])[np.int(np.ceil(databinsort[:,1].size/4.0))]
                    gap = f/(1.349)
                    #print i,'    GAP SIZE', str(gap)
                    if gap < 500.0: break
                        #gap = 500.0
                    #if gap >= 2.0*gap_prev: 
                    #    gap = gap_prev
                    #    #print '   Altered gap = %.3f'%(gap)
                    databelow = databinsort[databinsort[:,1]<=0]
                    gapbelow =databelow[:,1][1:]-databelow[:,1][:-1]
                    dataabove = databinsort[databinsort[:,1]>0]
                    gapabove = dataabove[:,1][1:]-dataabove[:,1][:-1]
                    try:
                        if np.max(gapbelow) >= gap: vgapbelow = np.where(gapbelow >= gap)[0][-1]
                        else: vgapbelow = -1
                        #print 'MAX BELOW GAP',np.max(gapbelow)
                        try: 
                            datanew = np.append(datanew,databelow[vgapbelow+1:],axis=0)
                        except:
                            datanew = databelow[vgapbelow+1:]
                    except ValueError:
                        pass
                    try:
                        if np.max(gapabove) >= gap: vgapabove = np.where(gapabove >= gap)[0][0]
                        else: vgapabove = 99999999
                        #print 'MAX ABOVE GAP',np.max(gapabove)
                        try: 
                            datanew = np.append(datanew,dataabove[:vgapabove+1],axis=0)
                        except:
                            datanew = dataabove[:vgapabove+1]
                    except ValueError:
                        pass
                    databin = datanew
                    datasize = datanew[:,0].size
                    datanew = None
                #print 'DATA SIZE OUT', databin[:,0].size
                #if gap >=500.0:
                #    gap_prev = gap
                #else:
                #    gap_prev = 500.0
                
            try:
                datafinal = np.append(datafinal,databin,axis=0)
            except:
                datafinal = databin
        #print 'GALAXIES CUT =',str(origsize-datafinal[:,0].size)
        return datafinal