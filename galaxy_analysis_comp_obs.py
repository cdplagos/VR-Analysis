#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import collections
import functools
import math

import numpy as np

import common
import utilities_statistics as us

##################################
# Constants
mlow = 6.0
mupp = 15.0
dm = 0.2
mbins = np.arange(mlow, mupp, dm)
xmf = mbins + dm/2.0
h0 = 0.67

vlow = 1
vupp = 3.2
dv = 0.2
vbins = np.arange(vlow, vupp, dv)
xvf = vbins + dv/2.0

zsun = 0.0127
log10zsun = np.log10(zsun)

observation = collections.namedtuple('observation', 'label x y yerrup yerrdn err_absolute')

def add_observations_to_plot(obsdir, fname, ax, marker, label, color='k', err_absolute=False):
    fname = '%s/Gas/%s' % (obsdir, fname)
    x, y, yerr_down, yerr_up = common.load_observation(obsdir, fname, (0, 1, 2, 3))
    common.errorbars(ax, x, y, yerr_down, yerr_up, color, marker, label=label, err_absolute=err_absolute)

def prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit):
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit)
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)
    #ax.text(xleg, yleg, 'z=0')

def plot_mf_z(plt, outdir, obsdir, snap, vol_eagle,  histmtot, histm30):
    
    #construct relevant observational datasets for SMF

    z0obs = []
    lm, p, dpdn, dpup = common.load_observation(obsdir, 'mf/SMF/GAMAII_BBD_GSMFs.dat', [0,1,2,3])
    xobs = lm
    indx = np.where(p > 0)
    yobs = np.log10(p[indx])
    ydn = yobs - np.log10(p[indx]-dpdn[indx])
    yup = np.log10(p[indx]+dpup[indx]) - yobs
    z0obs.append((observation("Wright+2017", xobs[indx], yobs, ydn, yup, err_absolute=False), 'o'))
       
    # Moustakas (Chabrier IMF), ['Moustakas+2013, several redshifts']
    zdnM13, lmM13, pM13, dp_dn_M13, dp_up_M13 = common.load_observation(obsdir, 'mf/SMF/SMF_Moustakas2013.dat', [0,3,5,6,7])
    xobsM13 = lmM13 

    yobsM13 = np.full(xobsM13.shape, -999.)
    lerrM13 = np.full(xobsM13.shape, -999.)
    herrM13 = np.full(xobsM13.shape, 999.)
    indx = np.where( pM13 < 1)
    yobsM13[indx] = (pM13[indx])
    indx = np.where( dp_dn_M13 > 0)
    lerrM13[indx]  = dp_dn_M13[indx] 
    indx = np.where( dp_up_M13 > 0)
    herrM13[indx]  = dp_up_M13[indx]

    # Muzzin (Kroupa IMF), ['Moustakas+2013, several redshifts']
    zdnMu13,zupMu13,lmMu13,pMu13,dp_dn_Mu13,dp_up_Mu13 = common.load_observation(obsdir, 'mf/SMF/SMF_Muzzin2013.dat', [0,1,2,4,5,5])
    # -0.09 corresponds to the IMF correction
    xobsMu13 = lmMu13 - 0.09
    yobsMu13 = np.full(xobsMu13.shape, -999.)
    lerrMu13 = np.full(xobsMu13.shape, -999.)
    herrMu13 = np.full(xobsMu13.shape, 999.)
    indx = np.where( pMu13 < 1)
    yobsMu13[indx] = (pMu13[indx])
    indx = np.where( dp_dn_Mu13 > 0)
    lerrMu13[indx]  = dp_dn_Mu13[indx] 
    indx = np.where( dp_up_Mu13 > 0)
    herrMu13[indx]  = dp_up_Mu13[indx]

    # Santini 2012 (Salpeter IMF)
    zdnS12, lmS12, pS12, dp_dn_S12, dp_up_S12 = common.load_observation(obsdir, 'mf/SMF/SMF_Santini2012.dat', [0,2,3,4,5])
    hobs = 0.7
    # factor 0.24 corresponds to the IMF correction.
    xobsS12 = lmS12 - 0.24 +  np.log10(hobs/h0)
    yobsS12 = np.full(xobsS12.shape, -999.)
    lerrS12 = np.full(xobsS12.shape, -999.)
    herrS12 = np.full(xobsS12.shape, 999.)
    indx = np.where( pS12 < 1)
    yobsS12[indx] = (pS12[indx]) + np.log10(pow(h0/hobs,3.0))
    indx = np.where( dp_dn_S12 > 0)
    lerrS12[indx]  = dp_dn_S12[indx]
    indx = np.where( dp_up_S12 > 0)
    herrS12[indx]  = dp_up_S12[indx]

    # Wright et al. (2018, several reshifts). Assumes Chabrier IMF.
    zD17, lmD17, pD17, dp_dn_D17, dp_up_D17 = common.load_observation(obsdir, 'mf/SMF/Wright18_CombinedSMF.dat', [0,1,2,3,4])
    hobs = 0.7
    pD17 = pD17 - 3.0*np.log10(hobs) 
    lmD17= lmD17 - np.log10(hobs)

    # z0.5 obs
    z05obs = []
    in_redshift = np.where(zdnM13 == 0.4)
    z05obs.append((observation("Moustakas+2013", xobsM13[in_redshift], yobsM13[in_redshift], lerrM13[in_redshift], herrM13[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zdnMu13 == 0.5)
    z05obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), '+'))
    in_redshift = np.where(zD17 == 0.5)
    z05obs.append((observation("Wright+2018", lmD17[in_redshift], pD17[in_redshift], dp_dn_D17[in_redshift], dp_up_D17[in_redshift], err_absolute=False), 'D'))

    # z1 obs
    z1obs = []
    in_redshift = np.where(zdnM13 == 0.8)
    z1obs.append((observation("Moustakas+2013", xobsM13[in_redshift], yobsM13[in_redshift], lerrM13[in_redshift], herrM13[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zdnMu13 == 1)
    z1obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), '+'))
    in_redshift = np.where(zD17 == 1)
    z1obs.append((observation("Wright+2018", lmD17[in_redshift], pD17[in_redshift], dp_dn_D17[in_redshift], dp_up_D17[in_redshift], err_absolute=False), 'D'))

    #z2 obs
    z2obs = []
    in_redshift = np.where(zupMu13 == 2.5)
    z2obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), '+'))
    in_redshift = np.where(zdnS12 == 1.8)
    z2obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zD17 == 2)
    z2obs.append((observation("Wright+2018", lmD17[in_redshift], pD17[in_redshift], dp_dn_D17[in_redshift], dp_up_D17[in_redshift], err_absolute=False), 'D'))

    # z3 obs
    z3obs = []
    in_redshift = np.where(zupMu13 == 3.0)
    z3obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), '+'))
    in_redshift = np.where(zdnS12 == 2.5)
    z3obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zD17 == 3)
    z3obs.append((observation("Wright+2018", lmD17[in_redshift], pD17[in_redshift], dp_dn_D17[in_redshift], dp_up_D17[in_redshift], err_absolute=False), 'D'))

    # z4 obs
    z4obs = []
    in_redshift = np.where(zupMu13 == 4.0)
    z4obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), '+'))
    in_redshift = np.where(zdnS12 == 3.5)
    z4obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zD17 == 4)
    z4obs.append((observation("Wright+2018", lmD17[in_redshift], pD17[in_redshift], dp_dn_D17[in_redshift], dp_up_D17[in_redshift], err_absolute=False), 'D'))

    ########################### total stellar mass function
    xtit="$\\rm log_{10} (\\rm M_{\\star,\\rm tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\star}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 7, 12, -6, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]
    observations = (z0obs, z1obs, z2obs)

    for subplot, idx, z, s, obs_and_markers in zip(subplots, idx, zins, snap, observations):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          # Observations
          for obs, marker in obs_and_markers:
              common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                               marker, err_absolute=obs.err_absolute, label=obs.label)
  
          #Predicted HMF
          y = histmtot[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='solid', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='solid')
          if idx == 0:
              cols = ['r'] + ['grey', 'grey','grey']
              common.prepare_legend(ax, cols)

    common.savefig(outdir, fig, "smf_tot_z_comp_obs.pdf")

    ############################# stellar mass function (30kpc aperture)
    xtit="$\\rm log_{10} (\\rm M_{\\star,\\rm 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\star}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 7, 12, -6, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s, obs_and_markers in zip(subplots, idx, zins, snap, observations):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          # Observations
          for obs, marker in obs_and_markers:
              common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                               marker, err_absolute=obs.err_absolute, label=obs.label)

          #Predicted HMF
          y = histm30[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='solid', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='solid')
          if idx == 0:
              cols = ['r'] + ['grey', 'grey','grey']
              common.prepare_legend(ax, cols)
          else:
              cols =  ['grey', 'grey','grey']
              common.prepare_legend(ax, cols)

    common.savefig(outdir, fig, "smf_30kpc_z_comp_obs.pdf")


def plot_scaling_z(plt, outdir, obsdir, snap, SFRMstar, R50Mstar, R50Mstar30, MBHMstar, SigmaMstar30, ZstarMstar, 
                   ZSFMstar, AgeSMstar, SFRMstar30, R50pMstar30):

    #define observaiton of the MS at z=0 to be plotted
    def obs_mainseq_z0():
        #best fit from Davies et al. (2016)
        xdataD16 = [9.3, 10.6]
        ydataD16 = [-0.39, 0.477]
        ax.plot(xdataD16,ydataD16, color='b',linestyle='dashdot',linewidth = 4, label='Davies+16')
        #SDSS z=0 relation
        lm, SFR = common.load_observation(obsdir, 'SFR/Brinchmann04.dat', (0, 1))
        hobs = 0.7
        #add cosmology correction plus IMF correction that goes into the stellar mass.
        corr_cos = np.log10(pow(hobs,2)/pow(h0,2)) - 0.09
        # apply correction to both stellar mass and SFRs.
        ax.plot(lm[0:35] + corr_cos, SFR[0:35] + corr_cos, color='PaleVioletRed', linewidth = 3, linestyle='dashed', label='Brinchmann+04')
        ax.plot(lm[36:70] + corr_cos, SFR[36:70] + corr_cos, color='PaleVioletRed',linewidth = 5, linestyle='dotted')
        ax.plot(lm[71:len(SFR)] + corr_cos, SFR[71:len(SFR)] + corr_cos, color='PaleVioletRed',linewidth = 5, linestyle='dotted')


    ########################### will plot main sequence for all stellar particles in the subhalo
    xtit="$\\rm log_{10} (\\rm M_{\\star, tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm SFR/M_{\odot} yr^{-1})$"
    xmin, xmax, ymin, ymax = 7, 12, -5, 1.5
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
 
          #observations z=0
          if (z == 0):
              obs_mainseq_z0()

          #VR
          ind = np.where(SFRMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = SFRMstar[idx,0,ind]
          errdn = SFRMstar[idx,1,ind]
          errup = SFRMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','PaleVioletRed', 'k'])

    common.savefig(outdir, fig, "main_sequence_z_comp_obs.pdf")

    ########################### will plot main sequence for 30kpc aperture
    xtit="$\\rm log_{10} (\\rm M_{\\star, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm SFR_{\\rm 30kpc}/M_{\odot} yr^{-1})$"
    xmin, xmax, ymin, ymax = 7, 12, -5, 1.5
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]


    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #observations z=0
          if (z == 0):
              obs_mainseq_z0()

          #VR
          ind = np.where(SFRMstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = SFRMstar30[idx,0,ind]
          errdn = SFRMstar30[idx,1,ind]
          errup = SFRMstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','PaleVioletRed', 'k'])

    common.savefig(outdir, fig, "main_sequence_30kpc_z_comp_obs.pdf")



    ########################### will plot r50 vs stellar mass for all stellar particles in the subhalo
    #define observations first
    def plot_gama_size_mass(): 
        m,r = common.load_observation(obsdir, 'SizesAndAM/rdisk_L16.dat', [0,1])
        ax.plot(m[0:36], r[0:36], linestyle='dotted',color='b', label='L16 disks')
        ax.plot(m[38:83], r[38:83], linestyle='dotted',color='b')
        ax.plot(m[85:128], r[85:129], linestyle='dotted',color='b')
        m,r = common.load_observation(obsdir, 'SizesAndAM/rbulge_L16.dat', [0,1])
        ax.plot(m[0:39], r[0:39], linestyle='dotted',color='darkgreen', label='L16 bulges')
        ax.plot(m[41:76], r[41:76], linestyle='dotted',color='darkgreen')
        ax.plot(m[78:115], r[78:115], linestyle='dotted',color='darkgreen')


    xtit="$\\rm log_{10} (\\rm M_{\\star,tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 50,tot}/pkpc)$"
    xmin, xmax, ymin, ymax = 7, 12, -0.3, 2
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)

          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          plt.subplots_adjust(left=0.2)

          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          if ( z == 0):
              plot_gama_size_mass()
         
          #VR
          ind = np.where(R50Mstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = R50Mstar[idx,0,ind] + 3.0
          errdn = R50Mstar[idx,1,ind]
          errup = R50Mstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','darkgreen','k'], loc = 'upper left')

    common.savefig(outdir, fig, "r50_Mstar_z_comp_obs.pdf")

    ################## will plot r50 vs stellar mass for quantities measured within 30kpc
    xtit="$\\rm log_{10} (\\rm M_{\\star,30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 50,30kpc}/pkpc)$"
    xmin, xmax, ymin, ymax = 7, 12, -0.3, 2
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)

          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          plt.subplots_adjust(left=0.2)

          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          if ( z == 0):
              plot_gama_size_mass()
 
          #VR
          ind = np.where(R50Mstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = R50Mstar30[idx,0,ind] + 3.0
          errdn = R50Mstar30[idx,1,ind]
          errup = R50Mstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','darkgreen','k'], loc = 'upper left')

    common.savefig(outdir, fig, "r50_Mstar_30kpc_z_comp_obs.pdf")

    ################## will plot r50 vs stellar mass for quantities measured within 30kpc, but in this case the r50 is projected
    xtit="$\\rm log_{10} (\\rm M_{\\star,30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 50,30kpc,2D}/pkpc)$"
    xmin, xmax, ymin, ymax = 7, 12, -0.3, 2
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)

          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          plt.subplots_adjust(left=0.2)

          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          if ( z == 0):
              plot_gama_size_mass()

          #VR
          ind = np.where(R50pMstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = R50pMstar30[idx,0,ind] + 3.0
          errdn = R50pMstar30[idx,1,ind]
          errup = R50pMstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','darkgreen','k'], loc = 'upper left')

    common.savefig(outdir, fig, "r50_projected_Mstar_30kpc_z_comp_obs.pdf")


    ########################### will plot stellar velocity dispersion vs. stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star,30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\sigma_{\\star,30kpc}/km s^{-1})$"
    xmin, xmax, ymin, ymax = 7, 12, 1, 3
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)

          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          plt.subplots_adjust(left=0.2)

          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #VR
          ind = np.where(SigmaMstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = SigmaMstar30[idx,0,ind]
          errdn = SigmaMstar30[idx,1,ind]
          errup = SigmaMstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          #observations
          if ( z == 0):
               lm, sig, sigdn, sigup = common.load_observation(obsdir, 'StellarPops/vdS19-sigma.csv', [0,1,2,3])
               sig   = np.log10(sig)
               sigdn = np.log10(sigdn)
               sigup = np.log10(sigup)
               common.errorbars(ax, lm, sig, sigdn, sigup, 'b', 'D', label='van de Sande+19')

          if idx == 0:
              common.prepare_legend(ax, ['k'], loc='upper left')

    common.savefig(outdir, fig, "vdisp_Mstar_30kpc_z_comp_obs.pdf")

    ############ will plot stellar metallicity-stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm Z_{\star}/Z_{\\odot})$"
    xmin, xmax, ymin, ymax = 7, 12, -2, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]


    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          plt.subplots_adjust(left=0.2)
 
          #VR
          ind = np.where(ZstarMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = ZstarMstar[idx,0,ind] - log10zsun
          errdn = ZstarMstar[idx,1,ind]
          errup = ZstarMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          #observations
          if ( z == 0):
               lm, mz, mzdn, mzup = common.load_observation(obsdir, 'MZR/MSZR-Gallazzi05.dat', [0,1,2,3])
               common.errorbars(ax, lm[0:7], mz[0:7], mzdn[0:7], mzup[0:7], 'b', 'D', label='Kirby+13')
               common.errorbars(ax, lm[7:22], mz[7:22], mzdn[7:22], mzup[7:22], 'b', 'o', label='Gallazzi+05')

          if idx == 0:
              common.prepare_legend(ax, ['k', 'b', 'b'], loc = 'lower right')

    common.savefig(outdir, fig, "zstar_mstar_30kpc_z_comp_obs.pdf")

    ################ will plot star-forming gas metallicity vs. stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm Z_{\\rm SF,gas}/Z_{\\odot})$"
    xmin, xmax, ymin, ymax = 7, 12, -2, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          plt.subplots_adjust(left=0.2)
          
          if( z == 0):  
              #MZR z=0
              corrzsun = 8.69 #solar oxygen abundance in units of 12 + log(O/H)
              hobs = 0.72
              #add cosmology correction plus IMF correction that goes into the stellar mass.
              corr_cos = np.log10(pow(hobs,2)/pow(h0,2)) - 0.09
              lm, mz, mzdn, mzup = common.load_observation(obsdir, 'MZR/MMAdrews13.dat', [0,1,2,3])
              hobs = 0.7
              #add cosmology correction plus IMF correction that goes into the stellar mass.
              corr_cos = np.log10(pow(hobs,2)/pow(h0,2)) - 0.09
              common.errorbars(ax, lm+ corr_cos, mz - corrzsun, mzdn - corrzsun, mzup - corrzsun, 'b', 's', label='Andrews+13')
              #correction for Tremonti is the same.
              lm, mz, mzdn, mzup = common.load_observation(obsdir, 'MZR/Tremonti04.dat', [0,1,2,3])
              common.errorbars(ax, lm+ corr_cos, mz - corrzsun, mzdn - corrzsun, mzup - corrzsun, 'b', 'o', label="Tremonti+04")

          #VR
          ind = np.where(ZSFMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = ZSFMstar[idx,0,ind] - log10zsun
          errdn = ZSFMstar[idx,1,ind]
          errup = ZSFMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['k','b','b'], loc = 'lower right')

    common.savefig(outdir, fig, "zsfgas_mstar_30kpc_z_comp_obs.pdf")

    ################ will plot stellar ages vs stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm age_{\\star}/Gyr)$"
    xmin, xmax, ymin, ymax = 7, 12, 0, 1.1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          plt.subplots_adjust(left=0.2)
  
          #VR
          ind = np.where(AgeSMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = AgeSMstar[idx,0,ind]
          errdn = AgeSMstar[idx,1,ind]
          errup = AgeSMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          #observations
          if ( z == 0):
               lm, age, agedn, ageup = common.load_observation(obsdir, 'StellarPops/vdS19-age.csv', [0,1,2,3])
               common.errorbars(ax, lm, age, agedn, ageup, 'b', 'D', label='van de Sande+19')

          if idx == 0:
              common.prepare_legend(ax, ['k'])

    common.savefig(outdir, fig, "starage_mstar_z_comp_obs.pdf")

    ################ will plot black hole mass vs stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm M_{\\rm BH}/M_{\odot})$"
    xmin, xmax, ymin, ymax = 7, 12, 6, 11
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
          plt.subplots_adjust(left=0.2)

          if (z == 0):
             #BH-bulge relation
             mBH_M13, errup_M13, errdn_M13, mBH_power, mbulge_M13 = common.load_observation(obsdir, 'BHs/MBH_sigma_Mbulge_McConnelMa2013.dat', [0,1,2,3,7])
         
             ind = np.where((mBH_M13 > 0) & (mbulge_M13 > 0))
             xobs = np.log10(mbulge_M13[ind])
             yobs = np.log10(mBH_M13[ind] * pow(10.0,mBH_power[ind]))
             lerr = np.log10((mBH_M13[ind] - errdn_M13[ind]) * pow(10.0,mBH_power[ind]))
             herr = np.log10((mBH_M13[ind] + errup_M13[ind]) * pow(10.0,mBH_power[ind]))
             ax.errorbar(xobs, yobs, yerr=[yobs-lerr,herr-yobs], ls='None', mfc='None', ecolor = 'r', mec='r',marker='^',label="McConnell & Ma 2013")
         
             #BH-bulge relation
             mBH_H04, errup_H04, errdn_H04, mbulge_H04 = common.load_observation(obsdir, 'BHs/MBH_sigma_Mbulge_HaeringRix2004.dat', [0,1,2,4])
         
             xobs = np.log10(mbulge_H04)
         
             yobs = xobs*0. - 999.
             indx = np.where( mBH_H04 > 0)
             yobs[indx] = np.log10(mBH_H04[indx])
         
             lerr = yobs*0. - 999.
             indx = np.where( (mBH_H04-errdn_H04) > 0)
             lerr[indx]  = np.log10(mBH_H04[indx] - errdn_H04[indx])
         
             herr = yobs*0. + 999.
             indx = np.where( (mBH_H04+errup_H04) > 0)
             herr[indx]  = np.log10(mBH_H04[indx] + errup_H04[indx])
             ax.errorbar(xobs, yobs, yerr=[yobs-lerr,herr-yobs], ls='None', mfc='None', ecolor = 'maroon', mec='maroon',marker='s',label="Haering+04")

          #VR
          ind = np.where(MBHMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = MBHMstar[idx,0,ind]
          errdn = MBHMstar[idx,1,ind]
          errup = MBHMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'k', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'k')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='k', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='k', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['k','r','maroon'], loc='upper left')

    common.savefig(outdir, fig, "blackhole_stellarmass_z_comp_obs.pdf")

def prepare_data(hdf5_data, j, histmtot, histm30, SFRMstar, R50Mstar, SigmaMstar,  
                 MBHMstar, R50Mstar30, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar, 
                 SFRMstar30, R50pMstar30):

    # Unpack data
    (ms30, mstot, mg30, r50stot, sfrtot, mtot, mbh, r50s30, vs30, zgas, zgas_sf, zstar, 
     age_star, sfr30, r50s30p1, r50s30p2, r50s30p3, zgas_nsf) = hdf5_data

    bin_it = functools.partial(us.wmedians, xbins=xmf)

    #take the average of the three projections provided by VR
    r50s30p = (r50s30p1 + r50s30p2 + r50s30p3)/3.0

    #mass functions
    ind = np.where(mstot > 0)
    H, bins_edges = np.histogram(np.log10(mstot[ind]) + 10.0,bins=np.append(mbins,mupp))
    histmtot[j,:] = histmtot[j,:] + H
    ind = np.where(ms30 > 0)
    H, bins_edges = np.histogram(np.log10(ms30[ind]) + 10.0,bins=np.append(mbins,mupp))
    histm30[j,:] = histm30[j,:] + H

    #main sequences
    ind = np.where((mstot > 0) & (sfrtot > 0))
    SFRMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(sfrtot[ind]))
    ind = np.where((ms30 > 0) & (sfr30 > 0))
    SFRMstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(sfr30[ind]))

    #BH-stellar mass relation
    ind = np.where((mstot > 0) & (mbh > 0))
    MBHMstar[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(mbh[ind]) + 10.0)

    #size-mass relations
    ind = np.where((mstot > 0) & (r50stot > 0))
    R50Mstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(r50stot[ind]))
    ind = np.where((ms30 > 0) & (r50s30 > 0))
    R50Mstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(r50s30[ind]))
    ind = np.where((ms30 > 0) & (r50s30p > 0))
    R50pMstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(r50s30p[ind]))

    #velocity dispersion-mass relation
    ind = np.where((ms30 > 0) & (vs30 > 0))
    SigmaMstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(vs30[ind]/1.73205080757))

    #age-mass relation
    ind = np.where((mstot > 0) & (age_star > 0))
    AgeSMstar[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(age_star[ind]/1e9))
 
    #mass-metallicity relations
    ind = np.where((mstot > 0) & (zstar > 0))
    ZstarMstar[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(zstar[ind]))
    ind = np.where((mstot > 0) & (zgas_sf > 0))
    ZSFMstar[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(zgas_sf[ind]))

def main():

    model_dir = '/fred/oz009/clagos/vr-testing-outputs/hydro/'
    output_dir = '/fred/oz009/clagos/vr-testing-outputs/Plots/hydro/'
    obsdir = 'data/'

    subvols = [0,1,2]
    snap = [28, 19, 15]
    vol = 25.0**3.0 #Mpc
    vol_eagle = 25.0**3.0

    name_file = 'fourth-try-6dfofsubhalos'#:'third-try'

    plt = common.load_matplotlib()
    fields = ['Aperture_mass_star_30_kpc','M_star','Aperture_mass_gas_30_kpc','R_HalfMass_star','SFR_gas','Aperture_mass_30_kpc','M_bh',
              'Aperture_rhalfmass_star_30_kpc','Aperture_veldisp_star_30_kpc', 'Zmet_gas', 'Zmet_gas_sf','Zmet_star','tage_star', 
              'Aperture_SFR_gas_30_kpc','Projected_aperture_1_rhalfmass_star_30_kpc','Projected_aperture_2_rhalfmass_star_30_kpc',
              'Projected_aperture_3_rhalfmass_star_30_kpc', 'Zmet_gas_nsf']

    # Create histogram for mass functions
    histmtot = np.zeros(shape = (len(snap),len(mbins)))
    histm30 = np.zeros(shape = (len(snap),len(mbins)))

    # create matrices for several scaling relations
    SFRMstar     = np.zeros(shape = (len(snap),3,len(mbins)))
    SFRMstar30   = np.zeros(shape = (len(snap),3,len(mbins)))
    R50Mstar     = np.zeros(shape = (len(snap),3,len(mbins)))
    SigmaMstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    MBHMstar     = np.zeros(shape = (len(snap),3,len(mbins)))
    R50Mstar30   = np.zeros(shape = (len(snap),3,len(mbins)))
    R50pMstar30  = np.zeros(shape = (len(snap),3,len(mbins)))
    SigmaMstar30 = np.zeros(shape = (len(snap),3,len(mbins)))
    ZstarMstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    AgeSMstar    = np.zeros(shape = (len(snap),3,len(mbins)))
    ZSFMstar     = np.zeros(shape = (len(snap),3,len(mbins)))

    for j in range(0,len(snap)):
        hdf5_data = common.read_data(model_dir, fields, snap[j], subvols, name_file)
        prepare_data(hdf5_data, j, histmtot, histm30, SFRMstar, R50Mstar, SigmaMstar, MBHMstar, R50Mstar30, SigmaMstar30, 
                     ZstarMstar, ZSFMstar, AgeSMstar, SFRMstar30, R50pMstar30)

    # Take logs for mass functions
    ind = np.where(histmtot > 0.)
    histmtot[ind] = np.log10(histmtot[ind]/vol/dm)

    ind = np.where(histm30 > 0.)
    histm30[ind] = np.log10(histm30[ind]/vol/dm)

    #mass funcion plots
    plot_mf_z(plt, output_dir+name_file+'/', obsdir, snap, vol_eagle, histmtot, histm30)
    #scaling relation plots
    plot_scaling_z(plt, output_dir+name_file+'/', obsdir, snap, SFRMstar, R50Mstar, R50Mstar30, MBHMstar, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar, 
                  SFRMstar30, R50pMstar30)

if __name__ == '__main__':
    main()

