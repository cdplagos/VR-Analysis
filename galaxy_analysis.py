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

import functools

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

def add_observations_to_plot(obsdir, fname, ax, marker, label, color='k', err_absolute=False):
    fname = '%s/Gas/%s' % (obsdir, fname)
    x, y, yerr_down, yerr_up = common.load_observation(obsdir, fname, (0, 1, 2, 3))
    common.errorbars(ax, x, y, yerr_down, yerr_up, color, marker, label=label, err_absolute=err_absolute)

def prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit):
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit)
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)
    #ax.text(xleg, yleg, 'z=0')

def plot_mf_z(plt, outdir, snap, vol_eagle,  histmtot, histm30, histmgas, histmall):

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

    sn, subgn, mstot, ms30,  mgas30, mdm30, mbh30 = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 1, 29, 22, 23, 31, 32])
    mall = ms30+mgas30+mdm30+mbh30

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #SMF from SUBFIND
          ind = np.where((mstot > 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(mstot[ind]),bins=np.append(mbins,mupp))
          histsfof = H

          y = histsfof[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b')
  
          #Predicted HMF
          y = histmtot[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed')
          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "smf_tot_z.pdf")

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

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #SMF from SUBFIND
          ind = np.where((ms30 > 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(ms30[ind]),bins=np.append(mbins,mupp))
          histsfof = H

          y = histsfof[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b')
  
          #Predicted HMF
          y = histm30[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed')
          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "smf_30kpc_z.pdf")

    ############################# gas mass function (30kpc aperture)
    xtit="$\\rm log_{10} (\\rm M_{\\rm gas, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\rm gas}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 7, 12, -6, 1
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
  
          #SMF from SUBFIND
          ind = np.where((mgas30 > 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(mgas30[ind]),bins=np.append(mbins,mupp))
          histsfof = H

          y = histsfof[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b')
  
          #Predicted HMF
          y = histmgas[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed')
          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "gasmf_30kpc_z.pdf")

    ##################################### total mass function (30kpc aperture)
    xtit="$\\rm log_{10} (\\rm M_{\\rm tot, 30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\rm ror}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 7, 14, -6, 1
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
  
          #SMF from SUBFIND
          ind = np.where((mall > 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(mall[ind]),bins=np.append(mbins,mupp))
          histsfof = H

          y = histsfof[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'b')
  
          #Predicted HMF
          y = histmall[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed', label ='VR')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed')
          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "allmassmf_30kpc_z.pdf")


def plot_scaling_z(plt, outdir, snap, SFRMstar, R50Mstar, R50Mstar30, MBHMstar, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar):

    bin_it = functools.partial(us.wmedians, xbins=xmf)

    ########################### will plot main sequence for all stellar particles in the subhalo
    xtit="$\\rm log_{10} (\\rm M_{\\star, tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm SFR/M_{\odot} yr^{-1})$"
    xmin, xmax, ymin, ymax = 7, 12, -5, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, mstot, sfr, r50, ms30 = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 29, 30, 26, 22])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (sfr > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(sfr[ind]))
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(SFRMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = SFRMstar[idx,0,ind]
          errdn = SFRMstar[idx,1,ind]
          errup = SFRMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "main_sequence_z.pdf")

    ########################### will plot r50 vs stellar mass for all stellar particles in the subhalo
    xtit="$\\rm log_{10} (\\rm M_{\\star,tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 50}/pMpc)$"
    xmin, xmax, ymin, ymax = 7, 12, -3.3, -1
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
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (r50 > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(r50[ind])-3.0)
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(R50Mstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = R50Mstar[idx,0,ind]
          errdn = R50Mstar[idx,1,ind]
          errup = R50Mstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "r50_Mstar_z.pdf")

    ################## will plot r50 vs stellar mass for quantities measured within 30kpc
    sn, mstot, sfr, r50 = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE-REF.data', [2, 22, 30, 33])
    xtit="$\\rm log_{10} (\\rm M_{\\star,30kpc}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 50,30kpc}/pMpc)$"
    xmin, xmax, ymin, ymax = 7, 12, -3.3, -1
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
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (r50 > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(r50[ind])-3.0)
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(R50Mstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = R50Mstar30[idx,0,ind]
          errdn = R50Mstar30[idx,1,ind]
          errup = R50Mstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "r50_Mstar_30kpc_z.pdf")


    ########################### will plot stellar velocity dispersion vs. stellar mass
    sn, mstot, sfr, r50, vs = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE-REF.data', [2, 22, 30, 33, 25])
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
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (vs > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(vs[ind]))
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(SigmaMstar30[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = SigmaMstar30[idx,0,ind]
          errdn = SigmaMstar30[idx,1,ind]
          errup = SigmaMstar30[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "vdisp_Mstar_30kpc_z.pdf")

    ############ will plot stellar metallicity-stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm Z_{\star})$"
    xmin, xmax, ymin, ymax = 7, 12, -5, -1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, mstot, zsf, zs, age = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 29, 33, 34, 35])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (zs > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(zs[ind]))
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(ZstarMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = ZstarMstar[idx,0,ind]
          errdn = ZstarMstar[idx,1,ind]
          errup = ZstarMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "zstar_mstar_z.pdf")

    ################ will plot gas metallicity vs. stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, tot}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm Z_{\\rm gas})$"
    xmin, xmax, ymin, ymax = 7, 12, -5, -1
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
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (zsf > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(zsf[ind]))
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(ZSFMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = ZSFMstar[idx,0,ind]
          errdn = ZSFMstar[idx,1,ind]
          errup = ZSFMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "zsfgas_mstar_z.pdf")

    ################ will plot stellar ages vs stellar mass
    xtit="$\\rm log_{10} (\\rm M_{\\star, tot}/M_{\odot})$"
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
  
          #from SUBFIND
          ind = np.where((mstot > 0) & (sn == s) & (age > 0))
          rplot = bin_it(x= np.log10(mstot[ind]), y = np.log10(age[ind]))
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='EAGLE L25')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #VR
          ind = np.where(AgeSMstar[idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = AgeSMstar[idx,0,ind]
          errdn = AgeSMstar[idx,1,ind]
          errup = AgeSMstar[idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "starage_mstar_z.pdf")


def prepare_data(hdf5_data, j, histmtot, histm30, histmgas, SFRMstar, R50Mstar, SigmaMstar,  histmall, MBHMstar, R50Mstar30, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar):

    # Unpack data
    (ms30, mstot, mg30, r50stot, sfrtot, mtot, mbh, r50s30, vs30, zgas, zgas_sf, zstar, age_star) = hdf5_data
    bin_it = functools.partial(us.wmedians, xbins=xmf)
    print age_star
    ind = np.where(mstot > 0)
    H, bins_edges = np.histogram(np.log10(mstot[ind]) + 10.0,bins=np.append(mbins,mupp))
    histmtot[j,:] = histmtot[j,:] + H
    ind = np.where(ms30 > 0)
    H, bins_edges = np.histogram(np.log10(ms30[ind]) + 10.0,bins=np.append(mbins,mupp))
    histm30[j,:] = histm30[j,:] + H
    ind = np.where(mg30 > 0)
    H, bins_edges = np.histogram(np.log10(mg30[ind]) + 10.0,bins=np.append(mbins,mupp))
    histmgas[j,:] = histmgas[j,:] + H
    ind = np.where(mtot > 0)
    H, bins_edges = np.histogram(np.log10(mtot[ind]) + 10.0,bins=np.append(mbins,mupp))
    histmall[j,:] = histmall[j,:] + H

    ind = np.where((mstot > 0) & (sfrtot > 0))
    SFRMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(sfrtot[ind]))
    ind = np.where((mstot > 0) & (r50stot > 0))
    R50Mstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(r50stot[ind]))

    ind = np.where((mstot > 0) & (mbh > 0))
    MBHMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(mbh[ind]) + 10.0)

    ind = np.where((ms30 > 0) & (r50s30 > 0))
    R50Mstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(r50s30[ind]))

    ind = np.where((ms30 > 0) & (vs30 > 0))
    SigmaMstar30[j,:] = bin_it(x=np.log10(ms30[ind]) + 10.0, y=np.log10(vs30[ind]/1.73205080757))

    ind = np.where((mstot > 0) & (zstar > 0))
    ZstarMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(zstar[ind]))
    ind = np.where((mstot > 0) & (age_star > 0))
    AgeSMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(age_star[ind]/1e9))
    ind = np.where((mstot > 0) & (zgas_sf > 0))
    ZSFMstar[j,:] = bin_it(x=np.log10(mstot[ind]) + 10.0, y=np.log10(zgas_sf[ind]))

def main():

    model_dir = '/fred/oz009/clagos/vr-testing-outputs/hydro/'
    output_dir = '/fred/oz009/clagos/vr-testing-outputs/Plots/hydro/'
    obs_dir = '/home/clagos/scm/analysis/VR'
    subvols = [0,1,2]
    snap = [28, 19, 15]
    vol = 25.0**3.0 #Mpc
    vol_eagle = 25.0**3.0

    name_file = 'fourth-try-6dfofsubhalos'#:'third-try'

    plt = common.load_matplotlib()
    fields = ['Aperture_mass_star_30_kpc','M_star','Aperture_mass_gas_30_kpc','R_HalfMass_star','SFR_gas','Aperture_mass_30_kpc','M_bh','Aperture_rhalfmass_star_30_kpc','Aperture_veldisp_star_30_kpc', 'Zmet_gas', 'Zmet_gas_sf','Zmet_star','tage_star']

    # Create histogram for mass functions
    histmtot = np.zeros(shape = (len(snap),len(mbins)))
    histm30 = np.zeros(shape = (len(snap),len(mbins)))
    histmgas = np.zeros(shape = (len(snap),len(mbins)))
    histmall = np.zeros(shape = (len(snap),len(mbins)))

    # create matrices for several scaling relations
    SFRMstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    R50Mstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    SigmaMstar = np.zeros(shape = (len(snap),3,len(mbins)))
    MBHMstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    R50Mstar30 = np.zeros(shape = (len(snap),3,len(mbins)))
    SigmaMstar30= np.zeros(shape = (len(snap),3,len(mbins)))
    ZstarMstar = np.zeros(shape = (len(snap),3,len(mbins)))
    ZSFMstar   = np.zeros(shape = (len(snap),3,len(mbins)))
    AgeSMstar  = np.zeros(shape = (len(snap),3,len(mbins)))

    for j in range(0,len(snap)):
        hdf5_data = common.read_data(model_dir, fields, snap[j], subvols, name_file)
        prepare_data(hdf5_data, j, histmtot, histm30, histmgas, SFRMstar, R50Mstar, SigmaMstar, histmall, MBHMstar, R50Mstar30, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar)

    # Take logs for mass functions
    ind = np.where(histmtot > 0.)
    histmtot[ind] = np.log10(histmtot[ind]/vol/dm)

    ind = np.where(histm30 > 0.)
    histm30[ind] = np.log10(histm30[ind]/vol/dm)

    ind = np.where(histmgas > 0.)
    histmgas[ind] = np.log10(histmgas[ind]/vol/dm)

    ind = np.where(histmall > 0.)
    histmall[ind] = np.log10(histmall[ind]/vol/dm)

    #mass funcion plots
    plot_mf_z(plt, output_dir+name_file+'/', snap, vol_eagle, histmtot, histm30, histmgas, histmall)
    #scaling relation plots
    plot_scaling_z(plt, output_dir+name_file+'/', snap, SFRMstar, R50Mstar, R50Mstar30, MBHMstar, SigmaMstar30, ZstarMstar, ZSFMstar, AgeSMstar)

if __name__ == '__main__':
    main()

