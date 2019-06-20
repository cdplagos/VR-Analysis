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


def add_observations_to_plot(obsdir, fname, ax, marker, label, color='k', err_absolute=False):
    fname = '%s/Gas/%s' % (obsdir, fname)
    x, y, yerr_down, yerr_up = common.load_observation(obsdir, fname, (0, 1, 2, 3))
    common.errorbars(ax, x, y, yerr_down, yerr_up, color, marker, label=label, err_absolute=err_absolute)

def prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit):
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit)
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)
    #ax.text(xleg, yleg, 'z=0')

def plot_halomf_z(plt, outdir, obs_dir, snap, vol_eagle, hist, histsh):

    xtit="$\\rm log_{10} (\\rm M_{\\rm halo}/M_{\odot})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\rm halo}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 8, 14, -6, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, subgn, mfofs, m200s,  m200sm = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 1, 3, 4, 6])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #HMF calc HMF calculated by Sheth & Tormen (2001)
          lmp, dp = common.load_observation(obs_dir, 'hmf_calc_z%01d.dat'% z, [0, 7])
          lmp_plot = np.log10(lmp) - np.log10(h0)
          dp_plot = np.log10(dp) + np.log10(pow(h0,3.))

          if(idx == 0):
             ax.plot(lmp_plot,dp_plot,'b', label = 'HMF SMT01')
          if(idx > 0):
             ax.plot(lmp_plot,dp_plot,'b')

          #HMF from SUBFIND
          ind = np.where((mfofs > 0) & (subgn == 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(mfofs[ind]),bins=np.append(mbins,mupp))
          histsfof = H
          ind = np.where((m200s > 0) & (subgn == 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(m200s[ind]),bins=np.append(mbins,mupp))
          hists200 =  H
          ind = np.where((m200sm > 0) & (subgn == 0) & (sn == s))
          H, bins_edges = np.histogram(np.log10(m200sm[ind]),bins=np.append(mbins,mupp))
          hists200m =  H 

          y = histsfof[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g', label ='EAGLE L25 SF $M_{\\rm FOF}$')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g')
  
          y = hists200[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dashed', label ='SF $M_{200\,crit}$')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dashed')
 
          y = hists200m[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dotted', label ='SF $M_{200\,mean}$')
          if idx > 0:
              ax.plot(xmf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dotted')
          #Predicted HMF
          y = hist[0,idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed', label ='VR $M_{200\,crit}$')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dashed')
          y = histsh[idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', label ='VR $M_{\\rm FOF}$')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r')
 
          y = hist[1,idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dotted', label ='VR $M_{200\,mean}$')
          if idx > 0:
              ax.plot(xmf[ind],y[ind],'r', linestyle='dotted') 
          if idx == 0:
              common.prepare_legend(ax, ['b','g','g','g','r','r','r'])

    common.savefig(outdir, fig, "halomf_z.pdf")

def plot_r200_z(plt, outdir, r200, snap):

    bin_it = functools.partial(us.wmedians, xbins=xmf)

    xtit="$\\rm log_{10} (\\rm M_{\\rm 200crit}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 200crit}/pMpc)$"
    xmin, xmax, ymin, ymax = 8, 14, -3, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, subgn, m200s, r200s = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 1, 4, 5])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #HMF from SUBFIND
          ind = np.where((m200s > 0) & (subgn == 0) & (sn == s) & (r200s > 0))
          rplot = bin_it(x= np.log10(m200s[ind]), y = np.log10(r200s[ind])-3.0)
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='SF')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #Predicted HMF
          ind = np.where(r200[0,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = r200[0,idx,0,ind]
          errdn = r200[0,idx,1,ind]
          errup = r200[0,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "r200M200_crit_z.pdf")

    xtit="$\\rm log_{10} (\\rm M_{\\rm 200mean}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm 200mean}/pMpc)$"
    xmin, xmax, ymin, ymax = 8, 14, -3, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, subgn, m200s, r200s = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 1, 6, 7])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #HMF from SUBFIND
          ind = np.where((m200s > 0) & (subgn == 0) & (sn == s) & (r200s > 0))
          rplot = bin_it(x= np.log10(m200s[ind]), y = np.log10(r200s[ind])-3.0)
 
          ind = np.where(rplot[0,:] != 0.)
          xplot = xmf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='SF')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          #Predicted HMF
          ind = np.where(r200[1,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = r200[1,idx,0,ind]
          errdn = r200[1,idx,1,ind]
          errup = r200[1,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          if idx == 0:
              common.prepare_legend(ax, ['b','r'])

    common.savefig(outdir, fig, "r200M200_mean_z.pdf")

def plot_rivmax_vmax_z(plt, outdir, rvmax, hist, snap, vol_eagle):

    bin_it = functools.partial(us.wmedians, xbins=xvf)

    xtit="$\\rm log_{10} (\\rm V_{\\rm max}/km s^{-1})$"
    ytit="$\\rm log_{10}(\\rm R_{\\rm V_{max}}/pMpc)$"
    xmin, xmax, ymin, ymax = 1, 3.2, -3, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))

    idx = [0,1,2]

    zins = [0, 1, 2]
    subplots = [311, 312, 313]

    sn, subgn, rvs, vs = common.load_observation('/fred/oz009/clagos/EAGLE/L0025N0376/REFERENCE/data/', 'SUBFIND-EAGLE-DATABASE.data', [2, 1, 18, 19])

    for subplot, idx, z, s in zip(subplots, idx, zins, snap):
          ax = fig.add_subplot(subplot)
          if (idx == 2):
              xtitplot = xtit
          else:
              xtitplot = ' '
          common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytit, locators=(0.1, 1, 0.1))
          ax.text(xleg,yleg, 'z=%s' % (str(z)))
  
          #HMF from SUBFIND
          ind = np.where((rvs > 0) & (subgn == 0) & (sn == s) & (vs> 0))
          rplot = bin_it(x= np.log10(vs[ind]), y = np.log10(rvs[ind])-3.0)
 
          ind = np.where((rvs > 0) & (subgn > 0) & (sn == s) & (vs> 0))
          rplotsubh = bin_it(x= np.log10(vs[ind]), y = np.log10(rvs[ind])-3.0)

          ind = np.where(rplot[0,:] != 0.)
          xplot = xvf[ind]
          yplot = rplot[0,ind]
          errdn = rplot[1,ind]
          errup = rplot[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='solid', label ='SF hosts')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='solid')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='b', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='b', alpha=0.2,interpolate=True)

          ind = np.where(rplotsubh[0,:] != 0.)
          xplot = xvf[ind]
          yplot = rplotsubh[0,ind]
          errdn = rplotsubh[1,ind]
          errup = rplotsubh[2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed', label ='SF subh')
          if idx > 0:
              ax.plot(xplot,yplot[0],'b', linestyle='dashed')

          #Predicted HMF
          ind = np.where(rvmax[0,idx,0,:] != 0.)
          xplot = xvf[ind]
          yplot = rvmax[0,idx,0,ind]
          errdn = rvmax[0,idx,1,ind]
          errup = rvmax[0,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR hosts')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          ind = np.where(rvmax[1,idx,0,:] != 0.)
          xplot = xvf[ind]
          yplot = rvmax[1,idx,0,ind]
          errdn = rvmax[1,idx,1,ind]
          errup = rvmax[1,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed', label ='VR subh')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed')

          if idx == 0:
              common.prepare_legend(ax, ['b','b','r','r'], loc='upper left')

    common.savefig(outdir, fig, "rvmax_vmax_z.pdf")

    xtit="$\\rm log_{10} (\\rm V_{\\rm max}/km s^{-1})$"
    ytit="$\\rm log_{10}(\Phi/dlog{\\rm M_{\\rm halo}}/{\\rm Mpc}^{-3} )$"
    xmin, xmax, ymin, ymax = 1, 3.2, -6, 1
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
  
          #HMF from SUBFIND
          ind = np.where((rvs > 0) & (subgn == 0) & (sn == s) & (vs> 0))
          H, bins_edges = np.histogram(np.log10(vs[ind]),bins=np.append(vbins,vupp))
          histsvh = H
          ind = np.where((rvs > 0) & (subgn > 0) & (sn == s) & (vs> 0))
          H, bins_edges = np.histogram(np.log10(vs[ind]),bins=np.append(vbins,vupp))
          histsvsubh =  H
  
          y = histsvh[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xvf[ind],np.log10(y[ind]/vol_eagle/dm),'g', label ='EAGLE L25 SF hosts')
          if idx > 0:
              ax.plot(xvf[ind],np.log10(y[ind]/vol_eagle/dm),'g')
  
          y = histsvsubh[:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xvf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dashed', label ='SF subh')
          if idx > 0:
              ax.plot(xvf[ind],np.log10(y[ind]/vol_eagle/dm),'g', linestyle='dashed')
 
          #Predicted HMF
          y = hist[0,idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xvf[ind],y[ind],'r', linestyle='solid', label ='VR hosts')
          if idx > 0:
              ax.plot(xvf[ind],y[ind],'r', linestyle='solid')
          y = hist[1,idx,:]
          ind = np.where(y != 0.)
          if idx == 0:
              ax.plot(xvf[ind],y[ind],'r',  linestyle='dashed', label ='VR subh')
          if idx > 0:
              ax.plot(xvf[ind],y[ind],'r',  linestyle='dashed')
  
          if idx == 0:
              common.prepare_legend(ax, ['g','g','r','r'])

    common.savefig(outdir, fig, "vmaxf_z.pdf")

def plot_con_lambda(plt, outdir, con, lam, snap):

    xtit="$\\rm log_{10} (\\rm M_{\\rm tot}/M_{\odot})$"
    ytit="$\\rm concentration$"
    xmin, xmax, ymin, ymax = 8, 14, 0.5, 30
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
  
          #Predicted HMF
          ind = np.where(con[0,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = con[0,idx,0,ind]
          errdn = con[0,idx,1,ind]
          errup = con[0,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR hosts')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          ind = np.where(con[1,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = con[1,idx,0,ind]
          errdn = con[1,idx,1,ind]
          errup = con[1,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed', label ='VR subh')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed')

          if idx == 0:
              common.prepare_legend(ax, ['r','r'], loc='upper left')

    common.savefig(outdir, fig, "con_mtot_z.pdf")

    xtit="$\\rm log_{10} (\\rm M_{\\rm tot}/M_{\odot})$"
    ytit="$\lambda_{\\rm Bullock}$"
    xmin, xmax, ymin, ymax = 8, 14, 0.01, 1
    xleg = xmax - 0.2 * (xmax-xmin)
    yleg = ymax - 0.3 * (ymax-ymin)

    fig = plt.figure(figsize=(5,10))
    plt.subplots_adjust(left=0.15)
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
          ax.set_yscale("log")
 
          #Predicted HMF
          ind = np.where(lam[0,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = lam[0,idx,0,ind]
          errdn = lam[0,idx,1,ind]
          errup = lam[0,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', label ='VR hosts')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r')
          ax.fill_between(xplot,yplot[0],yplot[0]-errdn[0], facecolor='r', alpha=0.2,interpolate=True)
          ax.fill_between(xplot,yplot[0],yplot[0]+errup[0], facecolor='r', alpha=0.2,interpolate=True)

          ind = np.where(lam[1,idx,0,:] != 0.)
          xplot = xmf[ind]
          yplot = lam[1,idx,0,ind]
          errdn = lam[1,idx,1,ind]
          errup = lam[1,idx,2,ind]

          if idx == 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed', label ='VR subh')
          if idx > 0:
              ax.plot(xplot,yplot[0],'r', linestyle='dashed')

          if idx == 0:
              common.prepare_legend(ax, ['r','r'], loc='upper center')

    common.savefig(outdir, fig, "lambda_mtot_z.pdf")



def prepare_data(hdf5_data, j, hist, histsh, R200med, rvmaxmed, histvmax, cmed, lambdamed):

    # Unpack data
    (Mass_BN98, Mass_FOF, Mass_200crit, Mass_200mean, Mass_tot, R_200crit, R_200mean, hostHaloID, ID, vmax, rvmax, xh, yh, zh, con, lambdaB,npart) = hdf5_data

    mass   = np.zeros(shape = len(Mass_FOF))
    masssh = np.zeros(shape = len(Mass_FOF))
    massm  = np.zeros(shape = len(Mass_FOF))

    bin_it = functools.partial(us.wmedians, xbins=xmf)
    print max(vmax)
    ind = np.where(Mass_FOF > 0)
    masssh[ind] = np.log10(Mass_FOF[ind]) + 10.0
    H, bins_edges = np.histogram(masssh,bins=np.append(mbins,mupp))
    histsh[j,:] = histsh[j,:] + H

    ind = np.where((Mass_200crit > 0) & (hostHaloID < 0))
    mass[ind] = np.log10(Mass_200crit[ind]) + 10.0
    H, bins_edges = np.histogram(mass,bins=np.append(mbins,mupp))
    hist[0,j,:] = hist[0,j,:] + H
    R200med[0,j,:] = bin_it(x=mass[ind], y=np.log10(R_200crit[ind]))

    ind = np.where((Mass_200mean > 0) & (hostHaloID < 0))
    massm[ind] = np.log10(Mass_200mean[ind]) + 10.0
    H, bins_edges = np.histogram(massm,bins=np.append(mbins,mupp))
    hist[1,j,:] = hist[1,j,:] + H
    R200med[1,j,:] = bin_it(x=massm[ind], y=np.log10(R_200mean[ind]))

    ind = np.where((Mass_tot > 0) & (con >0) & (np.isfinite(con)) & (hostHaloID < 0))
    cmed[0,j,:] = bin_it(x=np.log10(Mass_tot[ind])+10.0, y=con[ind])
    ind = np.where((Mass_tot > 0) & (con >0) & (np.isfinite(con)) & (hostHaloID >= 0))
    cmed[1,j,:] = bin_it(x=np.log10(Mass_tot[ind])+10.0, y=con[ind])

    ind = np.where((Mass_tot > 0) & (lambdaB >0) & (hostHaloID < 0))
    lambdamed[0,j,:] = bin_it(x=np.log10(Mass_tot[ind])+10.0, y=lambdaB[ind])
    ind = np.where((Mass_tot > 0) & (lambdaB >0) & (hostHaloID >= 0))
    lambdamed[1,j,:] = bin_it(x=np.log10(Mass_tot[ind])+10.0, y=lambdaB[ind])

    ind = np.where((vmax > 0) & (rvmax > 0) & (hostHaloID < 0))
    bin_it = functools.partial(us.wmedians, xbins=xvf)
    rvmaxmed[0,j,:] = bin_it(x=np.log10(vmax[ind]), y=np.log10(rvmax[ind]))
    H, bins_edges = np.histogram(np.log10(vmax[ind]),bins=np.append(vbins,vupp))
    histvmax[0,j,:] = histvmax[0,j,:] + H

    ind = np.where((vmax > 0) & (rvmax > 0) & (hostHaloID >= 0))
    bin_it = functools.partial(us.wmedians, xbins=xvf)
    rvmaxmed[1,j,:] = bin_it(x=np.log10(vmax[ind]), y=np.log10(rvmax[ind]))
    H, bins_edges = np.histogram(np.log10(vmax[ind]),bins=np.append(vbins,vupp))
    histvmax[1,j,:] = histvmax[1,j,:] + H

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
    fields = ['Mass_BN98','Mass_FOF','Mass_200crit','Mass_200mean','Mass_tot','R_200crit','R_200mean','hostHaloID','ID','Vmax','Rmax','Xcmbp','Ycmbp','Zcmbp','cNFW','lambda_B','npart']

    # Create histogram
    hist = np.zeros(shape = (2,len(snap),len(mbins)))
    histsh = np.zeros(shape = (len(snap),len(mbins)))
    histvmax = np.zeros(shape = (2,len(snap),len(vbins)))

    R200med = np.zeros(shape = (2,len(snap),3,len(mbins)))
    rvmaxmed = np.zeros(shape = (2,len(snap),3,len(vbins)))
    cmed = np.zeros(shape = (2,len(snap),3,len(mbins)))
    lambdamed = np.zeros(shape = (2,len(snap),3,len(mbins)))

    for j in range(0,len(snap)):
        hdf5_data = common.read_data(model_dir, fields, snap[j], subvols, name_file)
        prepare_data(hdf5_data, j, hist, histsh, R200med, rvmaxmed, histvmax, cmed, lambdamed)

    # Take logs
    ind = np.where(hist > 0.)
    hist[ind] = np.log10(hist[ind]/vol/dm)

    ind = np.where(histsh > 0.)
    histsh[ind] = np.log10(histsh[ind]/vol/dm)

    ind = np.where(histvmax > 0.)
    histvmax[ind] = np.log10(histvmax[ind]/vol/dm)

    plot_halomf_z(plt, output_dir+name_file+'/', obs_dir, snap, vol_eagle, hist, histsh)
    plot_r200_z(plt, output_dir+name_file+'/', R200med, snap)
    plot_rivmax_vmax_z(plt, output_dir+name_file+'/', rvmaxmed, histvmax, snap, vol_eagle)
    plot_con_lambda(plt, output_dir+name_file+'/', cmed, lambdamed, snap)

if __name__ == '__main__':
    main()

