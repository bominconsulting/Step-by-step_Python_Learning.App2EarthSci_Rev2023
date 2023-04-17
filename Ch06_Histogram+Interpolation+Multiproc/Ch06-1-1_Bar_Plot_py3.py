"""
Read RMM Index from text file,
and display MJO strength distribution by MJO Phases.

1. (Vertical) Bar Plot
2. Step Plot (Lined Bar Plot)
3. Stacked Bar Plot

By Daeho Jin, 2020.03.04
---

Make compatible to object-oriented concept.

By Daeho Jin, 2023.01.30
"""
import sys
import os.path
import numpy as np
from datetime import date
import Ch06_common_functions as cf

def main():
    ### Parameters
    rmm_fname='./Data/rmm.74toRealtime.txt'
    tgt_dates=(date(2000,11,1),date(2020,3,31))
    ph2draw=[3,6] ## Display MJO Phases 3 and 6

    ### Read RMM data
    mons,pcs,phs=cf.read_rmm_text(rmm_fname,tgt_dates)
    print(pcs.shape) ## Check the dimension

    ### Filtering only for Nov-Mar
    tidx= np.logical_or(mons>=11,mons<=3)
    pcs=pcs[tidx,:]
    phs=phs[tidx]
    print(pcs.shape) ## Check the dimension after filtering

    ### Calculate strength from PCs
    strs= np.sqrt((pcs**2).sum(axis=1))
    print(strs.min(),strs.max()) ## Check the range of strength if any weird value exists

    ### Build histogram of strength by MJO phase
    bin_bounds=np.arange(0.,4.,0.5) ## Produce 7 histogram bins
    bin_bounds[-1]=9.9 ## Extend to extremely large values
    hists=[]
    for ph in range(1,9,1): ## For each MJO phase
        phidx= phs==ph
        hist= np.histogram(strs[phidx],bin_bounds)[0]
        hists.append(hist)
    hists=np.asarray(hists)
    print(hists.shape) ## Check the dimension of histogram results
    hists=hists/phs.shape[0]*100. ## Normalized by total population, now in percent

    ### Prepare data for plotting
    outdir= "./Pics/"
    fnout= outdir+"Code6-1-1_vertical_bar_example.png"
    suptit="Bar Plot Example with MJO RMM Index"
    pic_data= dict(hist_data=hists, ph2draw=ph2draw,
                   bin_bounds=bin_bounds,
                   suptit=suptit, fnout=fnout,
    )
    plot_main(pic_data)
    return

###---------
### Bar plot
###---------
### Plan to draw three axes
### 1. Usual(vertical) Bar plot
### 2. Lined Bar(Step) plot
### 3. Stacked Bar plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,AutoMinorLocator,FormatStrFormatter

def plot_main(pdata):
    hist_data= pdata['hist_data']
    bin_bounds= pdata['bin_bounds']
    ph2draw= pdata['ph2draw']
    abc, ai= 'abcdefghijklmn', 0
    ncol, nrow= 1, 3.3  ## The last panel would be larger.
    nbins= hist_data.shape[1]

    ###---
    fig= plt.figure()
    fig.set_size_inches(6,8.5)  ## (xsize,ysize)
    ### Page Title
    fig.suptitle(pdata['suptit'], fontsize=16, y=0.97, va='bottom') #stretch='semi-condensed'

    ### Parameters for subplot area
    left,right,top,bottom= 0.05, 0.95, 0.925, 0.05
    npnx,gapx,npny,gapy= ncol, 0.03, nrow, 0.09
    lx= (right-left-gapx*(npnx-1))/npnx
    ly= (top-bottom-gapy*(npny-1))/npny; ly2=ly*1.3
    ix,iy= left, top

    ###-- Top Panel: Compare strength distribution of phase 3 and 6
    ax1= fig.add_axes([ix,iy-ly,lx,ly])

    ### Parameters for this panel
    nvars= len(ph2draw)
    h_gap= 0.3
    wd= (1-h_gap)/nvars ## Width of bar
    xlocs= cf.bar_x_locator(wd, data_dim=[nvars, nbins])
    cc=['b','r'] ## Preset colors for each bars

    ### Draw Bar Plot
    for i,ph in enumerate(ph2draw):
        pbar= ax1.bar(xlocs[i], hist_data[ph-1,:],
                      width=wd, color=cc[i], alpha=0.8,
                      label='Phase{}'.format(ph),
        )

    ### Fine tuning and decorating
    y_range=[0,6]
    subtit='({}) MJO Phase{} vs. Phase{}'.format(abc[ai],*ph2draw); ai+=1
    bar_common(ax1, subtit, x_dim=nbins,
               xt_labs=bin_bounds, y_range=y_range )
    ax1.set_xlabel('MJO Strength', fontsize=11) #,labelpad=0)
    ax1.set_ylabel('Percent', fontsize=11)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.99,0.98),
               fontsize=10, framealpha=0.75 )
    iy=iy-ly-gapy

    ###-- Middle Panel: Lined Bar(Step) plot
    ax2=fig.add_axes([ix,iy-ly,lx,ly])

    ### Parameters for this panel
    xx= np.arange(nbins+2)-1  ## Need extra values for both sides

    ### Draw Lined Bar Plot
    for i,ph in enumerate(ph2draw):
        data1= np.pad(hist_data[ph-1,:],(1,1),
                      mode='constant', constant_values=0,
        )  ## Add zero values at both sides
        pbar2= ax2.step(xx, data1, where='mid',
                        color=cc[i], lw=2, alpha=0.8,
                        label='Phase{}'.format(ph),
        )

    ### Fine tuning and decorating
    subtit='({}) MJO Phase{} vs. Phase{}'.format(abc[ai],*ph2draw); ai+=1
    bar_common(ax2, subtit, x_dim=nbins,
               xt_labs=bin_bounds, y_range=y_range )
    ax2.set_xlabel('MJO Strength', fontsize=11) #,labelpad=0)
    ax2.set_ylabel('Percent', fontsize=11)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99,0.98),
               fontsize=10, framealpha=0.75 )
    iy=iy-ly-gapy

    ###-- Bottom Panel: Stacked bar for all phases
    ax3=fig.add_axes([ix,iy-ly2,lx,ly2])

    ### Parameters for this panel
    wd=0.7 ## Width of bar
    xlocs= cf.bar_x_locator(wd,data_dim=[1,nbins])

    ### Pick colors from existing colormap
    nph= hist_data.shape[0]
    #cm= plt.cm.get_cmap('nipy_spectral', nph+2) #nph*2+1)
    cm= mpl.colormaps['nipy_spectral'].resampled(nph+2)
    cm= cm(np.arange(nph+2)) ## +2 for excluding end colors
    cc= []
    for i in range(nph):
        #cc.append([tuple(cm[1+i*2,:-1]),])
        cc.append([tuple(cm[i+1,:-1]),])

    ### Draw stacked bar
    base=np.zeros([nbins,])  ## Need information of bar base
    for k in range(nph):
        pbar3= ax3.bar(xlocs[0], hist_data[k,:],
                       width=wd, bottom=base,
                       color=cc[k], alpha=0.9,
                       label='Ph{}'.format(k+1)
        )
        cf.write_val(ax3, values=hist_data[k,:],
                     xlocs=xlocs[0], ylocs=base+hist_data[k,:]/2.,
                     crt=3.5,
        )
        base+= hist_data[k,:] ## Update base of bar

    ### Fine tuning and decorating
    subtit='({}) Strength by MJO Phases'.format(abc[ai]); ai+=1
    bar_common(ax3, subtit, x_dim=nbins,
               xt_labs=bin_bounds, y_range=[0,31] )
    ax3.set_xlabel('MJO Strength', fontsize=11) #,labelpad=0)
    ax3.set_ylabel('Percent', fontsize=11)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01,1.),
               borderaxespad=0, fontsize=10 )

    ###--- Save or Show
    plt.savefig(pdata['fnout'], bbox_inches='tight', dpi=150)
    print(pdata['fnout'])
    #plt.show()
    return

def bar_common(ax,subtit, x_dim=10, xt_labs=[], y_range=[]):
    """
    Decorating Bar plot
    """
    ### Title
    ax.set_title(subtit, fontsize=13, ha='left', x=0.0)

    ### Axis Control
    xx= np.arange(x_dim+1)
    ax.set_xlim(xx[0]-0.6,xx[-2]+0.6)  ## Add space on both sides
    ax.set_xticks(xx-0.5)
    ax.set_xticklabels(xt_labs) #,rotation=35,ha='right')
    if len(y_range)==2:
        ax.set_ylim(y_range)

    ### Ticks and Grid
    ax.tick_params(axis='both',which='major',labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d%%"))  ## It's working
    ax.yaxis.set_major_formatter("{x:.0f}%")  ## It is also working on ver 3.3+
    if y_range[1]-y_range[0]<=5:
        ax.yaxis.set_major_locator(MultipleLocator(1))
    elif y_range[1]-y_range[0]<=10:
        ax.yaxis.set_major_locator(MultipleLocator(2))
    elif y_range[1]-y_range[0]<=30:
        ax.yaxis.set_major_locator(MultipleLocator(5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y',color='0.7', linestyle=':', linewidth=1)
    return

if __name__=="__main__":
    main()
